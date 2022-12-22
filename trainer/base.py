import os
import os.path as osp
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from abc import abstractmethod
from tensorboardX import SummaryWriter
from utils.base_utils import TrainClock, WorklogLogger, merge_dict
from utils.optimizers import RAdam, Ranger

class BaseTrainer(object):
    """Base trainer that provides common training behavior.
        All customized trainer should be subclass of this class.
    """

    def __init__(self, config):
        """Init BaseTrainer Class."""
        self.config = config
        self.mode = config.mode
        self.log_dir = config.log_dir
        self.ckpts_dir = config.ckpts_dir
        self.nets = dict()
        self.losses = dict()
        self.extra = dict()
        self.optimizers = dict()
        self.schedulers = dict()
        self.clock = TrainClock()

        # init distribute training or single gpu training
        self.init_dist(config)

        # init local txt logger and tensorboard or comet logger
        self.init_logger(config)

        if self.mode == 'train':
            # get dataloader
            self.prepare_dataloader(config['dataloader'])

            # build network
            self.build_model(config['model'])

            # set loss function
            self.set_loss_function(config['loss'])

            # configure optimizers
            self.configure_optimizers(config['optimizer'], config['scheduler'])

    def master_process(func):
        """ decorator for master process """
        def wrapper(self, *args, **kwargs):
            if self.is_master:
                return func(self, *args, **kwargs)
        return wrapper

    @abstractmethod
    def prepare_dataloader(self, data_config):
        """prepare dataloader for training"""
        raise NotImplementedError

    @abstractmethod
    def build_model(self, model_config):
        """build networks for training"""
        raise NotImplementedError

    @abstractmethod
    def set_loss_function(self, loss_config):
        """set loss function used in training"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, data):
        """forward logic in network"""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, data):
        """one step of training"""
        raise NotImplementedError

    @abstractmethod
    def val_step(self, data):
        """one step of validation"""
        raise NotImplementedError

    @abstractmethod
    def visualize_batch(self):
        """visualize results"""
        raise NotImplementedError

    def init_dist(self, config):
        """init dist config. If config.dist is True, else use single gpu"""
        self.dist = config.dist
        self.gpu = config.gpu_id
        self.is_master = config.is_master
        self.local_rank = config.local_rank
        self.device = torch.device('cuda:%s'%self.gpu)
        torch.cuda.set_device(self.device)
    
        if self.dist:
            dist.init_process_group(backend='nccl')            
            self.world_size = dist.get_world_size()

    @master_process
    def init_logger(self, config):
        """Init logger. Default use tensorboard, and comet_ml is optional"""
        # create logger
        self.logger = WorklogLogger(self.log_dir / 'log.txt')
        if self.config.seed is not None:
            self.logger.put_line(f'random seed: {self.config.seed}')
        self.logger.put_line(f'save ckpt to {self.ckpts_dir}')
        self.record_str(config)

        # set tensorboard writer
        self.tb = SummaryWriter(
            os.path.join(self.log_dir, 'train.events'))

    def get_optimizer(self, optimizer_config, parameters):
        """set optimizer used in training"""
        eps = 1e-8
        if optimizer_config['type'] == 'sgd':
            optimizer = optim.SGD(
                parameters, lr=optimizer_config['lr'], momentum=optimizer_config['momentum'], weight_decay=optimizer_config['weight_decay'])
        elif optimizer_config['type'] == 'adam':
            optimizer = optim.Adam(
                parameters, lr=optimizer_config['lr'], eps=eps, weight_decay=optimizer_config['weight_decay'])
        elif optimizer_config['type'] == 'radam':
            optimizer = RAdam(
                parameters, lr=optimizer_config['lr'], eps=eps, weight_decay=optimizer_config['weight_decay'])
        elif optimizer_config['type'] == 'ranger':
            optimizer = Ranger(
                parameters, lr=optimizer_config['lr'], eps=eps, weight_decay=optimizer_config['weight_decay'])
        else:
            raise NotImplementedError(f"Optimizer type {optimizer_config['type']} not implemented yet !!!")

        return optimizer

    def get_scheduler(self, scheduler_config, optimizer):
        """set lr scheduler used in training"""
        eps = 1e-8
        if scheduler_config['type'] == 'steplr':
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=[int(step // self.world_size) for step in scheduler_config['decay_step']], gamma=scheduler_config['decay_gamma'])
        elif scheduler_config['type'] == 'explr':
            scheduler = lr_scheduler.ExponentialLR(
                optimizer, scheduler_config['lr_decay'])
        elif scheduler_config['type'] == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=scheduler_config['num_epochs'], eta_min=eps)
        elif scheduler_config['type'] == 'poly':
            scheduler = lr_scheduler.LambdaLR(
                optimizer, lambda epoch: (1-epoch/scheduler_config['num_epochs'])**scheduler_config['poly_exp'])
        else:
            raise NotImplementedError('Scheduler type {} not implemented yet !!!'.format(
                scheduler_config['type']))
        return scheduler

    def configure_optimizers(self, optimizer_config, scheduler_config):
        """configure optimizers used in training"""
        parameters = []
        for key in self.nets.keys():
            parameters += list(self.nets[key].parameters())

        optimizer = self.get_optimizer(optimizer_config, parameters)
        scheduler = self.get_scheduler(scheduler_config, optimizer)

        self.optimizers['base'] = optimizer
        self.schedulers['base'] = scheduler

    def update_learning_rate(self, log_freq=1, mute=False):
        """record and update learning rate"""
        def get_learning_rate(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']
        for key in self.optimizers.keys():
            if self.is_master and not mute and self.clock.step % log_freq==0:
                current_lr = get_learning_rate(self.optimizers[key])
                self.logger.put_line('[Epoch/Step : {}/{}]: <optimizer {}> learning rate is: {}'.format(
                    self.clock.epoch, self.clock.step, key, current_lr))
                self.tb.add_scalar('learning_rate/{}_lr'.format(key), current_lr, self.clock.step)

            self.schedulers[key].step()

    def update_network(self):
        """update network by back propagation"""
        for key in self.optimizers.keys():
            self.optimizers[key].zero_grad()

        total_loss = sum(self.losses.values())
        total_loss.backward()

        for key in self.optimizers.keys():
            self.optimizers[key].step()

    @master_process
    def record_losses(self, mode='train', mute=False):
        """record loss to tensorboard and comet_ml if use comel is True"""
        record_str = ''
        dict_recorded = merge_dict(self.losses, self.extra)
        dict_recorded['total'] = sum(self.losses.values())
        for k, v in dict_recorded.items():
            record_str += '{}: {:.8f} '.format(k, v.item())
            self.tb.add_scalar('{}_loss/{}'.format(mode, k), v.item(), self.clock.step)
        if not mute:
            self.logger.put_line(
                '{}: [Epoch/Step: {}/{}]: {}'.format(mode, self.clock.epoch, self.clock.step, record_str))

    @master_process
    def record_scalar(self, dict_recorded, mode=None, mute=True):
        """record scalar to tensorboard and comet_ml if use comel is True"""
        str_recorded = ''
        for k, v in dict_recorded.items():
            str_recorded += '{}: {:.8f} '.format(k, v.item())
            self.tb.add_scalar(k if mode is None else '{}/{}'.format(mode, k), v.item(), self.clock.step)
        if not mute:
            self.logger.put_line(
                '[Epoch/Step : {}/{}]: {}'.format(self.clock.epoch, self.clock.step, str_recorded))

    @master_process
    def record_str(self, str_recorded):
        """record string in master process"""
        print(str_recorded)
        self.logger.put_line(
            '[Epoch/Step : {}/{}]: {}'.format(self.clock.epoch, self.clock.step, str_recorded))

    def train_func(self, data):
        """training function"""
        self.train_mode()

        self.train_step(data)
        self.update_network()

        if self.clock.step % self.config['trainer']['log_freq']==0:
            self.record_losses('train')

    def val_func(self, data):
        """validation function"""
        self.eval_mode()

        with torch.no_grad():
            self.val_step(data)

        if self.clock.step % self.config['trainer']['log_freq']==0:
            self.record_losses('valid')

    def DDP_mode(self):
        """set all networks to DistributedDataParallel wrapper"""
        for key in self.nets.keys():
            self.nets[key] = nn.SyncBatchNorm.convert_sync_batchnorm(self.nets[key])
            self.nets[key] = nn.parallel.DistributedDataParallel(\
                self.nets[key].to(self.device), device_ids=[self.gpu], output_device=self.gpu, find_unused_parameters=True)

    def CUDA_mode(self):
        """set all networks to cuda device"""
        for key in self.nets.keys():
            self.nets[key] = self.nets[key].to(self.device)

    def train_mode(self):
        """set networks to train mode"""
        for key in self.nets.keys():
            if isinstance(self.nets[key], DistributedDataParallel):
                self.nets[key] = self.nets[key].module.train()
            else:
                self.nets[key] = self.nets[key].train()

    def eval_mode(self):
        """set networks to eval mode"""
        for key in self.nets.keys():
            if isinstance(self.nets[key], DistributedDataParallel):
                self.nets[key] = self.nets[key].module.eval()
            else:
                self.nets[key] = self.nets[key].eval()

    @master_process
    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(
                self.ckpts_dir, f"epoch{self.clock.epoch}_iter{self.clock.step}.pth")
            print(f"Saving checkpoint epoch {self.clock.epoch} iter {self.clock.step}...")
        else:
            save_path = os.path.join(self.ckpts_dir, f"{name}.pth")
        
        save_dict = dict()
        save_dict['clock'] = self.clock.make_checkpoint()
        for key in self.nets.keys():
            if isinstance(self.nets[key], DistributedDataParallel):
                save_dict[key+'_net'] = self.nets[key].module.state_dict()
            else:
                save_dict[key+'_net'] = self.nets[key].state_dict()
        for key in self.optimizers.keys():
            save_dict[key+'_optimizer'] = self.optimizers[key].state_dict()
            save_dict[key+'_scheduler'] = self.schedulers[key].state_dict()
        torch.save(save_dict, save_path)

    def load_ckpt(self, name=None, restore_clock=True, restore_optimizer=True):
        """load checkpoint from saved checkpoint"""
        load_path = name if str(name).endswith('.pth') else f'{str(name)}.pth'
        if not os.path.exists(load_path):
            raise ValueError(f"Checkpoint {load_path} not exists.")

        checkpoint = torch.load(load_path, map_location=self.device)
        print(f"Loading checkpoint from {load_path} ...")

        for key in self.nets.keys():
            if isinstance(self.nets[key], DistributedDataParallel):
                self.nets[key].module.load_state_dict(checkpoint[key+'_net'], strict=False)
            else:
                self.nets[key].load_state_dict(checkpoint[key+'_net'], strict=False)
        if restore_clock:
            self.clock.restore_checkpoint(checkpoint['clock'])
        if restore_optimizer:
            for key in self.optimizers.keys():
                if key+'_optimizer' not in checkpoint.keys():
                    self.record_str(key+'_optimizer not exist in checkpoint.')
                    continue
                self.optimizers[key].load_state_dict(checkpoint[key+'_optimizer'])
            for key in self.schedulers.keys():
                if key+'_scheduler' not in checkpoint.keys():
                    self.record_str(key+'_scheduler not exist in checkpoint.')
                    continue
                self.schedulers[key].load_state_dict(checkpoint[key+'_scheduler'])
