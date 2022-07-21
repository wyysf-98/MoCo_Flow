import os
import os.path as osp
import torch
import argparse
import collections
from tqdm import tqdm
from collections import OrderedDict
from trainer import get_trainer
from utils.base_utils import inf_loop, merge_dict
from utils.parse_config import ConfigParser

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def train(config):
    # create network and trainer
    trainer = get_trainer(config)
    
    # load from checkpoint if provided
    if config.resume:
        trainer.load_ckpt(config.resume)

    # create training clock
    clock = trainer.clock

    # iter runner
    trainer.val_loader = inf_loop(trainer.val_loader)
    num_gpu = torch.distributed.get_world_size() if trainer.dist else 1
    num_train = len(trainer.train_loader)
    num_epochs = config['trainer']['num_iters'] // (num_train * num_gpu) + 1
    train_pbar = tqdm(range(clock.step, config['trainer']['num_iters']))
    for e in range(clock.epoch, num_epochs + 1):
        if clock.step >= config['trainer']['num_iters']:
            trainer.visualize_batch()
            trainer.save_ckpt('final') 
            break

        # if use DDP mode, set train sampler every epoch
        if trainer.dist:
            trainer.train_sampler.set_epoch(e)

        # begin train iteration
        for b, data in enumerate(trainer.train_loader):
            # train step
            trainer.train_func(data)

            # validation step
            if clock.step % config['trainer']['val_every_n_step'] == 0:
                data = next(trainer.val_loader)
                trainer.val_func(data)
                trainer.visualize_batch()

            # set pbar
            train_pbar.update(num_gpu)
            train_pbar.set_description("Train EPOCH[{}/{}]".format(clock.epoch, num_epochs))
            vis_dict = merge_dict(trainer.losses, trainer.extra)
            train_pbar.set_postfix(OrderedDict({k: '%.4f'%v.item()
                                        for k, v in vis_dict.items()}))

            # save checkpoint
            if clock.step % config['trainer']['save_every_n_step'] == 0:
                trainer.save_ckpt()

            # update learning rate
            trainer.update_learning_rate(log_freq=config['trainer']['num_iters'] // 1000)
            
            # clock tick
            clock.tick(num_gpu)

        # clock tock
        clock.tock()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='training pipeline defination')
    args.add_argument('-m', '--mode', default='train', type=str, help='current mode.')
    args.add_argument('-c', '--config', default=None, type=str, required=True,
                      help='config file path. (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='file path to retore the checkpoint. (default: None)')
    args.add_argument('-n', '--job_name', default=None, type=str,
                      help='job name. If None, use current time stamp. (default: None)')
    args.add_argument('-s', '--seed', default=None,
                      help='random seed used. (default: None)')
    args.add_argument('-g', '--gpu', default='0', type=str,
                      help='use single gpu to train. (defalut: 0)')
    args.add_argument('-d', '--dist', action='store_true',
                      help='whether to use distribute training.')
    args.add_argument('--local_rank', default=-1, type=int,
                      help='node rank for distributed training. (default: -1)')

    config = ConfigParser.from_args(args)
    train(config)
