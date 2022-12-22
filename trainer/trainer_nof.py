import os
import os.path as osp
import imageio
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import mcubes
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.utils as vutils

from datasets import get_dataset
from models import get_model, get_loss
from models.metrics import psnr, ssim
from models.rendering import render_rays
from trainer.base import BaseTrainer
from utils.base_utils import set_requires_grad
from utils.vis_utils import visualize_depth

class NoFTrainer(BaseTrainer):
    def prepare_dataloader(self, data_config):
        self.train_dataset = get_dataset(data_config, 'train')
        self.train_sampler = DistributedSampler(self.train_dataset) if self.dist else None
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=data_config['batch_size'],
                                       num_workers=data_config['workers'],
                                       sampler=self.train_sampler,
                                       shuffle=False,
                                       pin_memory=True)
        self.num_frames = self.train_dataset.num_frames

        self.val_dataset = get_dataset(data_config, 'val')
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=1,
                                     num_workers=data_config['workers'],
                                     shuffle=False,
                                     pin_memory=True)

    def load_pretrained_model(self, model, model_name, pretrained_path):
        try:
            self.record_str('load pretrained model %s from %s'%(model_name, pretrained_path))
            weights = torch.load(pretrained_path, map_location=self.device)[model_name]
        except Exception as e:
            raise ValueError(
                "local model {} error! Please check the model.\n{}".format(pretrained_path, e))
        self.nets[model].load_state_dict(weights, strict=False)

    def load_pretrained_nof(self, model, model_config):
        if self.config['model']['pretrained_nof'] is not None:
            self.load_pretrained_model(model, model+'_net', self.config['model']['pretrained_nof'])
        else:
            self.record_str('NOT load pretrained NoF !!!')

    def build_model(self, model_config):
        ## NoF model
        # create nof embedding 
        self.nof_embedding_xyz = get_model(model_config['nof_embedding_xyz']) if model_config['nof_embedding_xyz'] is not None else None
        self.nof_embedding_ind = get_model(model_config['nof_embedding_ind']) if model_config['nof_embedding_ind'] is not None else None
        self.nof_embeddings = [self.nof_embedding_xyz, self.nof_embedding_ind]
        # create backward nof
        self.nets['bw_NoF'] = get_model(model_config['bw_NoF'])
        self.load_pretrained_nof('bw_NoF', model_config['bw_NoF'])
        # create forward nof
        self.nets['fw_NoF'] = get_model(model_config['fw_NoF'])
        self.load_pretrained_nof('fw_NoF', model_config['fw_NoF'])
        self.nof_models = [self.nets['bw_NoF'], self.nets['fw_NoF']]

        self.record_str('-----trainable network architecture-----')
        self.record_str(self.nets)

        # load pretrained model
        if self.config['model']['pretrained_path'] is not None:
            self.load_ckpt(self.config['model']['pretrained_path'], restore_clock=False, restore_optimizer=False)

        if self.dist:
            self.DDP_mode()
        else:
            self.CUDA_mode()
 
    def set_loss_function(self, loss_config):
        self.criterion_nof = get_loss(loss_config).to(self.device)

    def forward(self, xyz, ind, model='fw_NoF'):
        # Embed xyz
        if isinstance(self.nets[model], nn.parallel.DistributedDataParallel):
            in_channels_xyz = self.nets[model].module.in_channels_xyz
            extra_feat_type = self.nets[model].module.extra_feat_type
            extra_feat_dim = self.nets[model].module.extra_feat_dim
        else:
            in_channels_xyz = self.nets[model].in_channels_xyz
            extra_feat_type = self.nets[model].extra_feat_type
            extra_feat_dim = self.nets[model].extra_feat_dim

        xyz_embedded = torch.zeros((xyz.shape[0], in_channels_xyz)).to(self.device)
        xyz_embedded_ = self.nof_embedding_xyz(xyz)
        xyz_embedded[:, :xyz_embedded_.shape[1]] = xyz_embedded_
        input_nof = xyz_embedded

        # Embed image index
        if extra_feat_type == "ind":
            ind_embedded = torch.zeros((xyz.shape[0], extra_feat_dim)).to(self.device)
            ind = ind.unsqueeze(dim=0).repeat((xyz.shape[0], 1)).to(self.device).float() * 2 / self.num_frames - 1.0
            ind_embedded_ = self.nof_embedding_ind(ind)
            ind_embedded[:, :ind_embedded_.shape[1]] = ind_embedded_
            # concate xyz and image index
            input_nof = torch.cat([input_nof, ind_embedded], -1)

        # apply nof model
        output_xyz = self.nets[model](input_nof, xyz, ind)

        return output_xyz

    def _shared_step(self, idx, nof_data):
        inside_pts, outside_pts = nof_data
        pts = torch.cat([inside_pts, outside_pts], dim = 0)

        query_pts, cano_pts = pts[:, :3], pts[:, 3:]
        query_pts, cano_pts = query_pts.to(self.device).float(), cano_pts.to(self.device).float()
        
        query_pts_bw = self.forward(query_pts, idx, 'bw_NoF')
        self.losses['nof_bw'] = self.criterion_nof(query_pts_bw, cano_pts)
        nof_fw_pts = self.forward(cano_pts, idx, 'fw_NoF')
        self.losses['nof_fw'] = self.criterion_nof(nof_fw_pts, query_pts)

    def train_step(self, data):
        nof_data = self.train_dataset.get_frame_correspondence(data['idx'].squeeze(), \
                                                                num_sampled=self.config['model']['N_sampled'], \
                                                                device = self.device)

        results = self._shared_step(data['idx'], nof_data)


    def val_step(self, data):
        nof_data = self.val_dataset.get_frame_correspondence(data['idx'].squeeze(), \
                                                                num_sampled=10000, \
                                                                device = self.device)

        results = self._shared_step(data['idx'], nof_data)

    def visualize_batch(self, save_path=None):
        pass