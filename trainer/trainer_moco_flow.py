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

class MoCoFlowTrainer(BaseTrainer):
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

        data_config['size'] = data_config['val_size']
        self.val_dataset = get_dataset(data_config, 'val')
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=1,
                                     num_workers=data_config['workers'],
                                     shuffle=False,
                                     pin_memory=True)

        self.spherical_dataset = get_dataset(data_config, 'val/spherical_path')

    def load_pretrained_model(self, model, model_name, pretrained_path):
        try:
            self.record_str('load pretrained model %s from %s'%(model_name, pretrained_path))
            weights = torch.load(pretrained_path, map_location=self.device)[model_name]
        except Exception as e:
            raise ValueError(
                "local model {} error! Please check the model.\n{}".format(pretrained_path, e))

        if 'NeRF' in model_name: # do not load nerf rgb branch
            weights = {k:v for k,v in weights.items() if 'xyz' in k or 'sigma' in k}

        self.nets[model].load_state_dict(weights, strict=False)

    def load_pretrained_nerf(self, model, model_config):
        if self.config['model']['pretrained_nerf'] is not None:
            # self.load_pretrained_model(model, model+'_net', self.config['model']['pretrained_nerf'])
            self.load_pretrained_model(model, 'fine_NeRF_net', self.config['model']['pretrained_nerf']) # TRICK. only load fine NeRF
        else:
            self.record_str('NOT loading pretrained NeRF !!!')

    def load_pretrained_nof(self, model, model_config):
        if self.config['model']['pretrained_nof'] is not None:
            self.load_pretrained_model(model, model+'_net', self.config['model']['pretrained_nof'])
        else:
            self.record_str('NOT load pretrained NoF !!!')

    def build_model(self, model_config):
        ## NeRF model
        # create nerf embedding
        self.nerf_embedding_xyz = get_model(model_config['nerf_embedding_xyz']) if model_config['nerf_embedding_xyz'] is not None else None
        self.nerf_embedding_ind = get_model(model_config['nerf_embedding_ind']) if model_config['nerf_embedding_ind'] is not None else None
        self.nerf_embedding_dir = get_model(model_config['nerf_embedding_dir']) if model_config['nerf_embedding_dir'] is not None else None
        self.nerf_embeddings = [self.nerf_embedding_xyz, self.nerf_embedding_ind, self.nerf_embedding_dir]
        # create nerf model
        self.nets['coarse_NeRF'] = get_model(model_config['coarse_NeRF'])
        self.load_pretrained_nerf('coarse_NeRF', model_config['coarse_NeRF'])
        self.nerf_models = [self.nets['coarse_NeRF']]
        self.N_importance = model_config['N_importance']
        if self.N_importance > 0:
            self.nets['fine_NeRF'] = get_model(model_config['fine_NeRF'])
            self.load_pretrained_nerf('fine_NeRF', model_config['fine_NeRF'])
            self.nerf_models.append(self.nets['fine_NeRF'])

        ## NoF model
        # create nof embedding 
        self.nof_embedding_xyz = get_model(model_config['nof_embedding_xyz']) if model_config['nof_embedding_xyz'] is not None else None
        self.nof_embedding_ind = get_model(model_config['nof_embedding_ind']) if model_config['nof_embedding_ind'] is not None else None
        self.nof_embeddings = [self.nof_embedding_xyz, self.nof_embedding_ind]
        # create backward nof
        self.nets['bw_NoF'] = get_model(model_config['bw_NoF'])
        self.load_pretrained_nof('bw_NoF', model_config['bw_NoF'])
        self.nof_models = [self.nets['bw_NoF']]
        # create forward nof
        if self.config['loss']['chain_local'] or self.config['loss']['chain_global']:
            self.nets['fw_NoF'] = get_model(model_config['fw_NoF'])
            self.load_pretrained_nof('fw_NoF', model_config['fw_NoF'])
            self.nof_models.append(self.nets['fw_NoF'])

        self.record_str('-----trainable network architecture-----')
        self.record_str(self.nets)

        # load pretrained model
        if self.config['model']['pretrained_path'] is not None:
            self.load_ckpt(self.config['model']['pretrained_path'], restore_clock=False, restore_optimizer=False)

        # coarse to fine
        if self.config['model']['coarse_to_fine']: # set init embbeding weights to 0
            self.nerf_embeddings[0].set_weights(0)
            self.nof_embeddings[0].set_weights(0)
        
        if self.dist:
            self.DDP_mode()
        else:
            self.CUDA_mode()
            
    def configure_optimizers(self, optimizer_config, scheduler_config):
        # configure all optimizer
        parameters = []
        for key in self.nets.keys():
            parameters += list(self.nets[key].parameters())
        moco_optimizer = self.get_optimizer(optimizer_config['moco'], parameters)
        self.optimizers['moco'] = moco_optimizer

        if self.config['loss']['chain_local'] or self.config['loss']['chain_global']:
            nof_optimizer = self.get_optimizer(optimizer_config['nof'], \
                list(self.nets['bw_NoF'].parameters()) + list(self.nets['fw_NoF'].parameters()))
        else:
            nof_optimizer = self.get_optimizer(optimizer_config['nof'], self.nets['bw_NoF'].parameters())
        self.optimizers['nof'] = nof_optimizer

        moco_scheduler = self.get_scheduler(scheduler_config, moco_optimizer)
        nof_scheduler = self.get_scheduler(scheduler_config, nof_optimizer)
        self.schedulers['moco'] = moco_scheduler
        self.schedulers['nof'] = nof_scheduler

    def set_loss_function(self, loss_config):
        self.criterion_img = get_loss(loss_config['img_loss']).to(self.device)
        self.criterion_nof = get_loss(loss_config['nof_loss']).to(self.device)
        self.criterion_msk = get_loss(loss_config['msk_loss']).to(self.device)

    def forwarf_nerf(self, xyz, deltas, model='coarse_NeRF'):
        # Embed xyz
        xyz_embedded = torch.zeros((xyz.shape[0], self.nets[model].in_channels_xyz)).to(self.device)
        xyz_embedded_ = self.nerf_embedding_xyz(xyz)
        xyz_embedded[:, :xyz_embedded_.shape[1]] = xyz_embedded_

        # apply nerf model
        sigmas = self.nets[model](xyz_embedded, sigma_only=True)
        alphas = 1 - torch.exp(-deltas * torch.nn.Softplus()(sigmas)) # trick: use softplus
        # alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))

        return alphas

    def forward_nof(self, xyz, ind, model='fw_NoF'):
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

    def forward(self, rays, background, use_nof=True, test_time=False):

        """
        rays: (N, 3), rendered rays.
        background: (N, 3), background color for each ray.
        test_time: In test mode, use some tricks to speed the inference.
        """
        B = rays.shape[0] # Do batched inference on rays using chunk.
        results = defaultdict(list)
        chunk = self.config['model']['chunk']
        for i in range(0, B, chunk):
            rendered_ray_chunks = \
                render_rays(rays[i:i+chunk].to(self.device), # move rays to gpu
                            background[i:i+chunk].to(self.device), # move background to gpu
                            self.nerf_embeddings,
                            self.nerf_models,
                            nof_embeddings=self.nof_embeddings if use_nof else None,
                            nof_models=self.nof_models if use_nof else None,
                            chain_local=self.config['loss']['chain_local'] if use_nof else False,
                            chain_global=self.config['loss']['chain_global'] if use_nof else False,
                            N_samples=self.config['model']['N_samples'],
                            N_importance=self.config['model']['N_importance'],
                            use_disp=self.config['model']['use_disp'],
                            perturb=self.config['model']['perturb'],
                            noise_std=self.config['model']['noise_std'],
                            nerf_activate_type=self.config['model']['nerf_activate_type'],
                            test_time=test_time,
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        
        return results
    
    def render(self, rays, background, rays_msk=None, use_nof=True, test_time=False):
        N_rand = self.config['model']['N_rand']

        if rays_msk is not None:
            msk = np.where(rays_msk==True)
            rendered_rays = rays[msk]
            rendered_background = background[msk]
        else:
            rendered_rays = rays
            rendered_background = background

        B = rays.shape[0] # Do batched inference on rays using N_rand.
        results = defaultdict(list)
        for i in range(0, B, N_rand):
            results_chunk = self.forward(rendered_rays[i:i+N_rand], 
                                         background=rendered_background[i:i+N_rand] if background is not None else None, 
                                         use_nof=use_nof,
                                         test_time=test_time) 
            for k, v in results_chunk.items():
                results[k] += [v]
        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        if rays_msk is not None:
            typ = 'fine' if 'rgb_fine' in results else 'coarse'

            num_ori_rays = rays.shape[0]
            img_raw = torch.zeros(num_ori_rays, 3).to(self.device)
            depth_raw = torch.ones(num_ori_rays).to(self.device)*10
            opacity = results['opacity_%s'%typ].cpu().numpy()

            foreground_idx = np.where(opacity>0)
            foreground_mask = np.zeros_like(rays_msk).astype(np.float)
            foreground_mask[msk] = opacity
            img_raw[foreground_mask>0] = results['rgb_%s'%typ][foreground_idx]
            depth_raw[msk] = 8
            depth_raw[foreground_mask>0] = results['depth_%s'%typ][foreground_idx]
            img_raw[foreground_mask==0] = background[foreground_mask==0].to(self.device)

            results['rgb_%s'%typ] = img_raw
            results['depth_%s'%typ] = depth_raw

        return results

    def increase_xyzemb_dim(self, increase_nerf_xyzemb=True, increase_nof_xyzemb=True):
        if self.config['model']['coarse_to_fine']:
            start_iter = self.config['trainer']['coarse2fine_start_iter']
            end_iter = self.config['trainer']['coarse2fine_end_iter']
            
            if self.clock.step > start_iter and self.clock.step <= end_iter:
                n_iters = end_iter - start_iter
                cur_iter = self.clock.step - start_iter
                if increase_nerf_xyzemb:
                    # for nerf embedding_xyz
                    nerf_n_freqs = self.config['model']['nerf_embedding_xyz']['N_freqs']
                    if nerf_n_freqs != 0:
                        nerf_delta_freq = n_iters // nerf_n_freqs
                        nerf_cur_freq = cur_iter // nerf_delta_freq
                        nerf_cur_freq_weight = cur_iter / nerf_delta_freq - nerf_cur_freq
                        nerf_weights = [0] * (nerf_n_freqs+1)
                        nerf_weights[:nerf_cur_freq] = [1] * nerf_cur_freq
                        nerf_weights[nerf_cur_freq] = nerf_cur_freq_weight
                        nerf_weights = nerf_weights[:nerf_n_freqs]
                        self.nerf_embeddings[0].weights = nerf_weights
                if increase_nof_xyzemb:
                    # for nof embedding_xyz
                    nof_n_freqs = self.config['model']['nof_embedding_xyz']['N_freqs']
                    if nof_n_freqs != 0:
                        nof_delta_freq = n_iters // nof_n_freqs
                        nof_cur_freq = cur_iter // nof_delta_freq
                        nof_cur_freq_weight = cur_iter / nof_delta_freq - nof_cur_freq
                        nof_weights = [0] * (nof_n_freqs+1)
                        nof_weights[:nof_cur_freq] = [1] * nof_cur_freq
                        nof_weights[nof_cur_freq] = nof_cur_freq_weight
                        nof_weights = nof_weights[:nof_n_freqs]
                        self.nof_embeddings[0].weights = nof_weights
                    
            elif self.clock.step > end_iter:
                self.nerf_embeddings[0].weights = [1] * self.config['model']['nerf_embedding_xyz']['N_freqs']
                self.nof_embeddings[0].weights = [1] * self.config['model']['nof_embedding_xyz']['N_freqs']

    def _shared_step(self, idx, rays, background, rgbs, nof_data):
        if self.config['loss']['chain_global']: # if chain global, random choose a frame
            chain_idx = np.random.randint(self.num_frames) * 2 / self.num_frames - 1.0
            rays = torch.cat([rays, 
                              chain_idx * torch.ones_like(rays[:,:1])],
                              dim = 1) # N_rand, 10

        results = self.forward(rays, 
                              background=background, 
                              use_nof=True)
        self.losses['img_loss'] = self.criterion_img(results, rgbs.to(self.device)) * self.config['loss']['img_loss']['weight']
        if self.config['loss']['chain_local']:
            nof_local = torch.mean(results['nof_local_disp_coarse'])
            if 'nof_local_disp_fine' in results.keys():
                nof_local += torch.mean(results['nof_local_disp_fine'])
            self.losses['nof_local'] = nof_local * self.config['loss']['nof_local_weight']

        if self.config['loss']['chain_global']:
            nof_global = torch.mean(results['nof_global_disp_coarse'])
            if 'nof_global_disp_fine' in results.keys():
                nof_global += torch.mean(results['nof_global_disp_fine'])
            self.losses['nof_global'] = nof_global * self.config['loss']['nof_global_weight']
            
        if nof_data is not None: # train NoF model
            inside_pts, outside_pts = nof_data
            inside_query_pts, inside_cano_pts = inside_pts[:, :3], inside_pts[:, 3:]
            inside_query_pts, inside_cano_pts = inside_query_pts.to(self.device).float(), inside_cano_pts.to(self.device).float()
            
            # for points which are near surface
            inside_query_pts_bw = self.forward_nof(inside_query_pts, idx, 'bw_NoF')
            self.losses['nof_bw'] = self.criterion_nof(inside_query_pts_bw, inside_cano_pts) * self.config['loss']['nof_loss']['weight']
            if self.config['loss']['chain_local'] or self.config['loss']['chain_global']:
                nof_fw_inside_pts = self.forward_nof(inside_cano_pts, idx, 'fw_NoF')
                self.losses['nof_fw'] = self.criterion_nof(nof_fw_inside_pts, inside_query_pts) * self.config['loss']['nof_loss']['weight']
            
            if self.config['loss']['msk_loss']['weight'] > 0:
                # # only train alphas mask when fix nerf
                # if self.clock.step < self.config['trainer']['fix_nerf_end_iter']:
                # for outside points
                outside_query_pts, outside_cano_pts = outside_pts[:, :3], outside_pts[:, 3:]
                outside_query_pts, outside_cano_pts = outside_query_pts.to(self.device).float(), outside_cano_pts.to(self.device).float()
                nof_bw_outside_pts = self.forward_nof(outside_query_pts, idx, 'bw_NoF')
                ## enforce the background density to 0
                coarse_deltas = 1 / self.config['model']['N_samples']
                fine_deltas = 1 / (self.config['model']['N_samples'] + self.config['model']['N_importance'])
                coarse_nerf_outside_alphas = self.forwarf_nerf(nof_bw_outside_pts, coarse_deltas, 'coarse_NeRF')
                if self.N_importance > 0:
                    fine_nerf_outside_alphas = self.forwarf_nerf(nof_bw_outside_pts, fine_deltas, 'fine_NeRF')
                    outside_alphas = torch.cat([coarse_nerf_outside_alphas, fine_nerf_outside_alphas], dim=0)
                else:
                    outside_alphas = coarse_nerf_outside_alphas
                self.losses['alphas_mask'] = self.criterion_msk(outside_alphas, torch.zeros_like(outside_alphas)) * \
                                                self.config['loss']['msk_loss'] ['weight']
            else:
                for key in list(self.losses.keys()):
                    if key in ['alphas_mask']:
                        del self.losses[key]
                torch.cuda.empty_cache()
        else:
            for key in list(self.losses.keys()):
                if key in ['nof_bw', 'nof_fw', 'alphas_mask']:
                    del self.losses[key]
                torch.cuda.empty_cache()
            if 'nof' in self.optimizers.keys():
                del self.optimizers['nof']
            if 'nof' in self.schedulers.keys():
                del self.schedulers['nof']

        return results

    def train_step(self, data):
        # increase xyz_embedding dim
        self.increase_xyzemb_dim()
        if self.is_master:
            if self.nerf_embeddings[0].N_freqs != 0:
                self.tb.add_histogram('nerf_xyzemb_weights', np.array(self.nerf_embeddings[0].weights), self.clock.step)
            if self.nof_embeddings[0].N_freqs != 0:
                self.tb.add_histogram('nof_xyzemb_weights', np.array(self.nof_embeddings[0].weights), self.clock.step)

       # fix NeRF density branch before coarse2fine stage
        if self.clock.step <= self.config['trainer']['coarse2fine_start_iter']:
            self.record_scalar({'state/nerf': torch.Tensor([0])})
            set_requires_grad(self.nerf_models, False)
            # set gradient to True in rgb branch
            set_requires_grad([self.nerf_models[0].rgb, 
                               self.nerf_models[0].xyz_encoding_final, 
                               self.nerf_models[0].extra_encoding], True)
            if self.N_importance > 0:
                set_requires_grad([self.nerf_models[1].rgb, 
                                   self.nerf_models[1].xyz_encoding_final, 
                                   self.nerf_models[1].extra_encoding], True)
        else:
            self.record_scalar({'state/nerf': torch.Tensor([1])})
            set_requires_grad(self.nerf_models, True)

        # get data
        idx, rays, rays_msk, rgbs, background = data['idx'], data['rays'], data['rays_msk'], data['rgbs'], data['background']
        batch_size, num_rays = data['rays'].shape[:2]
        assert batch_size == 1 # assert batch size is 1, iter mode
        rays, rays_msk, rgbs, background = \
            rays.squeeze(0), rays_msk.squeeze(0), rgbs.squeeze(0), background.squeeze(0)

        # select rays and rgbs from valid rays mask
        val_inds = torch.nonzero(rays_msk).squeeze(1)
        sel_inds = val_inds[torch.randperm(val_inds.shape[0])[:self.config['model']['N_rand']]]
        sampled_rays, sampled_rgbs, sampled_background = rays[sel_inds], rgbs[sel_inds], background[sel_inds]
        sampled_rgbs = sampled_rgbs.to(self.device)

        # train NoF before coarse2fine stage
        if self.clock.step < self.config['trainer']['coarse2fine_start_iter']:
            nof_data = self.train_dataset.get_frame_correspondence(idx.squeeze(), \
                                                                   num_sampled=self.config['model']['N_sampled'], \
                                                                   device=self.device)
        else:
            nof_data = None
        
        results = self._shared_step(idx, 
                                    sampled_rays, 
                                    sampled_background, 
                                    sampled_rgbs, 
                                    nof_data
                                    )

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], sampled_rgbs)
            self.extra['train_psnr'] = psnr_

    def val_step(self, data):
        idx, rays, rays_msk, rgbs, background = data['idx'], data['rays'], data['rays_msk'], data['rgbs'], data['background']
        rays, rays_msk, rgbs, background = \
            rays.squeeze(dim=0), rays_msk.squeeze(0), rgbs.squeeze(dim=0), background.squeeze(dim=0)
        rgbs = rgbs.to(self.device)

        results = self._shared_step(idx,
                                    rays,
                                    background, 
                                    rgbs,
                                    nof_data=None)

        img_size = self.val_dataset.size
        img_ori, img_pred, _, _ = self.decode_results(results, img_size)
        psnr_ = psnr(img_ori, rgbs)
        img_pred = img_pred.unsqueeze(dim=0)
        img_gt = rgbs.view(img_size[0], img_size[1], 3).permute(2, 0, 1).unsqueeze(dim=0) # (1, 3, W, H)
        ssim_ = ssim(img_pred, img_gt)

        self.extra['val_psnr'] = psnr_
        self.extra['ssim'] = ssim_

    def decode_results(self, results, img_size):
        H, W = img_size
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        img_ori = results['rgb_%s'%typ]
        img_pred = img_ori.view(H, W, 3).permute(2, 0, 1) # (3, H, W)
        depth_ori = results['depth_%s'%typ]
        depth_pred = visualize_depth(depth_ori.view(H, W)) # (3, H, W)
        return img_ori, img_pred, depth_ori, depth_pred

    @ torch.no_grad()
    def visualize_mesh(self, frame_idx, N_grid=256, sigma_threshold=10, chunk=10000, save_path=None, save_tb=False):
        x_range = [-1.5, 1.5]
        y_range = [-1.5, 1.5]
        z_range = [-1.5, 1.5]

        xmin, xmax = x_range
        ymin, ymax = y_range
        zmin, zmax = z_range
        x = np.linspace(xmin, xmax, N_grid)
        y = np.linspace(ymin, ymax, N_grid)
        z = np.linspace(zmin, zmax, N_grid)
        xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).to(self.device)

        # start  predicting
        print('Predicting occupancy ...')
        with torch.no_grad():
            B = xyz_.shape[0]
            out_chunks = []
            for i in range(0, B, chunk):
                # Embed positions by chunk
                xyz_input = xyz_[i:i+chunk]

                if frame_idx != -1:
                    xyz_input = self.forward_nof(xyz_input, torch.tensor([frame_idx]), 'bw_NoF').view(-1, 3)

                # forward nerf model
                if self.N_importance > 0:
                    if isinstance(self.nets['fine_NeRF'], nn.parallel.DistributedDataParallel):
                        xyz_embedded = torch.zeros((xyz_input.shape[0], self.nets['fine_NeRF'].module.in_channels_xyz)).to(self.device)
                    else:
                        xyz_embedded = torch.zeros((xyz_input.shape[0], self.nets['fine_NeRF'].in_channels_xyz)).to(self.device)
                    xyz_embedded_ = self.nerf_embedding_xyz(xyz_input)
                    xyz_embedded[:, :xyz_embedded_.shape[1]] = xyz_embedded_
                    out_chunks += [self.nets['fine_NeRF'](xyz_embedded, sigma_only=True)]
                else:
                    if isinstance(self.nets['fine_NeRF'], nn.parallel.DistributedDataParallel):
                        xyz_embedded = torch.zeros((xyz_input.shape[0], self.nets['coarse_NeRF'].module.in_channels_xyz)).to(self.device)
                    else:
                        xyz_embedded = torch.zeros((xyz_input.shape[0], self.nets['coarse_NeRF'].in_channels_xyz)).to(self.device)
                    xyz_embedded_ = self.nerf_embedding_xyz(xyz_input)
                    xyz_embedded[:, :xyz_embedded_.shape[1]] = xyz_embedded_
                    out_chunks += [self.nets['coarse_NeRF'](xyz_embedded, sigma_only=True)]

            sigma = torch.cat(out_chunks, 0)

        sigma = sigma.cpu().numpy()
        sigma = np.maximum(sigma, 0).reshape(N_grid, N_grid, N_grid)

        # perform marching cube algorithm to retrieve vertices and triangle mesh
        print('Extracting mesh ...')
        vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
        vertices[:,[0, 1]] = vertices[:,[1, 0]]
        triangles[:, [0, 1, 2]] = triangles[:,[0, 2, 1]]
        vertices = vertices / N_grid * 3.0 - 1.5
        if save_path is None:
            save_path = osp.join(
                self.log_dir, 'mesh_epoch_{}_step_{}/{}.obj'.format(\
                self.clock.epoch, self.clock.step, frame_idx if frame_idx != -1 else 'canonical'))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mcubes.export_obj(vertices, triangles, save_path)

        if save_tb and self.is_master:
            self.tb.add_mesh('mesh', torch.from_numpy(vertices).unsqueeze(dim=0), global_step=self.clock.step)

    @ torch.no_grad()
    def visualize_spherical_poses(self, frame_idx, save_path=None):
        img_size = self.spherical_dataset.size

        self.record_str(f'strat rendering video of frame {frame_idx} using spherical poses...')
        if save_path is None:
            save_path = self.log_dir
        video_img_save_path = osp.join(save_path, \
            'spherical_videos_epoch_{}_step_{}/{}_images'.format(\
                self.clock.epoch, self.clock.step, f'frame_{frame_idx}' if frame_idx != -1 else 'canonical'))
        os.makedirs(video_img_save_path, exist_ok=True)

        #start rendering
        vis_out = []
        vis_data = self.spherical_dataset[frame_idx] if frame_idx != -1 else self.spherical_dataset[0]
        for i, (rays, rays_msk) in tqdm(enumerate(zip(vis_data['rays_list'], vis_data['rays_msk_list']))):
            results = self.render(rays, 
                                  torch.ones_like(vis_data['background']),
                                  rays_msk=rays_msk,
                                  use_nof=(frame_idx != -1),
                                  test_time=True)
            _, img_pred, _, depth_pred = self.decode_results(results, img_size)
            img_pred, depth_pred = img_pred.cpu(), depth_pred.cpu()

            if frame_idx != -1:
                img_gt = vis_data['rgbs'].view(img_size[0], img_size[1], 3).permute(2, 0, 1)
                stack = torch.stack([img_gt, img_pred, depth_pred]) # (3, 3, H, W)
            else:
                stack = torch.stack([img_pred, depth_pred]) # (2, 3, H, W)
            cur_img_path = osp.join(video_img_save_path, f"{i:04d}.png")
            vutils.save_image(stack, cur_img_path)
            vis_out.append(imageio.imread(cur_img_path))
        
        imageio.mimwrite(video_img_save_path.replace('images', 'video.mp4'), vis_out, fps=16, quality=8)


    @ torch.no_grad()
    def visualize_video(self, save_path=None):
        img_size = self.val_dataset.size

        self.record_str(f'strat rendering video...')
        if save_path is None:
            save_path = self.log_dir
        video_img_save_path = osp.join(save_path, \
            'videos_epoch_{}_step_{}/images'.format(self.clock.epoch, self.clock.step))
        os.makedirs(video_img_save_path, exist_ok=True)

        vis_out = []
        for frame_idx in tqdm(range(self.num_frames)):
            # get data
            vis_data = self.val_dataset[frame_idx]
            rays, rays_msk, rays_novel, rays_msk_novel, rgbs, background = \
                vis_data['rays'], vis_data['rays_msk'], vis_data['rays_novel'], vis_data['rays_msk_novel'], vis_data['rgbs'], vis_data['background']
            img_gt = rgbs.view(img_size[0], img_size[1], 3).permute(2, 0, 1).cpu() # (3, H, W)

            # render overfit image
            results = self.render(rays, background, rays_msk=rays_msk, use_nof=True, test_time=True)
            _, img_pred, _, depth_pred = self.decode_results(results, img_size)
            img_pred, depth_pred = img_pred.cpu(), depth_pred.cpu()
            # render novel view image
            novel_results = self.render(rays_novel, torch.ones_like(background), rays_msk=rays_msk_novel, use_nof=True, test_time=True)
            _, novel_img_pred, _, novel_depth_pred = self.decode_results(novel_results, img_size)
            novel_img_pred, novel_depth_pred = novel_img_pred.cpu(), novel_depth_pred.cpu()

            stack = torch.cat([img_gt, img_pred, depth_pred, novel_img_pred, novel_depth_pred], dim=-1) # (3, H, W*5)
            cur_img_path = osp.join(video_img_save_path, f"{frame_idx:04d}.png")
            vutils.save_image(stack, cur_img_path)
            vis_out.append(imageio.imread(cur_img_path))
        
        imageio.mimwrite(video_img_save_path.replace('images', 'video.mp4'), vis_out, fps=16, quality=8)


    @ torch.no_grad()
    def visualize_frame(self, frame_idx, save_path=None, save_tb=False):
        img_size = self.val_dataset.size

        if save_path is None:
            save_path = self.log_dir
        img_save_name = 'images_epoch_{}_step_{}/{}'.format(\
                self.clock.epoch, self.clock.step, f'frame_{frame_idx}' if frame_idx != -1 else 'canonical')
        img_save_path = osp.join(save_path, img_save_name+'.png')
        os.makedirs(osp.dirname(img_save_path), exist_ok=True)

        # get vis data
        vis_data = self.val_dataset[frame_idx] if frame_idx != -1 else self.val_dataset[0]
        rays, rays_msk, rays_novel, rays_msk_novel, rgbs, background = \
            vis_data['rays'], vis_data['rays_msk'], vis_data['rays_novel'], vis_data['rays_msk_novel'], vis_data['rgbs'], vis_data['background']
        img_gt = rgbs.view(img_size[0], img_size[1], 3).permute(2, 0, 1).cpu() # (3, H, W)

        # render overfit image
        results = self.render(rays, background, rays_msk=rays_msk, use_nof=(frame_idx!=-1), test_time=True)
        _, img_pred, _, depth_pred = self.decode_results(results, img_size)
        img_pred, depth_pred = img_pred.cpu(), depth_pred.cpu()
        # render novel view image
        novel_results = self.render(rays_novel, torch.ones_like(background), rays_msk=rays_msk_novel, use_nof=(frame_idx!=-1), test_time=True)
        _, novel_img_pred, _, novel_depth_pred = self.decode_results(novel_results, img_size)
        novel_img_pred, novel_depth_pred = novel_img_pred.cpu(), novel_depth_pred.cpu()

        stack = torch.cat([img_gt, img_pred, depth_pred, novel_img_pred, novel_depth_pred], dim=-1) # (3, H, W*5)
        vutils.save_image(stack, img_save_path)

        if save_tb and self.is_master:
            self.tb.add_image(os.path.basename(img_save_name), stack, global_step=self.clock.step)
            
        return img_pred, depth_pred, novel_img_pred, novel_depth_pred


    def visualize_batch(self, save_path=None):
        n_val = self.clock.step // self.config['trainer']['val_every_n_step']

        if self.is_master: # in DDP mode, for master process, only vis canonical space
            if n_val % self.config['trainer']['vis_img_every_n_val'] == 0:
                self.visualize_frame(-1, save_tb=True if self.is_master else False, save_path=save_path)

            if n_val % self.config['trainer']['vis_mesh_video_every_n_val'] == 0:
                self.visualize_spherical_poses(-1, save_path=save_path) # -1 is in canonical space
                self.visualize_mesh(-1, save_path=save_path)
                self.visualize_video(save_path=save_path)

        if self.dist ^ self.is_master:
            vis_idx = np.random.randint(0, self.num_frames)
            if n_val % self.config['trainer']['vis_img_every_n_val'] == 0:
                self.visualize_frame(vis_idx, save_tb=True if self.is_master else False, save_path=save_path)

            if n_val % self.config['trainer']['vis_mesh_video_every_n_val'] == 0:
                self.visualize_spherical_poses(vis_idx, save_path=save_path) # -1 is in canonical space
                self.visualize_mesh(vis_idx, save_path=save_path)
