import os
import os.path as osp
import imageio
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import mcubes
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.utils as vutils

from datasets import get_dataset
from models import get_model, get_loss
from models.metrics import psnr, ssim
from models.rendering import render_rays
from trainer.base import BaseTrainer
from utils.vis_utils import visualize_depth

class NeRFTrainer(BaseTrainer):
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

        self.val_spherical_dataset = get_dataset(data_config, 'val/spherical_path')

    def build_model(self, model_config):
        # create nerf embedding
        self.nerf_embedding_xyz = get_model(model_config['nerf_embedding_xyz']) if model_config['nerf_embedding_xyz'] is not None else None
        self.nerf_embedding_ind = get_model(model_config['nerf_embedding_ind']) if model_config['nerf_embedding_ind'] is not None else None
        self.nerf_embedding_dir = get_model(model_config['nerf_embedding_dir']) if model_config['nerf_embedding_dir'] is not None else None
        self.nerf_embeddings = [self.nerf_embedding_xyz, self.nerf_embedding_ind, self.nerf_embedding_dir]

        # create nerf model
        self.nets['coarse_NeRF'] = get_model(model_config['coarse_NeRF'])
        self.nerf_models = [self.nets['coarse_NeRF']]
        self.N_importance = model_config['N_importance']
        if self.N_importance > 0:
            self.nets['fine_NeRF'] = get_model(model_config['fine_NeRF'])
            self.nerf_models.append(self.nets['fine_NeRF'])

        self.record_str('-----trainable network architecture-----')
        self.record_str(self.nets)

        if self.config['model']['pretrained_path'] is not None:
            self.load_ckpt(self.config['model']['pretrained_path'], restore_clock=False, restore_optimizer=False)

        if self.dist:
            self.DDP_mode()
        else:
            self.CUDA_mode()

    def set_loss_function(self, loss_config):
        self.criterion_img = get_loss(loss_config).to(self.device)

    def forward(self, rays, background, test_time=False):

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
    
    def render(self, rays, background, rays_msk=None, test_time=False):
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

    def train_step(self, data):
        # get data
        rays, rays_msk, rgbs, background = data['rays'], data['rays_msk'], data['rgbs'], data['background']
        batch_size, num_rays = rays.shape[:2]
        assert batch_size == 1, 'Only support batch size of 1.'
        rays, rays_msk, rgbs, background = \
            rays.squeeze(0), rays_msk.squeeze(0), rgbs.squeeze(0), background.squeeze(0)

        val_inds = torch.nonzero(rays_msk).squeeze(1)
        sel_inds = val_inds[torch.randperm(val_inds.shape[0])[:self.config['model']['N_rand']]]
        sampled_rays, sampled_rgbs, sampled_background = rays[sel_inds], rgbs[sel_inds], background[sel_inds]
        sampled_rgbs = sampled_rgbs.to(self.device)

        # forward
        results = self.render(sampled_rays, sampled_background, test_time=False)
        self.losses['img_loss'] = self.criterion_img(results, sampled_rgbs)

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], sampled_rgbs)
            self.extra['train_psnr'] = psnr_

    def val_step(self, data):
        rays, rays_msk, rgbs, background= data['rays'], data['rays_msk'], data['rgbs'], data['background']
        rays, rays_msk, rgbs, background = \
            rays.squeeze(dim=0), rays_msk.squeeze(0), rgbs.squeeze(dim=0), background.squeeze(dim=0)
        rgbs = rgbs.to(self.device)

        results = self.render(rays, background, test_time=False)
        self.losses['img_loss'] = self.criterion_img(results, rgbs)

        img_size = self.val_dataset.size
        img_ori, img_pred, _, _ = self.decode_results(results, img_size)
        psnr_ = psnr(img_ori, rgbs)
        img_pred = img_pred.unsqueeze(dim=0)
        img_gt = rgbs.view(img_size[0], img_size[1], 3).permute(2, 0, 1).unsqueeze(dim=0) # (1, 3, H, W)
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
    def visualize_mesh(self, N_grid=256, sigma_threshold=10, chunk=10000, save_path=None):
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

                # forward nerf model
                if self.N_importance > 0:
                    if isinstance(self.nets['fine_NeRF'], nn.parallel.DistributedDataParallel):
                        fine_NeRF_in_channels_xyz = self.nets['fine_NeRF'].module.in_channels_xyz
                    else:
                        fine_NeRF_in_channels_xyz = self.nets['fine_NeRF'].in_channels_xyz
                    xyz_embedded = torch.zeros((xyz_input.shape[0], fine_NeRF_in_channels_xyz)).to(self.device)
                    xyz_embedded_ = self.nerf_embedding_xyz(xyz_input)
                    xyz_embedded[:, :xyz_embedded_.shape[1]] = xyz_embedded_
                    out_chunks += [self.nets['fine_NeRF'](xyz_embedded, sigma_only=True)]
                else:
                    if isinstance(self.nets['coarse_NeRF'], nn.parallel.DistributedDataParallel):
                        coarse_NeRF_in_channels_xyz = self.nets['coarse_NeRF'].module.in_channels_xyz
                    else:
                        coarse_NeRF_in_channels_xyz = self.nets['coarse_NeRF'].in_channels_xyz
                    xyz_embedded = torch.zeros((xyz_input.shape[0], coarse_NeRF_in_channels_xyz)).to(self.device)
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
                self.log_dir, 'mesh_epoch_{}_step_{}.obj'.format(self.clock.epoch, self.clock.step))
        mcubes.export_obj(vertices, triangles, save_path)
        self.tb.add_mesh('mesh', torch.from_numpy(vertices).unsqueeze(dim=0), global_step=self.clock.step)

    @ torch.no_grad()
    def visualize_frame(self, frame_idx, save_path=None, save_tb=False):
        img_size = self.val_dataset.size

        if save_path is None:
            save_path = self.log_dir
        img_save_name = 'images_epoch_{}_step_{}/frame_{}'.format(self.clock.epoch, self.clock.step, frame_idx)
        img_save_path = osp.join(save_path, img_save_name+'.png')
        os.makedirs(osp.dirname(img_save_path), exist_ok=True)

        # get vis data
        vis_data = self.val_dataset[frame_idx]
        rays, rays_msk, rays_novel, rays_msk_novel, rgbs, background = \
            vis_data['rays'], vis_data['rays_msk'], vis_data['rays_novel'], vis_data['rays_msk_novel'], vis_data['rgbs'], vis_data['background']
        img_gt = rgbs.view(img_size[0], img_size[1], 3).permute(2, 0, 1).cpu() # (3, H, W)

        # render overfit image
        results = self.render(rays, background, rays_msk=rays_msk, test_time=True)
        _, img_pred, _, depth_pred = self.decode_results(results, img_size)
        img_pred, depth_pred = img_pred.cpu(), depth_pred.cpu()
        # render novel view image
        novel_results = self.render(rays_novel, background, rays_msk=rays_msk_novel, test_time=True)
        _, novel_img_pred, _, novel_depth_pred = self.decode_results(novel_results, img_size)
        novel_img_pred, novel_depth_pred = novel_img_pred.cpu(), novel_depth_pred.cpu()

        stack = torch.cat([img_gt, img_pred, depth_pred, novel_img_pred, novel_depth_pred], dim=-1) # (3, H, W*5)
        vutils.save_image(stack, img_save_path)

        if save_tb:
            self.tb.add_image('frame_%s'%frame_idx, stack, global_step=self.clock.step)

        return stack

    @ torch.no_grad()
    def visualize_spherical_poses(self, save_path=None):
        img_size = self.val_spherical_dataset.size

        self.record_str('strat rendering video using spherical poses...')
        if save_path is None:
            save_path = self.log_dir
        video_img_save_path = osp.join(save_path, \
            'videos_epoch_{}_step_{}/images'.format(self.clock.epoch, self.clock.step))
        os.makedirs(video_img_save_path, exist_ok=True)

        #start rendering
        vis_out = []
        vis_data = self.val_spherical_dataset[0]
        for i, (rays, rays_msk) in tqdm(enumerate(zip(vis_data['rays_list'], vis_data['rays_msk_list']))):
            results = self.render(rays, 
                                  vis_data['background'],
                                  rays_msk=rays_msk,
                                  test_time=True)
            _, img_pred, _, depth_pred = self.decode_results(results, img_size)
            img_pred, depth_pred = img_pred.cpu(), depth_pred.cpu()

            stack = torch.stack([img_pred, depth_pred]) # (2, 3, H, W)
            cur_img_path = osp.join(video_img_save_path, f"{i:04d}.png")
            vutils.save_image(stack, cur_img_path)
            vis_out.append(imageio.imread(cur_img_path))
        
        imageio.mimwrite(video_img_save_path.replace('images', 'video.mp4'), vis_out, fps=16, quality=8)

    def visualize_batch(self, save_path=None):
        n_val = self.clock.step // self.config['trainer']['val_every_n_step']

        if n_val % self.config['trainer']['vis_img_every_n_val'] == 0:
            frame_idx = np.random.randint(0, self.num_frames)
            self.visualize_frame(frame_idx, save_tb=True if self.is_master else False)

        if n_val % self.config['trainer']['vis_mesh_video_every_n_val'] == 0:
            self.visualize_mesh()
            self.visualize_spherical_poses()