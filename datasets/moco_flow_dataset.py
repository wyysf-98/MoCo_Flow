import os
import os.path as osp
import json
import cv2
import random
import trimesh
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms as T
from knn_cuda import KNN

from utils.camera import Camera, rescale_AABB, convert_AABB_to_verts
from utils.smpl.smpl_model import SMPL
from utils.vis_utils import create_spheric_poses, write_ply, write_ply_rgb

class MoCoFlowDataset(Dataset):
    def __init__(self, root_dir, imgs_dir, size, aabb, bkgd, interval=1, cache=True, mode='train'):
        self.root_dir = root_dir
        self.imgs_dir = imgs_dir
        self.size = size
        self.aabb = np.array(aabb)
        self.bkgd = bkgd
        self.interval = interval
        self.cache = cache
        self.mode = mode
        self.vis_mode = None
        if '/' in mode:
            self.mode, self.vis_mode = mode.split('/')

        # prepare others
        self.get_img_transforms()
        self.knn = KNN(k=1, transpose_mode=True)
        self.camera = None # create camera
        if cache:
            self.cached_data = dict()

        # get background image
        if isinstance(self.bkgd, float):
            self.bkgd_img = self.bkgd*torch.ones((3, *self.size))
        elif isinstance(self.bkgd, str):
            if self.bkgd != 'rand':
                self.bkgd_img = self.transform(Image.open(self.bkgd))
        else:
            raise ValueError('background must be float or image path, current is: %s'%self.bkgd)

        self.read_meta()

    def get_img_transforms(self):
        self.transform = T.Compose([
            T.Resize(self.size), 
            T.ToTensor()
        ])

    def read_meta(self):
        with open(os.path.join(self.root_dir, self.mode+'.json'), 'r') as f:
            self.meta = json.load(f)
            frames = []
            for i, cur_frame in enumerate(self.meta['frames']):
                if i % self.interval == 0:
                    frames += [cur_frame]
            self.meta['frames'] = frames
        self.num_frames = len(self.meta['frames']) 

        # get SMPL model
        self.smpl_gpu = SMPL(self.meta['gender'])
        self.smpl_cpu = SMPL(self.meta['gender'])

        # get origin image width and height and set camera parameters
        h_ori, w_ori = self.meta['image_height'], self.meta['image_width']
        scale = [self.size[0]/h_ori, self.size[1]/w_ori]
        D = np.array(self.meta['D'])
        K = np.array([[self.meta['camera_focal']*scale[0], 0, self.meta['camera_c'][0]*scale[0]],
                      [0, self.meta['camera_focal']*scale[1], self.meta['camera_c'][1]*scale[1]],
                      [0, 0, 1]])
        self.camera = Camera(self.size, K, D)

        # create novel camera poses
        if self.mode == 'val':
            canonical_c2w = np.array(self.meta['frames'][0]['camera_pose'])
            canonical_transl = np.array(self.meta['frames'][0]['transl'])
            radius = np.sqrt(((canonical_c2w[:3, 3] - canonical_transl) ** 2).sum())
            self.spherical_poses = create_spheric_poses(radius=radius, center=[0, 0, 0], vec_up=[0, -1, 0])

    def get_frame_correspondence(self, src_frame, tgt_frame=0, num_sampled=10000, thickness=0.2, device=torch.device('cpu')):
        self.smpl_gpu.to(device)
        src_frame_info = self.meta['frames'][src_frame]
        tgt_frame_info = self.meta['frames'][tgt_frame]

        src_pose = torch.from_numpy(np.array(src_frame_info['pose'])).unsqueeze(0).float().to(device)
        src_betas = torch.from_numpy(np.array(src_frame_info['betas'])).unsqueeze(0).float().to(device)
        tgt_pose = torch.from_numpy(np.array(tgt_frame_info['pose'])).unsqueeze(0).float().to(device)
        tgt_betas = torch.from_numpy(np.array(tgt_frame_info['betas'])).unsqueeze(0).float().to(device)

        # source pose -> t-pose
        trans = self.smpl_gpu.get_vertex_transformation(src_pose, src_betas)[0].inverse() # 7877, 4, 4. in smpl space
        # t-pose -> target pose
        trans = self.smpl_gpu.get_vertex_transformation(tgt_pose, tgt_betas)[0] @ trans
        
        # sample some background points in cube
        aabb_box = trimesh.primitives.Box(
            center=[0, 0, 0],
            extents=[3, 3, 3]
        )
        aabb_box_pts = torch.from_numpy(aabb_box.sample_volume(num_sampled)).float().to(device)
        # sample more pts near mesh surface
        src_smpl_verts = self.smpl_gpu.forward(src_pose, src_betas)[0]
        near_surface_pts = src_smpl_verts[torch.randint(src_smpl_verts.shape[0], (num_sampled, ))]
        near_surface_pts += torch.randn_like(near_surface_pts) * thickness
        query_xyzs = torch.cat([aabb_box_pts, near_surface_pts], dim=0)

        # # DEBUG: vis data
        # write_ply(query_xyzs.view(-1, 3).cpu().numpy(), 'query_xyzs.ply')
        # write_ply(src_smpl_verts.view(-1, 3).cpu().numpy(), 'src_smpl_verts.ply')
        # exit()

        # perform knn
        dist, ind = self.knn(src_smpl_verts.unsqueeze(dim=0), query_xyzs.unsqueeze(dim=0))
        dist, ind = dist[0], ind[0]
        # split inside and outside pts
        mask = torch.where(dist<thickness, 1, 0)
        inside_idx = np.where(mask.flatten().cpu().numpy() == 1)[0]
        outside_idx = np.where(mask.flatten().cpu().numpy() == 0)[0]

        homogen_coord = torch.ones((query_xyzs.shape[0], 1)).to(device)
        inputs_homo = torch.cat([query_xyzs, homogen_coord], dim=-1) # N_rays_ x N_samples_, 4
        cano_xyzs = (trans[ind][:,0,:,:] @ inputs_homo.unsqueeze(dim=-1))[:, :3, 0]

        inside_xyzs = torch.cat([query_xyzs.view(-1, 3)[inside_idx], cano_xyzs.view(-1, 3)[inside_idx]], dim=-1)
        outside_xyzs = torch.cat([query_xyzs.view(-1, 3)[outside_idx], cano_xyzs.view(-1, 3)[outside_idx]], dim=-1)
        
        # # DEBUG: vis results
        # inside_corr_pts = inside_xyzs[:,3:].cpu().numpy()
        # cmap = (inside_corr_pts - inside_corr_pts.min(0)) / (inside_corr_pts.max(0) - inside_corr_pts.min(0))
        # write_ply_rgb(np.concatenate([inside_xyzs[:,:3].cpu().numpy(), cmap * 255], axis=-1), 'inside_query_pts_frame%s.ply'%src_frame.item())
        # write_ply_rgb(np.concatenate([inside_xyzs[:,3:].cpu().numpy(), cmap * 255], axis=-1), 'inside_coord_pts_frame%s.ply'%src_frame.item())
        # # write_ply(inside_xyzs[:,:3].cpu().numpy(), 'inside_query_pts_frame%s.ply'%src_frame.item())
        # # write_ply(inside_xyzs[:,3:].cpu().numpy(), 'inside_coord_pts_frame%s.ply'%src_frame.item())
        # exit()
        return inside_xyzs, outside_xyzs

    def gen_smpl_verts(self, frame_info):
        smpl_pose = torch.tensor(np.array(frame_info['pose'])).unsqueeze(0).float()
        smpl_betas = torch.tensor(np.array(frame_info['betas'])).unsqueeze(0).float()
        smpl_verts = self.smpl_cpu.forward(smpl_pose, smpl_betas)[0]

        smpl_mesh = trimesh.Trimesh(vertices=smpl_verts.cpu().numpy(), faces=self.smpl_cpu.faces)
        # smpl_mesh.export('smpl_mesh.obj', file_type='obj')
        aabb = smpl_mesh.bounding_box.bounds
        del smpl_mesh

        return smpl_verts, aabb

    def __len__(self):
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        sample = dict()

        if self.mode == 'train' and self.cache and idx in self.cached_data.keys():
            return self.cached_data[idx]

        sample['idx'] = idx
        img_path = osp.join(self.root_dir, self.imgs_dir, self.meta['frames'][idx]['file_path'])
        sample['image_path'] = img_path
        img = self.transform(Image.open(img_path)) # (3|4, H, W)
        if img.shape[0] == 4:
            # print('image with alpha channel, use custimized background')
            if self.bkgd == 'rand':
                if self.mode == 'val':
                    self.bkgd_img = torch.ones((3, *self.size))
                else:
                    self.bkgd_img = torch.rand(3, 1, 1).repeat(1, *self.size)
            img = img[:3, ...] * img[-1:, ...] + self.bkgd_img * (1 - img[-1:, ...])
        sample['rgbs'] = img.view(3, -1).permute(1, 0) # (H*W, 3) RGB
        sample['background'] = self.bkgd_img.view(3, -1).permute(1, 0) # (H*W, 3) RGB

        # print(img.shape)
        # torchvision.utils.save_image(img, 'debug.png')
        # exit()

        transl = np.array(self.meta['frames'][idx]['transl'])
        # perform SMPL on the frame
        smpl_verts, aabb = self.gen_smpl_verts(self.meta['frames'][idx])
        sample['smpl_verts'] = smpl_verts
        aabb = rescale_AABB(aabb, self.aabb[0], self.aabb[1])
        aabb_verts = convert_AABB_to_verts(aabb) 

        # setup the camera
        self.camera.c2w = np.array(self.meta['frames'][idx]['camera_pose'])
        self.camera.c2w[:3, 3] = self.camera.c2w[:3, 3] - transl
        sample['c2w'] = self.camera.c2w
        # create valid mask
        sample['rays_msk'] = self.camera.get_valid_rays_mask(aabb_verts)
        # make rays sampling for each image
        sample['rays'] = self.camera.make_rays(aabb_verts, idx * 2 / self.num_frames - 1)

        if self.mode == 'train' and self.cache:
            self.cached_data[idx] = sample
        else: # create data for each image separately in val dataset
            if self.mode == 'val' and self.vis_mode == None:
                self.camera.c2w = self.spherical_poses[np.random.randint(0, self.spherical_poses.shape[0])] # random select a view
                sample['c2w_novel'] = self.camera.c2w
                sample['rays_msk_novel'] = self.camera.get_valid_rays_mask(aabb_verts)
                sample['rays_novel'] = self.camera.make_rays(aabb_verts, idx * 2 / self.num_frames - 1)

            elif self.mode == 'val' and self.vis_mode == 'spherical_path':
                rays_list = []
                valid_rays_mask_list = []
                for c2w in self.spherical_poses: # use sprial posese
                    self.camera.c2w = c2w
                    valid_rays_mask_list += [self.camera.get_valid_rays_mask(aabb_verts)]
                    rays_list += [self.camera.make_rays(aabb_verts, idx * 2 / self.num_frames - 1)]
                sample['c2w_list'] = self.spherical_poses
                sample['rays_msk_list'] = valid_rays_mask_list
                sample['rays_list'] = rays_list
            
            else:
                raise ValueError('dataset mode error')
            
        return sample