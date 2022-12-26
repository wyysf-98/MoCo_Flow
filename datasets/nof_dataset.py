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

class NoFDataset(Dataset):
    def __init__(self, root_dir, interval=1, cache=True, mode='train'):
        self.root_dir = root_dir
        self.interval = interval
        self.cache = cache
        self.mode = mode

        # prepare others
        self.knn = KNN(k=1, transpose_mode=True)
        if cache:
            self.cached_data = dict()

        self.read_meta()

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

    def get_frame_correspondence(self, src_frame, tgt_frame=0, num_sampled=10000, thickness=0.1, device=torch.device('cpu')):
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


    def __len__(self):
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        sample = dict()

        if self.mode == 'train' and self.cache and idx in self.cached_data.keys():
            return self.cached_data[idx]

        sample['idx'] = idx

        if self.mode == 'train' and self.cache:
            
            self.cached_data[idx] = sample

        return sample