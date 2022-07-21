import torch
from torch import nn
import torch.nn.init as init

class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=33, 
                 skips=[4],
                 extra_feat_type="none",
                 extra_feat_dim=0,
                 ):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*5*2=33 by default)
        skips: add skip connection in the Dth layer
        extra_feat_type: extra feature for rgb branch, support ["none", "ind", "dir", "latent_code"]
        extra_feat_dim: extra feature length for rgb branch
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)
        
        ## rgb branch
        self.extra_feat_type = extra_feat_type
        self.extra_feat_dim = extra_feat_dim
        assert extra_feat_type in ["none", "ind", "dir", "latent_code"], \
            f"extra_feat_type {extra_feat_type} for NeRF model not supported!!!"
        if extra_feat_type != "none":
            if extra_feat_type == "latent_code":
                self.app_code = torch.randn(1000, extra_feat_dim, requires_grad=True)
            self.extra_encoding = nn.Sequential(
                                        nn.Linear(W+extra_feat_dim, W//2),
                                        nn.ReLU(True))
        else:
            self.extra_encoding = nn.Sequential(
                                        nn.Linear(W, W//2),
                                        nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())
        
    def forward(self, inputs, sigma_only=False, img_ind=None):
        """
        Encodes input to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            - inputs: (B, self.in_channels_xyz(+self.in_channels_dir, +self.ind_emd_dim))
               the embedded vector of position and direction
            - sigma_only: whether to infer sigma only. If True,
                        inputs is of shape (B, self.in_channels_xyz)
            - img_ind: current image index, used when self.use_latent_code is True
        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, extra_feat = torch.split(inputs, [self.in_channels_xyz, self.extra_feat_dim], dim=-1)
        else:
            input_xyz = inputs

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        if self.extra_feat_type == "latent_code":
            raise NotImplementedError("NeRF model does not support latent code yet!!!")
            extra_feat = self.extra_encoding(torch.cat([xyz_encoding_final, self.app_code[img_ind.long()].to(inputs.device)], dim=-1))
        else:
            extra_feat = self.extra_encoding(torch.cat([xyz_encoding_final, extra_feat], dim=-1))
        rgb = self.rgb(extra_feat)

        out = torch.cat([rgb, sigma], dim=-1)
        return out