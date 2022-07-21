import torch
from torch import nn
import torch.nn.init as init
from kornia.geometry.conversions import quaternion_log_to_exp, quaternion_to_rotation_matrix

class NoF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=33, 
                 skips=[4], 
                 extra_feat_type="ind",
                 extra_feat_dim=0,
                 use_quat=False):
        """
        D: number of layers in each layer
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*5*2=33 by default)
        skips: add skip connection in the Dth layer
        extra_feat_type: extra feature for rgb branch, support ["ind", "latent_code"]
        extra_feat_dim: extra feature length
        use_quat: False: directly to predit optical flow. 
                  True: predict rotation q, pivot point s followed by translation t to get the final xyz
        """
        super(NoF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.skips = skips
        self.use_quat = use_quat

        # get time step encoding dim
        self.extra_feat_type = extra_feat_type
        self.extra_feat_dim = extra_feat_dim
        assert extra_feat_type in ["ind", "latent_code"], \
            f"extra_feat_type {extra_feat_type} for NoF model not supported!!!"
        if extra_feat_type == "latent_code":
            self.time_code = torch.randn(1000, extra_feat_dim, requires_grad=True)

        # encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz+extra_feat_dim, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz+extra_feat_dim, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"nof_encoding_{i+1}", layer)

        if self.use_quat:
            self.nof_encoding_final = nn.Linear(W, 9) # 3(log quaternion) + 3(pivot point) + 3(translation)
        else:
            self.nof_encoding_final = nn.Linear(W, 3) # 3(optical flow)

    def forward(self, inputs, xyz, img_ind=None):
        '''
        inputs
            - inputs: tensor, (N, in_channels_xyz (+ self.ind_emd_dim))
            - xyz: tensor, (N, 3). origin points
            - img_ind: current image index, used when self.use_latent_code is True
        return:
            - outputs: predicted location, tensor, (N, 3)
        '''
        if self.extra_feat_type == "latent_code":
            raise NotImplementedError("NoF model does not support latent code yet!!!")
            assert img_ind is not None and inputs.shape[1] == self.in_channels_xyz
            inputs = torch.cat([inputs, self.time_code[img_ind.long()].to(inputs.device)], dim=-1)

        inputs_ = inputs
        for i in range(self.D):
            if i in self.skips:
                inputs_ = torch.cat([inputs, inputs_], -1)
            inputs_ = getattr(self, f"nof_encoding_{i+1}")(inputs_)

        if self.use_quat:
            transforms = self.nof_encoding_final(inputs_)
            v, s, t = transforms[:,:3], transforms[:,3:6], transforms[:,6:9]
            q = quaternion_log_to_exp(v)
            r = quaternion_to_rotation_matrix(q)
            outputs = torch.bmm((xyz-s).unsqueeze(dim=1), r).squeeze(dim=1) + s + t
        else:
            outputs = self.nof_encoding_final(inputs_) + xyz
        # outputs = torch.tanh(outputs)
        
        return outputs