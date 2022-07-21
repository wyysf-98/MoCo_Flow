from models.embedding import Embedding
from models.nerf import NeRF
from models.nof import NoF

import torch.nn as nn
from models.losses import MSELoss

def get_model(model_config):    
    if model_config['type'] == "Embedding":
        return Embedding(model_config['in_channels'],
                         model_config['N_freqs'],
                         model_config['logscale'])
    elif model_config['type'] == "NeRF":
        return NeRF(model_config['D'],
                    model_config['W'],
                    model_config['in_channels_xyz'],
                    model_config['skips'],
                    model_config['extra_feat_type'],
                    model_config['extra_feat_dim'])
    elif model_config['type'] == "NoF":
        return NoF(model_config['D'],
                   model_config['W'],
                   model_config['in_channels_xyz'],
                   model_config['skips'],
                    model_config['extra_feat_type'],
                    model_config['extra_feat_dim'],
                   model_config['use_quat'])
    else:
        raise ValueError('model type: {} not valid'.format(model_config['type']))

def get_loss(loss_config):    
    if loss_config['type'] == "MSE":
        return MSELoss()
    elif loss_config['type'] == 'L1':
        return nn.L1Loss()
    elif loss_config['type'] == 'BCE':
        return nn.BCELoss()
    else:
        raise ValueError('loss type: {} not support'.format(loss_config['type']))
