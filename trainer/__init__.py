from trainer.trainer_nerf import NeRFTrainer
from trainer.trainer_nof import NoFTrainer
from trainer.trainer_moco_flow import MoCoFlowTrainer

def get_trainer(config):
    model_type = config['model']['type']

    if model_type == 'nerf':
        return NeRFTrainer(config)
    if model_type == 'nof':
        return NoFTrainer(config)
    elif model_type == 'moco_flow':
        return MoCoFlowTrainer(config)
    else:
        raise ValueError('trainer model type: {} not valid'.format(model_type))
