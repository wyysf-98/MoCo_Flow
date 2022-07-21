from datasets.nof_dataset import NoFDataset
from datasets.moco_flow_dataset import MoCoFlowDataset

def get_dataset(data_config, mode):
    if data_config['type'] == 'nof':
        return NoFDataset(data_config['root_dir'],
                          data_config['canonical_pose'],
                          interval=data_config['interval'],
                          cache=data_config['cache'],
                          mode=mode,
                          )
    elif data_config['type'] == 'moco_flow':
        return MoCoFlowDataset(data_config['root_dir'],
                               data_config['imgs_dir'],
                               data_config['size'],
                               data_config['aabb'],
                               data_config['bkgd'],
                               data_config['canonical_pose'],
                               interval=data_config['interval'],
                               cache=data_config['cache'],
                               mode=mode,
                               )
    else:
        raise ValueError('dataloader type: {} not valid'.format(data_config['type']))
