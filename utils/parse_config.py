import os
import os.path as osp
import re
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
from functools import reduce, partial
from operator import getitem
from glob import glob, iglob
from .base_utils import ensure_dirs, set_seed

class ConfigParser:
    def __init__(self, config, job_name=None, seed=None, resume=None, gpu_id=0, local_rank=-1, mode='train'):
        """
        class to parse configuration .yaml file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config      : Dict containing configurations, hyperparameters for training.
        :param job_name    : str, Unique Identifier for training processes. Used to save checkpoints and training log. (Timestamp is being used as default)
        :param seed        : int, random seed for training.
        :param resume      : str, path to the checkpoint file to resume training.
        :param gpu_id      : int, current gpu id.
        :param local_rank  : int, local rank for current process.
        :param mode        : str, config mode.
        """
        self.config = config
        self.job_name = job_name
        self.seed = seed
        self.resume = resume
        self.gpu_id = gpu_id
        self.local_rank = local_rank
        self.dist = False if local_rank == -1 else True 
        self.mode = mode

        # set random seed
        if seed is not None:
            set_seed(seed)

        # set save_dir where trained model and log will be saved.
        exp_name = self.config['exp_name']
        if job_name is None: # if job_name is None, use current timestamp as default
            job_name = datetime.now().strftime('%m%d_%H%M%S') 
        save_dir = Path(self.config['save_dir']) / exp_name / job_name
        os.makedirs(save_dir, exist_ok=True)
        self.ckpts_dir = save_dir / 'ckpts'
        self.log_dir = save_dir / 'log'
        if self.is_master:
            # save code backup and config file
            _save_codes(save_dir / 'code')
            _save_yaml(self.config, save_dir / 'config.yaml')
            if mode == 'train': # make directory for saving checkpoints and log in train mode
                ensure_dirs([self.ckpts_dir, self.log_dir])
                

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def __str__(self):
        """
        print config dict with pretty format.
        """
        head_str = '*'*20+'  Config  '+'*'*20
        end_str = '='*50
        config_str = head_str + '\n' + _remove_punctuation(json.dumps(self.config, indent=2, ensure_ascii=False)) + '\n' + end_str + '\n'
        return config_str

    @classmethod
    def from_args(cls, args):
        """
        Initialize this class from cli arguments and options.
        """
        if not isinstance(args, tuple):
            args = args.parse_args()
        
        if args.dist:
            print('Distribute Training Mode')
            local_rank = args.local_rank
            gpu_id = local_rank
        if not args.dist and args.gpu:
            print('Single GPU Training Mode, use gpu %s'%args.gpu)
            local_rank = -1
            gpu_id = args.gpu

        if args.resume is not None:
            cfg_fpath = Path(args.resume).parent.parent / 'config.yaml'
            config = _load_yaml(cfg_fpath)
        else:
            assert args.config is not None, "Configuration file need to be specified. Add '-c config.yaml', for example."
            config = _load_yaml(args.config)

        return cls(config,
                   job_name=args.job_name, 
                   seed=args.seed, 
                   resume=args.resume, 
                   gpu_id=gpu_id,
                   local_rank=local_rank,
                   mode=args.mode)

    @property
    def is_master(self):
        return True if self.local_rank in [-1, 0] else False
    

def _load_yaml(file_path):
    '''
        load config file with .yaml format and return a dict.
    '''
    with open(file_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    return config


def _save_yaml(d, save_path):
    """
    save d with Dict format to $save_path.
    """
    with open(save_path, 'w') as f:
        f.write(yaml.dump(d))
    f.close()


def _save_codes(save_path):
    """
    save codes to $save_path.
    """
    cur_codes_path = osp.dirname(osp.dirname(os.path.abspath(__file__)))
    shutil.copytree(cur_codes_path, save_path, \
        ignore=shutil.ignore_patterns('data', 'outputs', 'exps', 'docker', 'log', 'scripts', '*.txt', '*.png', '*.gif', '*.pkl'))


def _remove_punctuation(text):
    """
    remove punctuation for better print.
    """
    punctuation = '{!,;?"\'、，；}'
    text = re.sub(r'[{}]+'.format(punctuation), ' ', text)
    return text.strip()


def _merge_config(base_config, user_config):
    """
    helper functions to merge two config dicts
    """
    merged_dict = {}
    merged_dict = base_config.copy()
    for k in user_config.keys():
        if not isinstance(user_config[k], dict):
            merged_dict[k] = user_config[k]
        elif k not in base_config:
            merged_dict[k] = _merge_config({}, user_config[k])
        else:
            merged_dict[k] = _merge_config(base_config[k], user_config[k])
    return merged_dict


def _update_config(config, modification=None):
    """
    helper functions to update config dict with custom cli options
    """
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

