import os
import os.path as osp
import torch
import argparse
import collections
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from trainer import get_trainer
from utils.parse_config import ConfigParser

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def test(config, args):
    # create network and trainer
    trainer = get_trainer(config)
    
    # load from checkpoint
    config['dataloader']['val_size'] = [args.reso, args.reso]
    config['model']['pretrained_nerf'] = None
    config['model']['pretrained_nof'] = None
    config['dataloader']['bkgd'] = 1.0
    trainer.build_model(config['model'])
    if config.resume:
        trainer.load_ckpt(config.resume)
    else:
        ckpts = {}
        for ckpt in glob(f"{config['save_dir']}/{config['exp_name']}/*/ckpts/*.pth"):
            itr = ckpt.split('_iter')[1].split('.')[0]
            ckpts[int(itr)] = ckpt
        ckpts = sorted(ckpts.items())
        if len(ckpts) != 0:
            trainer.load_ckpt(ckpts[-1][1])
    trainer.increase_xyzemb_dim()


    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
    if args.render_gt:
        assert osp.exists(config['dataloader']['root_dir']), 'please provide the dataset path.'
        trainer.prepare_dataloader(config['dataloader'])
    else:
        assert args.test_json != None, 'please provide the .json file using --test_json.'
        config['dataloader']['root_dir'] = osp.dirname(args.test_json)
        trainer.prepare_dataloader(config['dataloader'])
        trainer.num_frames = trainer.val_dataset.num_frames

    if args.render_training_poses:
        trainer.visualize_video(vis_novel_view=False, save_path=args.out_dir)

    if args.render_spherical_poses:
        trainer.visualize_spherical_poses(args.spherical_poses_frame, save_path=args.out_dir)

    if args.extract_mesh:
        trainer.visualize_mesh(args.mesh_frame,  N_grid=args.N_grid, sigma_threshold=args.sigma_threshold, save_path=args.out_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='testing pipeline defination')
    args.add_argument('-m', '--mode', default='test', type=str, help='current mode.')
    args.add_argument('-c', '--config', default=None, type=str, required=True,
                      help='config file path. (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='file path to retore the checkpoint. (default: None)')
    args.add_argument('-n', '--job_name', default=None, type=str,
                      help='job name. If None, use current time stamp. (default: None)')
    args.add_argument('-s', '--seed', default=None,
                      help='random seed used. (default: None)')
    args.add_argument('-g', '--gpu', default='0', type=str,
                      help='use single gpu to train. (defalut: 0)')
    args.add_argument('-d', '--dist', action='store_true',
                      help='whether to use distribute training.')
    args.add_argument('--local_rank', default=-1, type=int,
                      help='node rank for distributed training. (default: -1)')

    # other arguments
    args.add_argument('--out_dir', default=None, type=str,
                      help='output path for visulation')
    args.add_argument('--reso', default=512, type=int,
                      help='rendering resolution. (default: 512)')
    args.add_argument('--render_gt', action='store_true',
                      help='whether to render the GT image.')
    args.add_argument('--test_json', default=None, type=str,
                      help='.json file for testing, used when render_gt == False.')
    args.add_argument('--render_training_poses', action='store_true',
                      help='whether to render using training poses.')
    args.add_argument('--render_spherical_poses', action='store_true',
                      help='whether to render the input frame using spherical poses.')
    args.add_argument('--spherical_poses_frame', default=-1, type=int, # -1 is in canonical space
                      help='vis a single frame using spherical poses. (default: -1)')
    args.add_argument('--extract_mesh', action='store_true',
                      help='whether to extract current mesh.')
    args.add_argument('--mesh_frame', default=-1, type=int, # -1 is in canonical space
                      help='frame index to extract mesh.')
    args.add_argument('--N_grid', default=512, type=int,
                      help='N_grid for extracting mesh.')
    args.add_argument('--sigma_threshold', default=10, type=int,
                      help='sigma_threshold for extracting mesh.')

    config = ConfigParser.from_args(args)
    test(config, args.parse_args())
