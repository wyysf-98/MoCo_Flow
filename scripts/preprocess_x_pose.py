import sys 
sys.path.append("..") 
import os
import os.path as osp
import argparse
import imageio
import numpy as np
from tqdm import tqdm
from data_utils import clip_video, matting, generate_background_image, create_moco_flow_data, create_init_nerf_data

args = argparse.ArgumentParser(description='Clip the input video and save to iamge squence.')
args.add_argument('--output_folder', required=True, type=str, help='input video path.')
args.add_argument('--gender', default='neutral', type=str, help='gender for smpl model.')

args = args.parse_args()

def main(args):
    H, W = imageio.imread(args.output_folder + '/images_w_bkgd/0000.png').shape[:2]
    print('H: {}, W: {}'.format(H, W))

    # create init nerf data
    create_init_nerf_data(args.output_folder + '/vibe_output.pkl', size=(H, W), focal=2000, gender=args.gender, canonical_pose="xpose")
    # create_init_nerf_data(args.output_folder + '/vibe_output.pkl', size=(H, W), focal=2000, gender=args.gender)

if __name__ == '__main__':
    main(args)
