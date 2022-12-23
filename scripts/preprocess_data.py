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
args.add_argument('--input_video', required=True, type=str, help='input video path.')
args.add_argument('--output_folder', required=True, type=str, help='input video path.')
args.add_argument('--start_frame', default=0, type=int, help='clip start frame index.')
args.add_argument('--end_frame', default=-1, type=int, help='clip end frame index.')
args.add_argument('--interval', default=1, type=int, help='sampling interval of the video.')
args.add_argument('--thres', default=128, type=int, help='threshold for matting.')
args.add_argument('--vibe_path', default='./VIBE', type=str, help='vibe code path for pose estimation.')
args.add_argument('--grender', default='neutral', type=str, help='gender for smpl model.')

args = args.parse_args()

def main(args):
    # prepare save folder
    os.makedirs(args.output_folder + '/images_w_bkgd', exist_ok=True)
    os.makedirs(args.output_folder + '/images', exist_ok=True)

    # clip the video
    video_path = args.input_video
    assert osp.exists(video_path), '{} does not exist'.format(video_path)
    start_frame, end_frame, interval = args.start_frame, args.end_frame, args.interval
    if end_frame == -1:
        end_frame = int(imageio.get_reader(video_path).count_frames())
    print('Clip video {} from {} to {} with interval {}'.format(video_path, start_frame, end_frame, interval))

    # save the image sequence
    frames = clip_video(video_path, start_frame, end_frame, interval)
    # video_path = args.input_video.replace('.mp4', f'_clip_from{start_frame}to{end_frame}with_interval{interval}.mp4')
    video_path = args.output_folder + '/video.mp4'
    for i, frame in tqdm(enumerate(frames)):
        imageio.imwrite(args.output_folder + '/images_w_bkgd/{:04d}.png'.format(i), frame)
    imageio.mimsave(video_path, frames, fps=30)

    # matting
    matting(args.output_folder + '/images_w_bkgd', args.output_folder + '/images', thres=args.thres)

    # generate the background image
    generate_background_image(images_path = args.output_folder + '/images_w_bkgd', \
                              masks_path = args.output_folder + '/images/mask', \
                              save_path = args.output_folder + '/background.png')

    # run VIBE to get pose estimation results
    cmd = f'cd {args.vibe_path} && \
            python demo.py --vid_file ../{video_path} --output_folder ../{args.output_folder} --sideview --run_smplify --smooth && \
            mv ../{args.output_folder}/video/* ../{args.output_folder}/ && \
            rm -rf ../{args.output_folder}/video'
    os.system(cmd)
    
    # convert VIBE ouput to json
    H, W = imageio.imread(args.output_folder + '/images_w_bkgd/0000.png').shape[:2]
    print('H: {}, W: {}'.format(H, W))
    create_moco_flow_data(args.output_folder + '/vibe_output.pkl', size=(H, W), focal=2000, gender='m', vis=False)

    # create init nerf data
    create_init_nerf_data(args.output_folder + '/vibe_output.pkl', size=(H, W), focal=2000, gender='m')

if __name__ == '__main__':
    main(args)
