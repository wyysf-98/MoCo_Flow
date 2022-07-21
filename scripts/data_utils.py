import sys 
sys.path.append("..") 
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import os.path as osp
import argparse
import cv2
import json
import torch
import imageio
import trimesh
import pyrender
import joblib
import colorsys
import numpy as np
from tqdm import tqdm
from glob import glob
from pyrender.constants import RenderFlags

from utils.smpl.smpl_model import SMPL

class Renderer:
    def __init__(self, resolution=(1080,1080)):
        self.resolution = resolution

        self.renderer = pyrender.OffscreenRenderer(
            viewport_height=self.resolution[0],
            viewport_width=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, -5]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, -5]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, -4]
        self.scene.add(light, pose=light_pose)

    def render(self, mesh, camera, camera_pose, bkgd=None, color=None, color_map=None, wireframe=False):
        if color is None:
            color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        if color_map is not None:
            mesh.primitives[0].color_0 = color_map

        mesh_node = self.scene.add(mesh, 'mesh')
        cam_node = self.scene.add(camera, pose=camera_pose)

        if wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        if rgb.shape[2] == 4:
            if bkgd is None:
                output_img = rgb[:, :, :-1] * valid_mask
            else:
                output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * bkgd
        else:
            if bkgd is None:
                output_img = rgb * valid_mask
            else:
                output_img = rgb * valid_mask + (1 - valid_mask) * bkgd[:, :, :-1]
        render_img = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return render_img, valid_mask

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def clip_video(video_path, start_frame, end_frame, interval):
    vid = imageio.get_reader(video_path)
    frames = []
    # imageio.imwrite('start.png', vid.get_data(start_frame))
    # imageio.imwrite('end.png', vid.get_data(end_frame))
    # exit()
    for i in tqdm(range(start_frame, end_frame, interval)):
        frames.append(vid.get_data(i))
    return frames

def find_max_region(msk):
    contours, _ = cv2.findContours(msk,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
 
    max_idx = np.argmax(area)
    max_area = cv2.contourArea(contours[max_idx])
    for i in range(len(contours)):
        if i != max_idx:
            cv2.fillPoly(msk, [contours[i]], 0)

    return msk


def matting(video_path, output_path, thres=10):
    '''
    use matting method from: https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md
    '''
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50") # or "mobilenetv3"
    convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")
    convert_video(
                model,                                     # The loaded model, can be on any device (cpu or cuda).
                input_source=video_path,                   # A video file or an image sequence directory.
                downsample_ratio=1.0,                      # [Optional] If None, make downsampled max size be 512px.
                output_type='png_sequence',                # Choose "video" or "png_sequence"
                output_composition=output_path,            # File path if video; directory path if png sequence.
                output_alpha=f"{output_path}/pha",         # [Optional] Output the raw alpha prediction.
                seq_chunk=12,                              # Process n frames at once for better parallelism.
                num_workers=0,                             # Only for image sequence input. Reader threads.
                progress=True                              # Print conversion progress.
            )

    # post-processing
    os.makedirs(f'{output_path}/mask', exist_ok=True)
    print(f'post-processing: {output_path}')
    for i in tqdm(range(len(glob(osp.join(output_path, '*.png'))))):
        alpha = imageio.imread(f'{output_path}/pha/{i:04d}.png')
        alpha[alpha<thres] = 0
        alpha[alpha>=thres] = 255
        alpha = find_max_region(alpha)
        imageio.imwrite(f'{output_path}/mask/{i:04d}.png', alpha)

        img = imageio.imread(osp.join(output_path, f'{i:04d}.png'))
        img[:, :, 3] = alpha
        imageio.imwrite(osp.join(output_path, f'{i:04d}.png'), img)


def generate_background_image(images_path, masks_path, save_path):
    num_images = len(glob(osp.join(images_path, '*.png')))
    print(f'generate background image: {save_path}')

    bkgd_imgs = []
    for i in range(num_images):
        img = imageio.imread(osp.join(images_path, f'{i:04d}.png'))
        msk = imageio.imread(osp.join(masks_path, f'{i:04d}.png'))
        bkgd = img * (1 - msk[:, :, np.newaxis] / 255)
        bkgd_imgs += [bkgd]

    bkgd_imgs = np.array(bkgd_imgs)
    bkgd_img = np.sort(bkgd_imgs, axis=0)[int(num_images*0.9)]
    imageio.imwrite(save_path, bkgd_img)


def sample_on_sphere(num_samples, dist=1, half=False):
    """ sample x,y,z location from the sphere
    reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
    """
    num_samples = num_samples * 2 if half else num_samples
    phi = (np.sqrt(5) - 1.0) / 2.
    pos_list = []
    for n in range(1, num_samples + 1):
        y = (2. * n - 1) / num_samples - 1.
        x = np.cos(2*np.pi*n*phi)*np.sqrt(1-y*y)
        z = np.sin(2*np.pi*n*phi)*np.sqrt(1-y*y)
        if half and y < 0:
            continue
        pos_list.append((x*dist, y*dist, z*dist))

    return np.array(pos_list)


def get_camera_pose(camera_position, obj_potision):
    direction = camera_position - obj_potision
    cameraZ = direction / np.linalg.norm(direction)
    if abs(cameraZ[2]) < 0.999:
        cameraX = np.cross(np.array([0, 0, 1]), cameraZ)
        cameraX = cameraX / np.linalg.norm(cameraX)
        cameraY = np.cross(cameraZ, cameraX)
    else:
        cameraX = np.array([1, 0, 0])
        cameraY = np.cross(cameraZ, cameraX)

    cameraPose = np.eye(4)
    cameraPose[:3, 0] = cameraX
    cameraPose[:3, 1] = cameraY
    cameraPose[:3, 2] = cameraZ
    cameraPose[:3, 3] = camera_position
    return cameraPose


def create_moco_flow_data(pkl_path, size, focal=2000, gender='neutral', vis=False):
    vibe_output = joblib.load(open(pkl_path, 'rb'))
    save_folder = osp.dirname(pkl_path)
    print(f'create moco flow data: {pkl_path}')
    assert len(vibe_output) == 1

    H, W = size
    cams = vibe_output[1]['orig_cam']
    betas = vibe_output[1]['betas']
    poses = vibe_output[1]['pose']
    frame_ids = vibe_output[1]['frame_ids']
    moco_dict = {
        'gender': gender,
        'image_height': H,
        'image_width': W,
        'camera_focal': focal,
        'camera_c': np.array([W / 2, H / 2]),
        'D': np.zeros((5,)),
        'frames': []
        }

    if vis:
        renderer = Renderer(resolution=size)
        camera = pyrender.camera.IntrinsicsCamera(
                fx=focal,
                fy=focal,
                cx=W/2,
                cy=H/2,
            )
        smpl = SMPL(gender)
        vis_imgs = []

    for i in tqdm(range(len(frame_ids))):
        frame_id = frame_ids[i]
        cam = cams[frame_id]
        cur_pose = poses[frame_id]
        cur_betas = betas[frame_id]
        cur_transl = np.array([cam[2], cam[3], 2*focal/(cam[0]*W)])
        camera_pose = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
        moco_dict['frames'] += [{
            'file_path': f'{frame_id:04d}.png',
            'camera_pose': camera_pose,
            'pose': cur_pose,
            'betas': cur_betas,
            'transl': cur_transl,
        }]

        if vis:
            verts = smpl.forward(torch.from_numpy(cur_pose).unsqueeze(dim=0).float(), \
                                           torch.from_numpy(cur_betas).unsqueeze(dim=0).float())[0]
            mesh = trimesh.Trimesh(verts + cur_transl, smpl.faces)
            rendered_img, _mask = renderer.render(mesh, camera, camera_pose, \
                bkgd=imageio.imread(f'{save_folder}/images/{frame_id:04d}.png'), color=(0.5, 0.8, 1.0))
            vis_imgs += [rendered_img]
            # os.makedirs(f'{save_folder}/vis', exist_ok=True)
            # imageio.imwrite(f'{save_folder}/vis/{frame_id:04d}.png', rendered_img)
    
    if vis:
        imageio.mimwrite(f'{save_folder}/video_vis_moco_flow_data.mp4', vis_imgs, fps=30)
    

    json.dump(moco_dict, open(save_folder + '/train.json', 'w'), indent=4, cls=NumpyEncoder)
    json.dump(moco_dict, open(save_folder + '/val.json', 'w'), indent=4, cls=NumpyEncoder)


def create_init_nerf_data(pkl_path, size, focal=2000, gender='neutral', num_images=120, canonical_pose="frame0"):
    vibe_output = joblib.load(open(pkl_path, 'rb'))
    save_folder = osp.dirname(pkl_path)
    print(f'create init nerf data: {pkl_path}')
    assert len(vibe_output) == 1

    H, W = size
    cam = vibe_output[1]['orig_cam'][0]
    cur_betas = vibe_output[1]['betas'][0]
    if canonical_pose == 'frame0':
        cur_pose = vibe_output[1]['pose'][0]
        save_folder += '/init_nerf'
    elif canonical_pose == 'xpose':
        cur_pose = np.array([-np.pi,   0.,   0.,  0. ,  0. ,  0.5,  0. ,  0. , -0.5,  0. ,  0. ,  0. ,  0. ,  0. ,
                            0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                            0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                            0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                            0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                            0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                            0. ,  0. ,  0. ])
        save_folder += '/init_nerf_xpose'
    else:
        raise NotImplementedError(f"{canonical_pose} not support")

    cur_transl = np.array([cam[2], cam[3], 2*focal/(cam[0]*W)])
    smpl = SMPL(gender)
    verts = smpl.forward(torch.from_numpy(cur_pose).unsqueeze(dim=0).float(), \
                                    torch.from_numpy(cur_betas).unsqueeze(dim=0).float())[0]
    mesh = trimesh.Trimesh(verts + cur_transl, smpl.faces)
    # mesh.export('canonical_mesh.obj')
    dmax = np.max(mesh.vertices, axis=0)
    dmin = np.min(mesh.vertices, axis=0)
    color_map = (mesh.vertices - dmin)/(dmax-dmin)

    moco_dict = {
        'gender': gender,
        'image_height': H,
        'image_width': W,
        'camera_focal': focal,
        'camera_c': np.array([W / 2, H / 2]),
        'D': np.zeros((5,)),
        'frames': []
        }
    renderer = Renderer(resolution=size)
    camera = pyrender.camera.IntrinsicsCamera(
            fx=focal,
            fy=focal,
            cx=W/2,
            cy=H/2,
        )
    vis_imgs = []
    
    render_poses = sample_on_sphere(num_images, np.sqrt(np.sum(cur_transl**2)))
    for frame_id, camera_position in tqdm(enumerate(render_poses)):
        camera_pose = get_camera_pose(camera_position + cur_transl, cur_transl) 

        rendered_img, _mask = renderer.render(mesh, camera, camera_pose, \
            bkgd=255*np.ones((H, W, 4)), color_map=color_map)
        vis_imgs += [rendered_img]
        os.makedirs(f'{save_folder}/images', exist_ok=True)
        imageio.imwrite(f'{save_folder}/images/{frame_id:04d}.png', rendered_img)

        moco_dict['frames'] += [{
            'file_path': f'{frame_id:04d}.png',
            'camera_pose': camera_pose,
            'pose': cur_pose,
            'betas': cur_betas,
            'transl': cur_transl,
        }]

    imageio.mimwrite(f'{save_folder}/video_vis_init_nerf_data.mp4', vis_imgs, fps=30)
    
    json.dump(moco_dict, open(f'{save_folder}/train.json', 'w'), indent=4, cls=NumpyEncoder)
    json.dump(moco_dict, open(f'{save_folder}/val.json', 'w'), indent=4, cls=NumpyEncoder)