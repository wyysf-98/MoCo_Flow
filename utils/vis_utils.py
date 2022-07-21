import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from plyfile import PlyData, PlyElement

## vis point cloud
def write_ply(points, filename, text=False):
    """ 
    input: Nx3, write points to filename as PLY format. 
    """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)
        
def write_ply_rgb(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2], points[i,3], points[i,4], points[i,5]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)

## vis depth
def visualize_depth(depth, mi=None, ma=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    if mi is not None and ma is not None:
        x = np.nan_to_num(x, nan=ma) # change nan to 0
    else:
        x = np.nan_to_num(x) # change nan to 0
        mi = np.min(x) # get minimum depth
        ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

## camera pose
def pose_spherical(theta, phi, radius, offset=[0, 0, 0], vec_up=None):
    """
    Generate spherical rendering poses, from NeRF. 
    :return: r (3,), t (3,)
    """
    def _trans_t(t):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, t],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def _rot_phi(phi):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def _rot_theta(th):
        return np.array(
            [
                [np.cos(th), 0, -np.sin(th), 0],
                [0, 1, 0, 0],
                [np.sin(th), 0, np.cos(th), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    c2w = _trans_t(radius)
    c2w = _rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = _rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        np.array(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        @ c2w
    )
    if vec_up is not None:
        vec_up = vec_up / np.linalg.norm(vec_up)
        vec_1 = np.array([vec_up[0], -vec_up[2], vec_up[1]])
        vec_2 = np.cross(vec_up, vec_1)

        trans = np.eye(4, 4, dtype=np.float32)
        trans[:3, 0] = vec_1
        trans[:3, 1] = vec_2
        trans[:3, 2] = vec_up
        c2w = trans @ c2w
    # OpenGL -> OpenCV
    # c2w = c2w @ np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    if offset is not None:
        c2w[:3, 3] += offset
    return c2w

def create_spheric_poses(num=30, radius=2.0, center=[0, 0, 0], vec_up=None):
    """
    create spheric poses for vis
    """
    render_poses_0 = np.stack([pose_spherical(angle,   0.0, radius, center, vec_up) for angle in np.linspace(0,360,num+1)[:-1]], 0) # [N, 4, 4]
    render_poses_1 = np.stack([pose_spherical(angle, -15.0, radius, center, vec_up) for angle in np.linspace(0,360,num+1)[:-1]], 0) # [N, 4, 4]
    render_poses_2 = np.stack([pose_spherical(angle, -30.0, radius, center, vec_up) for angle in np.linspace(0,360,num+1)[:-1]], 0) # [N, 4, 4]
    render_poses = np.concatenate([render_poses_0, render_poses_1, render_poses_2], 0)

    return render_poses

def create_bullet_effect_poses(radius=2.0, move_dist=0.5, init_angle=0):
    """
    create bullet effect poses for vis
    """
    render_poses_0 = np.stack([pose_spherical(angle+init_angle, 0, radius) for angle in np.linspace(0,360,40+1)[:-1]], 0) # [N, 4, 4]
    render_poses_1 = np.stack([pose_spherical(angle+init_angle, 0, rad) for angle, rad in zip(np.linspace(0,360,40+1)[:-1], np.linspace(radius,radius-move_dist,40+1)[:-1])], 0) # [N, 4, 4]
    render_poses_2 = np.stack([pose_spherical(angle+init_angle, 0, rad) for angle, rad in zip(np.linspace(0,360,40+1)[:-1], np.linspace(radius-move_dist,radius+move_dist,40+1)[:-1])], 0) # [N, 4, 4]
    render_poses   = np.concatenate([render_poses_0, render_poses_1, render_poses_2], 0)

    return render_poses
