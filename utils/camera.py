import cv2
import torch
import numpy as np

# AABB utils
def rescale_AABB(aabb, scale=[1, 1, 1], offset=[0, 0, 0]):
    if isinstance(aabb, list):
        aabb = np.array(aabb)
    center = (aabb[1] + aabb[0]) / 2
    aabb_tmp = aabb - center
    aabb_tmp *= scale
    aabb = aabb_tmp + center
    aabb[0] = aabb[0] - offset
    aabb[1] = aabb[1] + offset
    return aabb

def convert_AABB_to_verts(aabb):
    aabb_verts = np.array([[aabb[0][0], aabb[0][1], aabb[0][2]],
                            [aabb[0][0], aabb[0][1], aabb[1][2]],
                            [aabb[0][0], aabb[1][1], aabb[0][2]],
                            [aabb[0][0], aabb[1][1], aabb[1][2]],
                            [aabb[1][0], aabb[0][1], aabb[0][2]],
                            [aabb[1][0], aabb[0][1], aabb[1][2]],
                            [aabb[1][0], aabb[1][1], aabb[0][2]],
                            [aabb[1][0], aabb[1][1], aabb[1][2]]])
    return aabb_verts

# rays utils
def gen_ray_directions(H, W, focal, camera_c=[0, 0]):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
        cameta_c: (optional) location of principle point

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    if len(focal) == 1:
        focal = [focal[0], focal[0]]
    directions = \
        torch.stack([(i-camera_c[0])/focal[0], -(j-camera_c[1])/focal[0], -torch.ones_like(i)], -1) # (H, W, 3)
    return directions

def gen_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
             If c2w is None, return rays in camera coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    if c2w is None:
        # Use directions in camera coordinate
        rays_d = directions # (H, W, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_o = torch.zeros_like(directions) # (H, W, 3)
    else:
        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ c2w[:, :3].T # (H, W, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        # The origin of all rays is the camera origin in world coordinate
        rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d
    
def calculate_2d_projections(pts_to_world, camera_to_world, camera_intrinsics):
    """
    Input: 
        pts_to_world: [N, 3]
        camera_to_world: [4, 4]
        camera_intrinstics: [3, 3]
    Return 
        projected_pixels: [N, 2]
    """
    pts_to_world = pts_to_world.transpose() # convert [N, 3] to [3, N]
    pts_to_world_homo = np.vstack([pts_to_world, np.ones((1, pts_to_world.shape[1]), dtype=np.float32)])
    pts_to_camera = np.linalg.inv(camera_to_world) @ pts_to_world_homo # [3, N]
    pts_to_camera = pts_to_camera[:3, :] / pts_to_camera[3, :]
    pts_to_camera[1:, :] *= -1

    projected_pixels = camera_intrinsics @ pts_to_camera[:3, :]
    projected_pixels = projected_pixels[:2, :] / projected_pixels[2, :]
    projected_pixels = projected_pixels.transpose()
    projected_pixels = np.array(projected_pixels, dtype=np.int32)

    return projected_pixels

# create camera class
class Camera():
    def __init__(self, size, K, D=np.zeros(5)):
        self.size = size
        self.K = K
        self.D = D
        self.c2w = None

        focal = [self.K[0][0], self.K[1][1]]
        center = [self.K[0][2], self.K[1][2]]

        self.directions = gen_ray_directions(size[0], size[1], focal, center)

    def get_valid_rays_mask(self, aabb_verts):
        assert self.c2w is not None, 'Camera is not initialized'
        projected_pixels = calculate_2d_projections(aabb_verts, 
                                                    self.c2w,
                                                    self.K)
        valid_rays_mask = np.zeros(self.size)
        hull = cv2.convexHull(projected_pixels.squeeze().astype(int))
        cv2.fillConvexPoly(valid_rays_mask, hull, 255)

        # import imageio
        # imageio.imwrite('valid_rays_mask.png', valid_rays_mask)
        # exit()

        return (valid_rays_mask > 0).flatten()

    def make_rays(self, aabb_verts, idx):
        assert self.c2w is not None, 'Camera is not initialized'

        # get near and far plane
        d = np.sqrt(np.sum((aabb_verts - self.c2w[:3, 3])**2, axis=-1))
        near, far = [min(d), max(d)]

        rays_o, rays_d = gen_rays(self.directions, torch.from_numpy(self.c2w[:3,:4]).float())
        rays = torch.cat([rays_o, rays_d, 
                          near*torch.ones_like(rays_o[:, :1]),
                          far*torch.ones_like(rays_o[:, :1]),
                          idx*torch.ones_like(rays_o[:, :1])],
                          dim = 1) # (H*W, 3+3+1+1)

        return rays

        