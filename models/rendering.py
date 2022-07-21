import torch
from torch import searchsorted


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def nof_inference(xyz_,
                  ind_,
                  nof_embeddings,
                  nof_model):
    """
    Inputs:
        xyz_: (N_rays, N_samples_, 3) sampled positions
                N_samples_ is the number of sampled points in each ray;
                            = N_samples for coarse model
                            = N_samples+N_importance for fine model
        ind_: (N_rays, N_samples_) image indexs
        nof_embeddings: embedding modules for NoF, [xyz, ind]
        nof_model: NoF model (backward or foreward)
    Outputs:
        output_xyz: (N_rays, 3) the transformed positions
    """
    N_rays_ = xyz_.shape[0]
    N_samples_ = xyz_.shape[1]

    # Embed xyz and image index
    xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
    xyz_embedded = torch.zeros((xyz_.shape[0], nof_model.in_channels_xyz)).to(xyz_.device)
    xyz_embedded_ = nof_embeddings[0](xyz_)
    xyz_embedded[:, :xyz_embedded_.shape[1]] = xyz_embedded_
    ind_embedded_ = nof_embeddings[1](ind_)
    ind_embedded = torch.repeat_interleave(ind_embedded_, repeats=N_samples_, dim=0)
    input_nof = torch.cat([xyz_embedded, ind_embedded], -1)

    # repeat image index
    img_ind = torch.repeat_interleave(ind_, repeats=N_samples_, dim=0).view(-1)

    # Perform model inference to get final output xyz
    output_xyz = nof_model(input_nof, xyz_, img_ind=img_ind)

    return output_xyz.view(N_rays_, N_samples_, -1)


def nerf_inference(xyz_, 
                   ind_, 
                   dir_, 
                   z_vals, 
                   noise_std, 
                   nerf_embeddings,
                   nerf_model, 
                   background=None, 
                   weights_only=False,
                   activate_type='relu'):
    """
    Inputs:
        xyz_: (N_rays, N_samples_, 3) sampled positions
                N_samples_ is the number of sampled points in each ray;
                            = N_samples for coarse model
                            = N_samples+N_importance for fine model
        ind_: (N_rays, N_samples_) image indexs
        dir_: (N_rays, 3) ray directions
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        noise_std: factor to perturb the model's prediction of sigma, default is 1
        nerf_embeddings: embedding modules for NeRF, [xyz, ind, dir]
        nerf_model: NeRF model (coarse or fine)
        dir_embedded: (N_rays, embed_dir_channels) embedded view directions
        ind_embedded: (N_rays, embed_ind_channels) embedded image index
        background: (N_rays, N_samples_, 3) background color
        volume_mask: (N_rays, N_samples_) volume mask
        weights_only: do inference on sigma only or not, default is False
    Outputs:
        if weights_only:
            weights: (N_rays, N_samples_): weights of each sample
        else:
            rgb_final: (N_rays, 3) the final rgb image
            depth_final: (N_rays) depth map
            weights: (N_rays, N_samples_): weights of each sample
    """
    N_rays_ = xyz_.shape[0]
    N_samples_ = xyz_.shape[1]
    dir_ = dir_.view(-1, 3) # (N_rays*N_samples_, 3)

    # Embed xyz and paded to fit the model's in_channels_xyz
    xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
    xyz_embedded = torch.zeros((xyz_.shape[0], nerf_model.in_channels_xyz)).to(xyz_.device)
    xyz_embedded_ = nerf_embeddings[0](xyz_)
    xyz_embedded[:, :xyz_embedded_.shape[1]] = xyz_embedded_
    input_nerf = xyz_embedded
    
    if not weights_only: # Embed directions or image index
        if nerf_model.extra_feat_type == 'ind': # Embed image index
            ind_embedded_ = torch.repeat_interleave(nerf_embeddings[1](ind_), repeats=N_samples_, dim=0)
            ind_embedded = torch.zeros((ind_embedded_.shape[0], nerf_model.extra_feat_dim)).to(xyz_.device)
            ind_embedded[:, :ind_embedded_.shape[1]] = ind_embedded_ # (N_rays*N_samples_, embed_dir_channels)
            input_nerf = torch.cat([input_nerf, ind_embedded], 1)
        elif nerf_model.extra_feat_type == 'dir': # Embed view dir
            dir_embedded_ = torch.repeat_interleave(nerf_embeddings[2](dir_), repeats=N_samples_, dim=0)
            dir_embedded = torch.zeros((dir_embedded_.shape[0], nerf_model.extra_feat_dim)).to(xyz_.device)
            dir_embedded[:, :dir_embedded_.shape[1]] = dir_embedded_ # (N_rays*N_samples_, embed_dir_channels)
            input_nerf = torch.cat([input_nerf, dir_embedded], 1)

    # repeat image index
    img_ind = torch.repeat_interleave(ind_, repeats=N_samples_, dim=0).view(-1)
    
    # Perform model inference to get rgb and raw sigma
    out = nerf_model(input_nerf, sigma_only=weights_only, img_ind=img_ind)

    if weights_only:
        sigmas = out.view(N_rays_, N_samples_)
    else:
        rgbsigma = out.view(N_rays_, N_samples_, 4)
        rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
        sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

    # Convert these values using volume rendering
    deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

    # compute alpha by formula in nerf paper
    if activate_type == 'relu':
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
    elif activate_type == 'softplus':
        alphas = 1-torch.exp(-deltas*torch.nn.Softplus()(sigmas+noise)) # (N_rays, N_samples_)
    else:
        raise ValueError('activation layer type: %s not support'%activate_type)

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
    weights = \
        alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
    weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    if weights_only:
        return weights, alphas

    # compute final weighted outputs
    rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
    depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

    if background is not None:
        rgb_final = rgb_final + background*(1-weights_sum.unsqueeze(-1))

    return rgb_final, depth_final, weights, alphas


def render_rays(rays,
                background,
                nerf_embeddings,
                nerf_models,
                nof_embeddings=None,
                nof_models=None,
                chain_local=False,
                chain_global=False,
                N_samples=64,
                N_importance=0,
                use_disp=False,
                perturb=0,
                noise_std=1,
                nerf_activate_type='relu',
                test_time=False,
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        rays: (N_rays, 3+3+2+1), ray origins, directions and near, far depth bounds, image index, [optional chained image index]
        background: (N_rays, 3), background color for each ray
        nerf_embeddings: list of embedding models of origin and direction defined in embedding.py
        nerf_models: list of NeRF models (coarse and fine) defined in nerf.py
        nof_embeddings: list of embedding models of origin and index defined in embedding.py
        nof_models: list of NoF models (coarse and fine) defined in nof.py
        chain_local: wheter to use NoF local chain regulization term
        chain_global: wheter to use NoF global chain regulization term
        N_samples: number of coarse samples per ray
        N_importance: number of fine samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        nerf_activate_type: activate type of last layer in NeRF
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    # Decompose the input rays
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    img_ind = rays[:,8:9] # (N_rays, 1) image index
    if nof_models is not None and chain_global:
        chained_img_ind = rays[:, 9:10] # (N_rays, 1) random chained image index

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    # from utils.vis_utils import write_ply
    # write_ply(xyz_coarse_sampled.view(-1, 3).cpu().numpy(), 'xyz_coarse_sampled_%s.ply'%int(img_ind[0].item()))
    # exit(0)

    # NoF prediction
    if nof_models is not None:
        # print(img_ind, chained_img_ind)
        bw_nof = nof_models[0]
        xyz_canonical_coarse_sampled = nof_inference(xyz_coarse_sampled, img_ind, nof_embeddings, bw_nof) 
        
        if chain_local and not test_time:
            fw_nof = nof_models[1]
            xyz_recon_coarse_sampled = nof_inference(xyz_canonical_coarse_sampled, img_ind, nof_embeddings, fw_nof) 
        
        if chain_global and not test_time:
            xyz_chained_coarse_sampled = nof_inference(xyz_canonical_coarse_sampled, chained_img_ind, nof_embeddings, fw_nof)
            xyz_chained_canonical_coarse_sampled = nof_inference(xyz_chained_coarse_sampled, chained_img_ind, nof_embeddings, bw_nof)
            xyz_chained_recon_coarse_sampled = nof_inference(xyz_chained_canonical_coarse_sampled, img_ind, nof_embeddings, fw_nof)
            
        coarse_nerf_input = xyz_canonical_coarse_sampled
    else:
        coarse_nerf_input = xyz_coarse_sampled

    # coarse NeRF prediction
    nerf_model_coarse = nerf_models[0]
    if N_importance > 0 and test_time:
        weights_coarse, alphas_coarse = \
            nerf_inference(coarse_nerf_input, img_ind, rays_d, z_vals, noise_std, nerf_embeddings, nerf_model_coarse, \
                background=background, weights_only=True, activate_type=nerf_activate_type)
        result = {'opacity_coarse': weights_coarse.sum(1)}

    else:
        rgb_coarse, depth_coarse, weights_coarse, alphas_coarse = \
            nerf_inference(coarse_nerf_input, img_ind, rays_d, z_vals, noise_std, nerf_embeddings, nerf_model_coarse, \
                background=background, weights_only=False, activate_type=nerf_activate_type)
        result = {'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'opacity_coarse': weights_coarse.sum(1)}

    if nof_models is not None and not test_time:
        # compute local and global disparity
        mask_coarse = alphas_coarse.ge(0.01)
        if not torch.any(mask_coarse):
            mask_coarse = torch.ones_like(mask_coarse).bool()
        if chain_local:
            nof_local_disp_coarse = torch.abs(xyz_coarse_sampled - xyz_recon_coarse_sampled)[mask_coarse] # [N_rays_masked, 3]
            result['nof_local_disp_coarse'] = torch.mean(nof_local_disp_coarse, dim=1)  # [N_rays_masked, 1]
        if chain_global:
            nof_global_disp_coarse = torch.abs(xyz_coarse_sampled - xyz_chained_recon_coarse_sampled)[mask_coarse]# [N_rays_masked, 3]
            result['nof_global_disp_coarse'] = torch.mean(nof_global_disp_coarse, dim=1)  # [N_rays_masked, 1]

    # fine NeRF prediction
    if N_importance > 0: # sample points for fine model
        # Extract fine model
        nerf_model_fine = nerf_models[1]

        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)
                    # values are interleaved actually, so maybe can do better than sort?

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

            
        # nof prediction
        if nof_models is not None:
            xyz_canonical_fine_sampled = nof_inference(xyz_fine_sampled, img_ind, nof_embeddings, bw_nof) 

            if chain_local and not test_time:
                xyz_recon_fine_sampled = nof_inference(xyz_canonical_fine_sampled, img_ind, nof_embeddings, fw_nof) 

            if chain_global and not test_time:
                xyz_chained_fine_sampled = nof_inference(xyz_canonical_fine_sampled, chained_img_ind, nof_embeddings, fw_nof)
                xyz_chained_canonical_fine_sampled = nof_inference(xyz_chained_fine_sampled, chained_img_ind, nof_embeddings, bw_nof)
                xyz_chained_recon_fine_sampled = nof_inference(xyz_chained_canonical_fine_sampled, img_ind, nof_embeddings, fw_nof)

            fine_nerf_input = xyz_canonical_fine_sampled
        else:
            fine_nerf_input = xyz_fine_sampled

        rgb_fine, depth_fine, weights_fine, alphas_fine = \
            nerf_inference(fine_nerf_input, img_ind, rays_d, z_vals, noise_std, nerf_embeddings, nerf_model_fine, \
                background=background, weights_only=False, activate_type=nerf_activate_type)
        # from utils.vis_utils import write_ply
        # import time
        # mask_fine = alphas_fine.ge(0.01)
        # write_ply(fine_nerf_input[mask_fine].view(-1, 3).detach().cpu().numpy(), 'fine_nerf_input_%s.ply'%int(time.time()))
        # # # exit(0)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

        if nof_models is not None and not test_time:
            # compute local and chain disparity
            mask_fine = alphas_fine.ge(0.01)
            if not torch.any(mask_fine):
                mask_fine = torch.ones_like(mask_fine).bool()
            if chain_local:
                nof_local_disp_fine = torch.abs(xyz_fine_sampled - xyz_recon_fine_sampled)[mask_fine] # [N_rays_masked, 3]
                result['nof_local_disp_fine'] = torch.mean(nof_local_disp_fine, dim=1)  # [N_rays_masked, 1]
            if chain_global:
                nof_global_disp_fine = torch.abs(xyz_fine_sampled - xyz_chained_recon_fine_sampled)[mask_fine]# [N_rays_masked, 3]
                result['nof_global_disp_fine'] = torch.mean(nof_global_disp_fine, dim=1)  # [N_rays_masked, 1]

    return result

