import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class NeRF(nn.Module):
    def __init__(self, layers=8, hidden_dims=256,
                 input_ch=3, input_ch_views=3,
                 output_ch=4, skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.layers = layers
        self.hidden_dims = hidden_dims
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs =use_viewdirs
        # position
        pts_layers = [nn.Linear(input_ch, hidden_dims)]
        for i in range(layers-1):
            if i in skips:
                pts_layers.append(nn.Linear(hidden_dims+input_ch, hidden_dims))
            else:
                pts_layers.append(nn.Linear(hidden_dims, hidden_dims))
        self.pts_layers = nn.ModuleList(pts_layers)

        # view
        self.view_layers = nn.ModuleList([nn.Linear(input_ch_views+hidden_dims, hidden_dims//2)])
        if self.use_viewdirs:
            self.feature_layer = nn.Linear(hidden_dims, hidden_dims)
            self.rgb_layer = nn.Linear(hidden_dims//2, 3)
            self.alpha_layer = nn.Linear(hidden_dims, 1)
        else:
            self.output_layer = nn.Linear(hidden_dims, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_layers):
            h = self.pts_layers[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_layer(h)

            feature = self.feature_layer(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.view_layers):
                h = self.view_layers[i](h)
                h = F.relu(h)
            rgb = self.rgb_layer(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_layer(h)
        return outputs


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    pass


class Renderer:
    def __init__(self, embedder_pos, embedder_view, net_coarse, net_fine, cfg, optimizer=None):
        """
        N_rand: number of shooting rays for each image
        N_samples: number of different times to sample along each ray
        N_importance: number of additional times to sample along each ray, only passed to network_fine
        chunk: how many rays are sent to GPU each time
        netchunk: how many sampled position are sent to GPU each time

        :return RGB and Mask image
        """
        self.embedder_pos = embedder_pos
        self.embedder_view = embedder_view
        self.net_coarse = net_coarse.to(device)
        self.net_fine = net_fine.to(device)

        self.near = cfg.near
        self.far = cfg.far

        self.N_rand = cfg.N_rand
        self.N_samples = cfg.N_samples
        self.N_importance = cfg.N_importance
        self.chunk = cfg.chunk
        self.netchunk = cfg.netchunk

    def get_rays(self, H, W, focal, c2w):
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                              torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
        i = i.t().to(c2w.device)
        j = j.t().to(c2w.device)
        dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0):
        # eqn3
        def raw2alpha(density, dists):
            alpha = 1. - torch.exp(-F.relu(density) * dists)
            return alpha

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        end = torch.Tensor([1e10]).expand(dists[..., :1].shape).to(device)
        dists = torch.cat([dists, end], -1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])
        alpha = raw2alpha(raw[..., 3], dists)
        ones = torch.ones((alpha.shape[0], 1)).to(device)
        Ti = torch.cumprod(torch.cat([ones, 1. - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        weights = alpha * Ti

        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * z_vals, dim=-1)
        # disp_map = 1. / torch.max(1e-10 * ones, depth_map / torch.sum(weights, -1))
        # disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        disp_map =  torch.sum(weights, -1) / depth_map
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1. - acc_map[..., None])
        return rgb_map, disp_map, acc_map, weights, depth_map

    def batchify_rays(self, rays_flat, chunk=1024*32):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(rays_flat[i:i + chunk])
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render_rays(self, ray_batch):
        # Time integration along each sampled ray
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        z_vals = near * (1. - t_vals) + far * t_vals
        z_vals = z_vals.expand([N_rays, self.N_samples])
        mids = .5 * (z_vals[..., 1:]+z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape).to(device)
        z_vals = lower + (upper - lower) * t_rand

        pts =  rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts_flat = pts.reshape([-1, pts.shape[-1]])
        embeddings_pts = self.embedder_pos(pts_flat) # 1024 * 64, 3

        dirs = viewdirs[:, None].expand(pts.shape)
        dirs_flat = dirs.reshape([-1, dirs.shape[-1]])
        embeddings_dirs = self.embedder_view(dirs_flat)

        embeddings = torch.cat([embeddings_pts, embeddings_dirs], -1)
        #===================== Time Integration ================================
        def batch_run(fn, inputs, total_amts, chunk):
            input_list = []
            for i in range(0, total_amts, chunk):
                input_list.append(fn(inputs[i:i+chunk]))
            return torch.cat(input_list, 0)
        raw = batch_run(self.net_coarse, embeddings, embeddings.shape[0], self.netchunk)
        raw = torch.reshape(raw, (N_rays, self.N_samples, raw.shape[-1]))
        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d)

        ret = {'rgb_map': rgb_map,
               'disp_map': disp_map,
               'acc_map': acc_map,
               'raw': raw}
        return ret

    def render(self, H, W, focal, c2w, target_img=None):
        """
        :param target_img:
        :return: if target_img is not None, render whole image by shooting HxW rays instead of self.N_rand rays
        """
        rays_o, rays_d = self.get_rays(H, W, focal, c2w)
        if target_img is not None:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)
            coords = torch.reshape(coords, [-1, 2])
            select_inds = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)
            select_coords = coords[select_inds].long()
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
            target_s = target_img[select_coords[:, 0], select_coords[:, 1]]

        sh = rays_d.shape
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        near = self.near * torch.ones_like(rays_d[..., :1])
        far = self.far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)
        all_ret = self.batchify_rays(rays)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'disp_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        if target_img is not None:
            ret_dict['target_rgb'] = target_s
        return ret_list + [ret_dict]
