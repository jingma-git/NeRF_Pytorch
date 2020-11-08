from load_blender_data import pose_spherical
from misc import mse, mse2psnr, to8b

import os
import imageio
import json
import torch
import torch.nn as nn
import numpy as np
import cv2


from torch.utils.data.dataset import Dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MLP(nn.Module):
    def __init__(self, in_ch=2, num_layers=4, num_neurons=256):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(in_ch, num_neurons))
        layers.append(nn.ReLU())
        for i in range(1, num_layers-1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_neurons, 3))
        layers.append(nn.Sigmoid())
        self.linears = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x

class BlenderDataset(Dataset):
    def __init__(self, datadir, split='train', testskip=8):
        super(BlenderDataset, self).__init__()
        images, poses, render_poses, hwf, i_split = self.load_blender_data(basedir=datadir,
                                                                           testskip=testskip,
                                                                           split=split)
        print(f'[{split}] Loaded blender images={images.shape} render_poses={render_poses.shape} intrinsics={hwf}')
        self.images = images
        self.poses = poses
        self.focal = hwf[-1]

    def load_blender_data(self, basedir, testskip=1, split='train'):
        splits = [split]
        metas = {}
        for s in splits:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        all_imgs = []
        all_poses = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []
            if s == 'train' or testskip == 0:
                skip = 1
            else:
                skip = testskip

            for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)

        i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)

        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)

        # rotate around z axis like blender
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]],
                                   0)


        focal = focal * (512./800.)

        imgs_half_res = np.zeros((imgs.shape[0], 512, 512, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
        imgs = imgs.astype(np.float32)
        return imgs, poses, render_poses, [512, 512, focal], i_split

    def get_rays_np(self, H, W, focal, c2w):
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                        -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d

    # def __getitem__(self, idx):
    #     img = self.images[idx]
    #     pose = self.poses[idx]
    #     H, W = img.shape[:2]
    #     rays_o, rays_d = self.get_rays_np(H, W, self.focal, pose)
    #     # ret =  {'img':img.transpose((2, 0, 1)),
    #     #         'rays_o': rays_o.transpose((2, 0, 1)),
    #     #         'rays_d': rays_d.transpose((2, 0, 1))}
    #     ret = {'img': img,
    #             'rays_o': rays_o,
    #             'rays_d': rays_d}
    #     return ret

    def get_coords2d(self, H, W):
        coord = np.linspace(0, 1, H, endpoint=False)
        coords = np.stack(np.meshgrid(coord, coord), -1)
        return coords

    def __getitem__(self, idx):
        img = self.images[idx]
        H, W = img.shape[:2]
        rays_o = self.get_coords2d(H, W)
        ret = {'img': img, 'rays_o': rays_o}
        return ret

    def __len__(self):
        return len(self.images)

class MLPRunner(object):
    def __init__(self, args):
        self.basedir = args.basedir
        self.expname = args.expname

        self.num_layers = 4
        self.num_neurons = 256
        self.mapping_size = 256
        self.iters = 2000
        self.lr = 1e-4

        self.train_set = BlenderDataset(args.datadir, split='train')

        self.i_print = 100


    def embed(self, x, B):
        if B is None:
            return x
        else:
            x_proj = (2. * np.pi * x).matmul(B.transpose(1, 0))
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)

    def train(self):
        B_dict = {}
        B_dict['none'] = None
        B_dict['basic'] = torch.eye(2).to(device) # wrap the cooridnates into a circle, make it shift-invariant
        B_gauss = torch.randn((self.mapping_size, 2)).to(device)
        scales = [1, 10, 100]


        for scale in scales:
            B_dict[f'guass_{scale}'] = B_gauss * scale

        train_data = self.train_set.__getitem__(0)
        imageio.imwrite(os.path.join(self.basedir, self.expname, 'gt.png'), to8b(train_data['img']))
        img = torch.tensor(train_data['img'], dtype=torch.float32).to(device)
        rays_o = torch.tensor(train_data['rays_o'], dtype=torch.float32).to(device)
        sh = img.shape

        for k in B_dict:
            embedding = self.embed(rays_o, B_dict[k])

            embedding = embedding.reshape((-1, embedding.shape[-1]))
            in_ch = embedding.shape[-1]
            model = MLP(in_ch)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            for i in range(self.iters):
                img_pred = model.forward(embedding)
                img_pred = img_pred.reshape((sh[0], sh[1], 3))
                loss = mse(img_pred, img)
                psnr = mse2psnr(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % self.i_print == 0:
                    print(f'[{k} | {i}] loss:{loss.item()} psnr:{psnr.item()}')
                    imageio.imwrite(os.path.join(self.basedir, self.expname, f'{k}_{i}.png'), to8b(img_pred.detach().cpu().numpy()))
