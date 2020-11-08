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
from torch.utils.data.dataloader import DataLoader

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
        imgs = []
        with open(os.path.join(datadir, split+".txt")) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                name = line.strip()
                pose_path = os.path.join(datadir, name, 'rendering/transforms.json')
                with open(pose_path, 'r') as f:
                    cam_params = json.load(f)['frames']
                    for cam_param in cam_params:
                        img_name = cam_param['file_path']
                        imgs.append(os.path.join(datadir, name, f'rendering/{img_name}.png'))
        self.images = imgs
        print(f'{split} dataset: {len(self.images)}')


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
        img_path = self.images[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR) / 255.
        H, W = img.shape[:2]
        rays_o = self.get_coords2d(H, W)
        ret = {'img': img.astype(np.float32), 'rays_o': rays_o.astype(np.float32)}
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
        self.num_epoch = 1000  # on average, each image is seen by network num_epoch times
        self.val_epoch = 100
        self.lr = 1e-4

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.train_set = BlenderDataset(args.datadir, split='train')
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       shuffle=True)
        self.val_set = BlenderDataset(args.datadir, split='val')
        self.val_idxs = [i for i in range(len(self.val_set))]

        self.i_print = 1000
        self.scale = 10
        self.in_ch = self.mapping_size * 2
        self.B_gauss = torch.randn((self.mapping_size, 2)).to(device)
        self.model = MLP(in_ch=self.in_ch)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def embed(self, x, B):
        if B is None:
            return x
        else:
            x_proj = (2. * np.pi * x).matmul(B.transpose(1, 0))
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)

    def train(self):
        self.model.to(device)
        global_step = 0
        for epoch in range(self.num_epoch):
            for i, data in enumerate(self.train_loader):
                img = data['img'].to(device)
                rays_o = data['rays_o'].to(device)
                embedding = self.embed(rays_o, self.B_gauss)

                embedding = embedding.reshape((-1, embedding.shape[-1]))
                img_pred = self.model.forward(embedding)
                img_pred = img_pred.reshape(img.shape)
                loss = mse(img_pred, img)
                psnr = mse2psnr(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if global_step % self.i_print == 0:
                    print(f'[{epoch} | {global_step}] loss:{loss.item()} psnr:{psnr.item()}')
                    # cv2.imwrite(os.path.join(self.basedir, self.expname, f'train_gt_{epoch}_{global_step}.png'),
                    #             to8b(img[0].detach().cpu().numpy()))
                    cv2.imwrite(os.path.join(self.basedir, self.expname, f'train_{epoch}_{global_step}.png'),
                                    to8b(img_pred[0].detach().cpu().numpy()))
                global_step += 1

            if epoch % self.val_epoch == 0:
                idx = np.random.choice(self.val_idxs, 1)[0]
                data = self.val_set.__getitem__(idx)
                img = torch.tensor(data['img']).to(device)
                rays_o = torch.tensor(data['rays_o']).to(device)
                with torch.no_grad():
                    embedding = self.embed(rays_o, self.B_gauss)

                    embedding = embedding.reshape((-1, embedding.shape[-1]))
                    img_pred = self.model.forward(embedding)
                    img_pred = img_pred.reshape(img.shape)
                    loss = mse(img_pred, img)
                    psnr = mse2psnr(loss)
                    print(f'[{epoch} | val] loss:{loss.item()} psnr:{psnr.item()}')
                    # cv2.imwrite(os.path.join(self.basedir, self.expname, f'val_gt_{epoch}_{global_step}.png'),
                    #             to8b(img.detach().cpu().numpy()))
                    cv2.imwrite(os.path.join(self.basedir, self.expname, f'val_{epoch}_{global_step}.png'),
                                    to8b(img_pred.detach().cpu().numpy()))
