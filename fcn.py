from load_blender_data import pose_spherical
from misc import mse, mse2psnr, to8b
from embedder import get_embedder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


import os
import numpy as np
import cv2
import json
import imageio
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class FCN(nn.Module):
    """
    input: image-B x H x W x 3, view-direction B x H x W x 3
    output
    """
    def __init__(self, layers=8, in_channels=6, out_channels=3, max_channels=256):
        super(FCN, self).__init__()
        self.bottle_neck = layers // 2
        hidden_channels = [32 * (2**i) for i in range(self.bottle_neck)]
        print(f"hidden_channels: {hidden_channels}")
        # encoders
        for i in range(0, self.bottle_neck):
            self.__setattr__(f'e{i}', nn.Conv2d(in_channels, hidden_channels[i], kernel_size=3, stride=2, padding=1))
            in_channels = hidden_channels[i]

        # decoders with skip connection
        for i in range(self.bottle_neck-1, -1, -1):
            if i == 0:
                self.__setattr__(f'd{i}',
                                 nn.ConvTranspose2d(hidden_channels[i]+hidden_channels[i+1], out_channels,
                                                    kernel_size=2, stride=2))
            elif i == self.bottle_neck-1:
                self.__setattr__(f'd{i}',
                                 nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i],
                                                    kernel_size=2, stride=2))
            else:
                self.__setattr__(f'd{i}',
                                 nn.ConvTranspose2d(hidden_channels[i]+hidden_channels[i+1], hidden_channels[i],
                                                    kernel_size=2, stride=2))

    def forward(self, rays_o, rays_d):
        x = torch.cat([rays_o, rays_d], dim=1)
        encodings = []
        for i in range(0, self.bottle_neck):
            encoder_i = self.__getattr__(f'e{i}')
            x = encoder_i(x)
            encodings.append(x)
            # print(f'encoder{i}: {encodings[i].shape}')


        for i in range(self.bottle_neck-1, -1, -1):
            decoder_i = self.__getattr__(f'd{i}')
            if i != (self.bottle_neck-1):
                x = torch.cat([x, encodings[i]], dim=1)
            x = decoder_i(x)
            # print(f'decoder{i}: {x.shape}')
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

    def __getitem__(self, idx):
        img = self.images[idx]
        pose = self.poses[idx]
        H, W = img.shape[:2]
        rays_o, rays_d = self.get_rays_np(H, W, self.focal, pose)
        # ret =  {'img':img.transpose((2, 0, 1)),
        #         'rays_o': rays_o.transpose((2, 0, 1)),
        #         'rays_d': rays_d.transpose((2, 0, 1))}
        ret = {'img': img,
                'rays_o': rays_o,
                'rays_d': rays_d}
        return ret

    def __len__(self):
        return len(self.images)

class FCNRunner(object):
    def __init__(self, args):
        self.basedir = args.basedir
        self.expname = args.expname

        self.train_set = BlenderDataset(args.datadir, 'train', args.testskip)
        self.val_set = BlenderDataset(args.datadir, 'val', args.testskip)
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers)
        self.val_loader = DataLoader(self.val_set,
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=args.num_workers)

        embedder_pos, in_ch = get_embedder(multires=args.multires)
        embedder_dir, in_ch_dir = get_embedder(multires=args.multires_views)
        in_channels = in_ch + in_ch_dir
        self.model = FCN(layers=8, in_channels=in_channels)
        self.embedder_pos = embedder_pos
        self.embedder_dir = embedder_dir
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                 lr=args.lr,
                                 betas=(0.9, 0.999))

        self.checkpoint = args.checkpoint
        self.num_epoch = args.num_epoch
        self.val_epoch = args.val_epoch
        self.i_print = args.i_print


    def load_checkpoint(self, path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['start_epoch']
        return start_epoch

    def save_checkpoint(self, path, epoch):
        torch.save({
            'start_epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def train(self):
        global_step = 0
        start_epoch = 0
        if self.checkpoint is not None:
            start_epoch = self.load_checkpoint(self.checkpoint)
            global_step = start_epoch * len(self.train_set)

        log_dir = os.path.join(self.basedir, self.expname)
        ckpts = [os.path.join(log_dir, f) for f in sorted(os.listdir(log_dir)) if '.pth' in f]
        if len(ckpts) > 0:
            print('Found checkpoints', ckpts[-1])
            start_epoch = self.load_checkpoint(ckpts[-1])
            global_step = start_epoch * len(self.train_set)

        self.model.to(device)
        start_time = time.time()
        for epoch in range(start_epoch, self.num_epoch):
            for step, data in enumerate(self.train_loader):
                time0 = time.time()
                gt_img = data['img'].to(device)
                rays_o = data['rays_o'].to(device)
                rays_d = data['rays_d'].to(device)
                embedding_pos = self.embedder_pos(rays_o).permute((0, 3, 1, 2))
                embedding_dir = self.embedder_dir(rays_d).permute((0, 3, 1, 2))

                img = self.model.forward(embedding_pos, embedding_dir)
                img_loss = mse(img, gt_img.permute((0, 3, 1, 2)))

                loss = img_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                elapsed = time.time()-time0
                if global_step % self.i_print==0:
                    print(f'[Train {global_step}] loss:{loss.item()} time:{elapsed} sec')
                global_step += 1

            if epoch % self.val_epoch==0:
                with torch.no_grad():
                    data = next(self.val_loader.__iter__())
                    gt_img = data['img'].to(device)
                    rays_o = data['rays_o'].to(device)
                    rays_d = data['rays_d'].to(device)
                    embedding_pos = self.embedder_pos(rays_o).permute((0, 3, 1, 2))
                    embedding_dir = self.embedder_dir(rays_d).permute((0, 3, 1, 2))

                    img = self.model.forward(embedding_pos, embedding_dir)

                    img_loss = mse(img, gt_img.permute((0, 3, 1, 2)))
                    print(f'[Val {global_step}] loss:{img_loss.item()}')

                    rgb_img = to8b(img[0].cpu().numpy().transpose((1, 2, 0)))
                    imageio.imwrite(os.path.join(self.basedir, self.expname, f"{global_step}.png"), rgb_img)


                self.save_checkpoint(os.path.join(self.basedir, self.expname, f"{epoch:03d}.pth"), epoch)

        total_time = (time.time() - start_time) / 60.0
        print(f'{total_time:.4f} min')