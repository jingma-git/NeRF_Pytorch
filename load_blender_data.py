import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w

def trans_matrix(x, y, z):
    trans = np.eye(4)
    trans[:3, 3] = (x, y, z)
    return trans

def euler_matrix(x, y, z):
    rot_x = np.eye(4)
    rot_y = np.eye(4)
    rot_z = np.eye(4)

    rot_x[1, 1] = np.cos(x)
    rot_x[1, 2] = -np.sin(x)
    rot_x[2, 1] = np.sin(x)
    rot_x[2, 2] = np.cos(x)

    rot_y[0, 0] = np.cos(y)
    rot_y[0, 2] = np.sin(y)
    rot_y[2, 0] = -np.sin(y)
    rot_y[2, 2] = np.cos(y)

    rot_z[0, 0] = np.cos(z)
    rot_z[0, 1] = -np.sin(z)
    rot_z[1, 0] = np.sin(z)
    rot_z[1, 1] = np.cos(z)

    return rot_z @ rot_y @ rot_x

def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
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
            # print(fname, np.max(imgs[-1]))
            # cv2.imshow(fname, imgs[-1][..., [2,1,0]])
            # cv2.imshow('alpha', imgs[-1][..., -1])
            # cv2.waitKey(-1)
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # rotate around z axis like blender
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    imgs = imgs[..., :3]*imgs[..., -1:] + (1. - imgs[..., -1:])
    return imgs, poses, render_poses, [H, W, focal], i_split


if __name__ == "__main__":
    angles = np.linspace(-180, 180, 30)
    phi = np.deg2rad(-30)
    for angle in angles:
        theta = np.deg2rad(angle)
        pose = pose_spherical(angle, phi, 4.0).numpy()
        my_pose = euler_matrix(0, -phi, theta) @ trans_matrix(0, 0, 4.0)
        print(angle)
        print("--------------------------------")
        print(pose)
        # print(my_pose)
        print("--------------------------------")