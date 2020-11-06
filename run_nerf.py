from load_blender_data import load_blender_data
from config import config_parser, log_config
from nerf import NeRF, Renderer
from embedder import get_embedder
from utils import load_checkpoint, setup_runtime, cal_model_params
from misc import mse, mse2psnr, to8b

from tqdm import tqdm, trange
import os
import numpy as np
import torch
import imageio

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train():
    # ==================== setup config ==========================
    parser = config_parser()
    args = parser.parse_args()
    setup_runtime(args)
    log_config(args)

    # ==================== create NeRF model =====================
    output_ch = 5 if args.N_importance > 0 else 4
    embed_pos, ch_pos = get_embedder(args.multires)
    embed_dir, ch_dir = get_embedder(args.multires_views)
    net_coarse = NeRF(layers=args.netdepth,
                      hidden_dims=args.netwidth,
                      input_ch=ch_pos,
                      input_ch_views=ch_dir,
                      output_ch=output_ch,
                      use_viewdirs=True)

    net_fine = NeRF(layers=args.netdepth_fine,
                    hidden_dims=args.netwidth_fine,
                    input_ch=ch_pos,
                    input_ch_views=ch_dir,
                    output_ch=output_ch,
                    use_viewdirs=True)
    params = list(net_coarse.parameters())
    params += list(net_fine.parameters())

    optimizer = torch.optim.Adam(params=params,
                                 lr=args.lrate,
                                 betas=(0.9, 0.999))
    neural_renderer = Renderer(embed_pos, embed_dir, net_coarse, net_fine, cfg=args)
    mem_coarse = cal_model_params(net_coarse)
    mem_fine = cal_model_params(net_fine)
    print(f'memory usage: net_coarse:{mem_coarse:.4f} MB  net_fine:{mem_fine:.4f} MB')
    # ==================== load pretrained model =========================================================
    start = 0
    if args.checkpoint is not None:
        start = load_checkpoint(args.checkpoint, net_coarse, net_fine, optimizer)
    log_dir = os.path.join(args.basedir, args.expname)
    ckpts = [os.path.join(log_dir, f) for f in sorted(os.listdir(log_dir)) if 'tar' in f]
    if len(ckpts) > 0:
        print('Found checkpoints', ckpts[-1])
        start = load_checkpoint(ckpts[-1], net_coarse, net_fine, optimizer)

    # ==================== load data ========================================================================
    images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
    print('Loaded blender images={} render_poses={} intrinsics={}'.format(images.shape, render_poses.shape, hwf))
    i_train, i_val, i_test = i_split

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
    render_poses = torch.Tensor(render_poses).to(device)
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    # ==================== train  ===========================================================================
    global_step = start
    for i in trange(start, args.num_iter):
        img_i = np.random.choice(i_train)
        target = images[img_i]
        pose = poses[img_i, :3, :4]

        rgb, disp, acc, extras = neural_renderer.render(H, W, focal, c2w=pose, target_img=target)

        img_loss = mse(rgb, extras['target_rgb'])
        loss = img_loss
        psnr = mse2psnr(img_loss)
        if 'rgb0' in extras:
            img_loss0 = mse(extras['rgb0'], extras['target_rgb'])
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update learning rate
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lr = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


        if global_step % args.i_print==0:
            mem = torch.cuda.max_memory_cached() / (1024**2)
            tqdm.write(f"[TRAIN] iter{global_step}: loss:{loss.item()} PSNR:{psnr.item()} lr:{new_lr} mem:{mem} MB")

        if global_step % args.i_weights==0:
            path = os.path.join(args.basedir, args.expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'net_coarse': net_coarse.state_dict(),
                'net_fine': net_fine.state_dict(),
                'optimizer': optimizer.state_dict()
            }, path)
            print('Saved checkpoint at', path)

        if global_step % args.i_img == 0:
            img_i = np.random.choice(i_val)
            pose = poses[img_i, :3, :4]
            with torch.no_grad():
                rgb, disp, acc, extras = neural_renderer.render(H, W, focal, c2w=pose)
                rgb_img = to8b(rgb.cpu().numpy())
                imageio.imwrite(os.path.join(args.basedir, args.expname, f"{global_step}.png"), rgb_img)
        global_step += 1

if __name__ == "__main__":
        train()