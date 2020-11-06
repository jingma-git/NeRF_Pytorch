from nerf import *

def testRaw2Outs():
    num_rays = 1024
    num_samples = 64
    raw = torch.randn([num_rays, num_samples, 4])


def testEmbedder():
    from embedder import get_embedder
    x = torch.randn(2, 512, 512, 3)
    embedder, out_dim = get_embedder(10)
    output = embedder(x)
    print(out_dim, output.shape)

def testPSNR():
    from misc import mse2psnr
    loss = torch.linspace(1e-4, 1., 10)
    psnr = mse2psnr(loss)
    print(psnr)


def testFCN():
    from gpu_util import log_mem
    from fcn import FCN, device
    bs = 1
    image = torch.randn([bs, 3, 512, 512]).to(device)
    rays_o = torch.randn([bs, 3, 512, 512]).to(device)
    rays_d = torch.randn([bs, 3, 512, 512]).to(device)

    model = FCN()
    model.to(device)
    out = model(image, rays_o, rays_d)
    # loss = out.sum()
    # loss.backward()

    # mem_log =log_mem(model, (image, rays_o, rays_d), exp='fcn')
    #
    # for log in mem_log:
    #     print(log)

def get_coords2d(H, W):
    coord = np.linspace(0, 1, H, endpoint=False)
    coords = np.stack(np.meshgrid(coord, coord), -1)
    print(coords)

if __name__ == "__main__":
    # testEmbedder()
    # testPSNR()
    # testFCN()
    get_coords2d(5, 5)