import torch
import numpy as np

def mse(pred, gt):
    return torch.mean((pred-gt)**2)

def mse2psnr(x):
    """
    :param x: resonable range is between (0, 1.]
    :return:
    """
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)