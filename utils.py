import torch
import os
import numpy as np
import random

def load_checkpoint(path, net_coarse, net_fine, optimizer):
    print('Loading checkpoint from ', path)
    ckpt = torch.load(path)
    start = ckpt['global_step']
    optimizer.load_state_dict(ckpt['optimizer'])
    net_coarse.load_state_dict(ckpt['net_coarse'])
    net_fine.load_state_dict(ckpt['net_fine'])
    return start

def setup_runtime(args):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # Setup random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_default_dtype(torch.float32)

def cal_model_params(model):
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = (mem_bufs + mem_params) / (1024 **2) # 1024bytes=1KB 1024KB=1MB
    return mem