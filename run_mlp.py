from config import log_config
from utils import setup_runtime
from mlp import MLPRunner


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--seed", type=int, default=0, help='seed for reproduce')
    parser.add_argument("--lr", type=float, default=1e-4, help='learning rate')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size')
    parser.add_argument("--num_workers", type=int, default=4, help='num workers')
    parser.add_argument("--num_epoch", type=int, default=100, help='num epochs')
    parser.add_argument("--val_epoch", type=int, default=5, help='num epochs')
    parser.add_argument("--i_print", type=int, default=5, help='iterations to print logging info')
    parser.add_argument("--checkpoint", type=str,
                        help='specific weights file to reload')
    parser.add_argument("--embedding_strategy", type=str, default='embed',
                        help='how to embed coordinates')


    # model options
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, help='will load 1/N images from test/val sets')

    return parser

def train():
    parser = config_parser()
    args = parser.parse_args()
    setup_runtime(args)
    log_config(args)
    mlp = MLPRunner(args)
    mlp.train()

if __name__ == "__main__":
    train()