import os
import torch
import torch.nn as nn
import argparse
import datetime
import glob
import torch.distributed as dist
from dataset.data_utils import build_dataloader
from test_util import test_model
from model.RoofSeg import RoofSeg
from torch import optim
from utils import common_utils
from model import model_utils


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset', help='dataset path')
    parser.add_argument('--dataset', type=str, default='Roofpc3d', help='Roofpc3d/RoofN3d')
    parser.add_argument('--cfg_file', type=str, default='./roofseg1.yaml', help='model config for training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--gpu', type=str, default='0', help='gpu for training')
    parser.add_argument('--test_tag', type=str, default='test_roofN3d1.7k_xyzfea', help='extra tag for this experiment')
    parser.add_argument('--cluster', default=False, action='store_true', help='output clustering results for testing')


    args = parser.parse_args()
    cfg = common_utils.cfg_from_yaml_file(args.cfg_file)
    return args, cfg

def main():
    args, cfg = parse_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    extra_tag = args.test_tag
    output_dir = cfg.ROOT_DIR / 'output' / extra_tag
    assert output_dir.exists(), '%s does not exist!!!' % str(output_dir)
    ckpt_dir = output_dir #/ 'ckpt'
    output_dir = output_dir / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'log.txt'
    logger = common_utils.create_logger(log_file)


    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_config_to_file(cfg, logger=logger)

    test_loader = build_dataloader(args.data_path, args.dataset, args.batch_size, cfg.DATA, training=False, logger=logger)
    
    input_num = 3
    net = RoofSeg(cfg.MODEL, input_num, args.cluster)
    net.cuda()
    net.eval()

    ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        model_utils.load_params(net, ckpt_list[-1], logger=logger)

    logger.info('**********************Start testing**********************')
    logger.info(net)
    test_model(net, test_loader, logger, args.test_tag)


if __name__ == '__main__':
    main()
