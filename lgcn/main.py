from __future__ import division

import os
import torch
import argparse

from mmcv import Config

from utils import (create_logger, set_random_seed, rm_suffix,
                   mkdir_if_no_exists)

from lgcn.models import build_model
from lgcn import build_handler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Linkage-based Face Clustering via GCN')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--phase', choices=['test', 'train'], default='test')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--resume_from',
                        default=None,
                        help='the checkpoint file to resume from')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus(only applicable to non-distributed training)')
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--save_output', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # set cuda
    cfg.cuda = not args.no_cuda and torch.cuda.is_available()

    # set cudnn_benchmark & cudnn_deterministic
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('cudnn_deterministic', False):
        torch.backends.cudnn.deterministic = True

    # update configs according to args
    if not hasattr(cfg, 'work_dir'):
        if args.work_dir is not None:
            cfg.work_dir = args.work_dir
        else:
            cfg_name = rm_suffix(os.path.basename(args.config))
            cfg.work_dir = os.path.join('./data/work_dir', cfg_name)
    mkdir_if_no_exists(cfg.work_dir, is_folder=True)

    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from

    cfg.gpus = args.gpus
    cfg.distributed = args.distributed
    cfg.save_output = args.save_output
    cfg.force = args.force

    logger = create_logger()

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_model(cfg.model['type'], **cfg.model['kwargs'])
    handler = build_handler(args.phase)

    handler(model, cfg, logger)


if __name__ == '__main__':
    main()
