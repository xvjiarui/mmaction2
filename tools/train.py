import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist, set_random_seed

from mmaction import __version__
from mmaction.apis import train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='automatically resume training')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--zip', action='store_true', help='reading from zip file')
    parser.add_argument(
        '--s3', action='store_true', help='reading from ceph s3 file')
    parser.add_argument('--suffix', type=str, help='work_dir suffix')
    parser.add_argument(
        '--disable-wandb', action='store_true', help='disable wandb')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        print('cudnn_benchmark=True')
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.suffix is not None:
        cfg.work_dir = f'{cfg.work_dir}-{args.suffix}'
    for i, h in enumerate(cfg.log_config.hooks):
        if h.type == 'WandbLoggerHook':
            if args.disable_wandb:
                cfg.log_config.hooks.pop(i)
                break
            if args.suffix is not None:
                wandb_dir = cfg.log_config.hooks[i].init_kwargs.dir
                cfg.log_config.hooks[i].init_kwargs.dir = f'{wandb_dir}-' \
                                                          f'{args.suffix}'
            mmcv.mkdir_or_exist(cfg.log_config.hooks[i].init_kwargs.dir)
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    elif args.auto_resume:
        if osp.exists(osp.join(cfg.work_dir, 'latest.pth')):
            cfg.resume_from = osp.join(cfg.work_dir, 'latest.pth')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.zip:
        if cfg.data.train.get('dataset') is not None:
            for i, trans in enumerate(cfg.data.train.dataset.pipeline):
                io_trans = [
                    'RawImageDecode', 'DecordInit', 'OpenCVInit',
                    'RawFrameDecode', 'PyAVInit'
                ]
                if trans.type in io_trans:
                    cfg.data.train.dataset.pipeline[i].io_backend = 'zip'
                    cfg.data.train.dataset.pipeline[i].path_mapping = {
                        'data/': 'data_zip/',
                        'imagenet/2012/train': 'imagenet/2012/train.zip@',
                        'kinetics400/videos_train':
                        'kinetics400.zip@/kinetics400/train',
                        'kinetics400/train':
                        'kinetics400.zip@/kinetics400/train'
                    }
        else:
            for i, trans in enumerate(cfg.data.train.pipeline):
                io_trans = [
                    'RawImageDecode', 'DecordInit', 'OpenCVInit',
                    'RawFrameDecode', 'PyAVInit'
                ]
                if trans.type in io_trans:
                    cfg.data.train.pipeline[i].io_backend = 'zip'
                    cfg.data.train.pipeline[i].path_mapping = {
                        'data/': 'data_zip/',
                        'imagenet/2012/train': 'imagenet/2012/train.zip@',
                        'kinetics400/videos_train':
                        'kinetics400.zip@/kinetics400/train',
                        'kinetics400/train':
                        'kinetics400.zip@/kinetics400/train'
                    }
    if args.s3:
        os.system('cp .dev/ceph_s3/petreloss.conf $HOME/')
        os.system('cp .dev/ceph_s3/.s3cfg $HOME/')
        if cfg.data.train.get('dataset') is not None:
            for i, trans in enumerate(cfg.data.train.dataset.pipeline):
                io_trans = [
                    'RawImageDecode', 'DecordInit', 'OpenCVInit',
                    'RawFrameDecode', 'PyAVInit'
                ]
                if trans.type in io_trans:
                    cfg.data.train.dataset.pipeline[i].io_backend = 'petrel'
                    cfg.data.train.dataset.pipeline[i].path_mapping = {
                        'data/': 's3://data/',
                        'kinetics400/videos_train': 'kinetics400/train',
                    }
        else:
            for i, trans in enumerate(cfg.data.train.pipeline):
                io_trans = [
                    'RawImageDecode', 'DecordInit', 'OpenCVInit',
                    'RawFrameDecode', 'PyAVInit'
                ]
                if trans.type in io_trans:
                    cfg.data.train.pipeline[i].io_backend = 'petrel'
                    cfg.data.train.pipeline[i].path_mapping = {
                        'data/': 's3://data/',
                        'kinetics400/videos_train': 'kinetics400/train',
                    }

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config: {cfg.text}')
    logger.info(f'Config.pretty_text: {cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    logger.info(f'Model: {str(model)}')

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        if args.validate:
            warnings.warn('val workflow is duplicated with `--validate`, '
                          'it is recommended to use `--validate`. see '
                          'https://github.com/open-mmlab/mmaction2/pull/123')
        val_dataset = copy.deepcopy(cfg.data.val)
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__, config=cfg.text)

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
