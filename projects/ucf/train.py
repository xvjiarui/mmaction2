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
    parser.add_argument(
        '--suffix', type=str, default='ucf', help='result save suffix')
    parser.add_argument(
        '--ucf-cfg',
        type=str,
        default='projects/ucf/configs/tsn_r18_1x1x3_75e_ucf101_rgb.py')
    parser.add_argument('--pretrained', type=str, help='pretrained file')
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
        '--disable-wandb', action='store_true', help='disable wandb')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.ucf_cfg)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0],
                                osp.splitext(osp.basename(args.ucf_cfg))[0],
                                args.suffix)
    old_cfg = Config.fromfile(args.config)
    if args.pretrained is not None:
        assert osp.exists(args.pretrained)
        weight_path = osp.realpath(args.pretrained).replace(
            'epoch_',
            osp.basename(args.config) + '_ep').replace('.pth', '-backbone.pth')
        if not osp.exists(weight_path):
            os.system(f'MKL_THREADING_LAYER=GNU python '
                      f'tools/convert_weights/convert_to_pretrained.py '
                      f'{args.pretrained} {weight_path}')
        assert osp.exists(weight_path)
        cfg.model.backbone.pretrained = weight_path
        for i, h in enumerate(old_cfg.log_config.hooks):
            if h.type == 'WandbLoggerHook' and not args.disable_wandb:
                mmcv.mkdir_or_exist(f'wandb/{os.path.basename(weight_path)}')
                h.init_kwargs.name = os.path.basename(weight_path)
                h.init_kwargs.resume = False
                h.init_kwargs.dir = f'wandb/{os.path.basename(weight_path)}'
                h.init_kwargs.tags = [*h.init_kwargs.tags, 'ucf', args.ucf_cfg]
                h.init_kwargs.config = cfg.to_dict()
                cfg.log_config.hooks.append(h)
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

    # for #param
    num_trainable_params = len(
        [p for p in model.parameters() if p.requires_grad])
    num_params = len([p for p in model.parameters()])
    logger.info(f'Number of trainable parameters: {num_trainable_params}')
    logger.info(f'Number of total parameters: {num_params}')
    for name, param in model.named_parameters():
        logger.info(f'{name}, grad: {param.requires_grad}')

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
