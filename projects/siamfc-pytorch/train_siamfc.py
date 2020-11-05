import argparse
import os
import os.path as osp
import time

import mmcv
import numpy as np
import torch
from got10k.datasets import GOT10k
from got10k.experiments import ExperimentOTB
from mmcv import Config, DictAction
from mmcv.runner import set_random_seed
from siamfc import TrackerSiamFC, default_cfg

from mmaction.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file')
    parser.add_argument('--pretrained', type=str, help='pretrained file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--suffix', type=str, default='siamfc', help='result save suffix')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--disable-wandb', action='store_true', help='disable wandb')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(default_cfg)
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
                                osp.splitext(osp.basename(args.config))[0])
    cfg.suffix = args.suffix
    cfg.checkpoint = args.checkpoint
    wandb = None
    if args.pretrained is not None:
        assert osp.exists(args.pretrained)
        weight_path = osp.realpath(args.pretrained).replace(
            'epoch_',
            osp.basename(args.config) + '_ep').replace('.pth', '-backbone.pth')
        os.system(f'MKL_THREADING_LAYER=GNU python '
                  f'tools/convert_weights/convert_to_pretrained.py '
                  f'{args.pretrained} {weight_path}')
        assert osp.exists(weight_path)
        cfg.model.backbone.pretrained = weight_path
        for h in cfg.log_config.hooks:
            if h.type == 'WandbLoggerHook' and not args.disable_wandb:
                import wandb
                init_kwargs = h.init_kwargs.to_dict()
                mmcv.mkdir_or_exist(f'wandb/{os.path.basename(weight_path)}')
                init_kwargs.update(
                    dict(
                        name=os.path.basename(weight_path),
                        resume=False,
                        dir=f'wandb/{os.path.basename(weight_path)}',
                        tags=[*h.init_kwargs.tags, 'siamfc'],
                        config=cfg.to_dict()))
                wandb.init(**init_kwargs)
                mmcv.mkdir_or_exist(
                    osp.join('./wandb',
                             osp.splitext(osp.basename(args.config))[0]))

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    logger.info(f'Config: {cfg.text}')

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    tracker = TrackerSiamFC(cfg, logger)
    if args.checkpoint is None and (cfg.model.backbone.frozen_stages < 4
                                    or cfg.extra_conv):
        train_seqs = GOT10k('data/GOT-10k', subset='train', return_meta=True)
        tracker.train_over(train_seqs)

    with torch.no_grad():
        tracker.net.eval()
        e = ExperimentOTB(
            'data/otb',
            version=2015,
            result_dir=osp.join(cfg.work_dir, cfg.suffix, 'results'),
            report_dir=osp.join(cfg.work_dir, cfg.suffix, 'reports'))
        e.run(tracker)
    performance = e.report([tracker.name])
    overall = performance[tracker.name]['overall']
    success_curve = overall.pop('success_curve')
    precision_curve = overall.pop('precision_curve')
    logger.info(overall)
    if wandb is not None:
        wandb.log(overall)

        data = [[
            x, y
        ] for (x,
               y) in zip(np.linspace(0, 1, len(success_curve)), success_curve)]
        table = wandb.Table(
            data=data, columns=['Overlap threshold', 'Success rate'])
        wandb.log({
            'Success':
            wandb.plot.line(
                table,
                'Overlap threshold',
                'Success rate',
                title='Success plots of OPE')
        })

        data = [[x, y] for (x, y) in zip(
            np.linspace(0, 1, len(precision_curve)), precision_curve)]
        table = wandb.Table(
            data=data, columns=['Location error threshold', 'Precision'])
        wandb.log({
            'Precision':
            wandb.plot.line(
                table,
                'Location error threshold',
                'Precision',
                title='Precision plots of OPE')
        })
        wandb.join()


if __name__ == '__main__':
    main()
