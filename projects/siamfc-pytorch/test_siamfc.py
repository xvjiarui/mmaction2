import argparse
import os.path as osp
import time

import mmcv
import numpy as np
import torch
from got10k.experiments import ExperimentOTB
from mmcv import Config, DictAction
from siamfc import TrackerSiamFC, default_cfg

from mmaction.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Test a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--auto-load',
        action='store_true',
        help='automatically resume training')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
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
    wandb = None
    for h in cfg.log_config.hooks:
        if h.type == 'WandbLoggerHook':
            import wandb
            wandb.init(**h.init_kwargs.to_dict())
            mmcv.mkdir_or_exist(
                osp.join('./wandb',
                         osp.splitext(osp.basename(args.config))[0]))
    cfg.pretrained = None
    if args.load_from is not None:
        cfg.load_from = args.load_from
    elif args.auto_load:
        if osp.exists(osp.join(cfg.work_dir, 'siamfc', 'latest.pth')):
            cfg.load_from = osp.join(cfg.work_dir, 'siamfc', 'latest.pth')

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    logger.info(f'Config: {cfg.text}')

    tracker = TrackerSiamFC(cfg, logger)

    e = ExperimentOTB(
        'data/otb',
        version=2015,
        result_dir=osp.join(cfg.work_dir, 'siamfc', 'results'),
        report_dir=osp.join(cfg.work_dir, 'siamfc', 'reports'))
    e.run(tracker)
    performance = e.report([tracker.name])
    overall = performance[tracker.name]['overall']
    success_curve = overall.pop('success_curve')
    precision_curve = overall.pop('precision_curve')
    overall.pop('speed_fps')
    logger.info(overall)
    if wandb is not None:
        wandb.log(overall)
        data = [[label, val] for label, val in overall.items()]
        table = wandb.Table(data=data, columns=['label', 'value'])
        wandb.log({
            'overvall':
            wandb.plot.bar(table, 'label', 'value', title='Overall')
        })

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
