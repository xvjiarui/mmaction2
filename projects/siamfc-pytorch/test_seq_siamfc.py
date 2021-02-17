import argparse
import builtins
import os
import os.path as osp
import time

import mmcv
import numpy as np
import torch
from got10k.experiments import ExperimentOTB
from mmcv import Config, DictAction
from mmcv.runner import set_random_seed
from siamfc import TrackerSiamFC, default_cfg

from mmaction.utils import get_root_logger
from mmaction.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)',
        default=1)
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
    parser.add_argument(
        '--start-epoch', type=int, default=1, help='-1 for latest.pth')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='automatically resume training')
    parser.add_argument('--out-index', type=int, default=3)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--skip', action='store_true')
    args = parser.parse_args()

    return args

def convert_to_pretrained(pretrained, config):
    assert osp.exists(pretrained)
    weight_path = osp.realpath(pretrained).replace(
        'epoch_',
        osp.basename(config) + '_ep').replace('.pth', '-backbone.pth')
    os.system(f'MKL_THREADING_LAYER=GNU python '
              f'tools/convert_weights/convert_to_pretrained.py '
              f'{pretrained} {weight_path}')
    assert osp.exists(weight_path)

    return weight_path


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(default_cfg)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.gpus = args.gpus

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
    for h in cfg.log_config.hooks:
        if h.type == 'WandbLoggerHook' and not args.disable_wandb:
            import wandb
            init_kwargs = h.init_kwargs.to_dict()
            init_kwargs.update(
                dict(
                    name=h.init_kwargs.name + '-siamfc',
                    resume=args.auto_resume,
                    dir=f'wandb/{h.init_kwargs.name}-{args.suffix}',
                    tags=[*h.init_kwargs.tags, 'siamfc'],
                    config=cfg.to_dict()))
            mmcv.mkdir_or_exist(f'wandb/{h.init_kwargs.name}-{args.suffix}')
            wandb.init(**init_kwargs)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'test_seq_sf-{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    logger.info(f'Config: {cfg.text}')

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    def print_log(*args):
        logger.info(','.join(args))

    builtins.print = print_log
    json_path = osp.join(cfg.work_dir, 'test_seq_vip.json')
    if osp.exists(json_path) and args.auto_resume:
        eval_info = mmcv.load(json_path)
        start_epoch = eval_info['last_epoch'] + 1
    else:
        eval_info = dict()
        start_epoch = args.start_epoch

    # build the model and load checkpoint
    train_len = len(build_dataset(cfg.data.train))
    train_iters = train_len // 256
    for epoch in range(start_epoch, cfg.total_epochs + 1,
                       cfg.checkpoint_config.interval):
        if start_epoch == -1:
            args.auto_resume = False
            ckpt_path = osp.realpath(osp.join(cfg.work_dir, 'latest.pth'))
            epoch = int(
                osp.splitext(osp.basename(ckpt_path))[0].split('_')[-1])
        else:
            if epoch % args.eval_interval != 0:
                continue
            ckpt_path = osp.join(cfg.work_dir, f'epoch_{epoch}.pth')
            if not osp.exists(ckpt_path):
                if args.skip:
                    logger.info(f'{ckpt_path} not exist, skipping')
                    continue
                else:
                    logger.info(f'{ckpt_path} not exist, waiting')
            while not osp.exists(ckpt_path):
                time.sleep(300)
        logger.info(f'Found {ckpt_path}')
        weight_path = convert_to_pretrained(ckpt_path, args.config)

        cfg.model.backbone.pretrained = weight_path
        cfg.model.backbone.out_indices = (args.out_index,)
        logger.info(f'cfg: \n {cfg.pretty_text}')
        tracker = TrackerSiamFC(cfg, logger)

        with torch.no_grad():
            tracker.net.eval()
            e = ExperimentOTB(
                'data/otb',
                version=2015,
                result_dir=osp.join(cfg.work_dir, cfg.suffix, f'ep{epoch}', 'results'),
                report_dir=osp.join(cfg.work_dir, cfg.suffix, f'ep{epoch}', 'reports'))
            e.run(tracker)
        performance = e.report([tracker.name])
        overall = performance[tracker.name]['overall']
        success_score = overall['success_score'] * 100
        success_score = np.round(success_score, 2)
        precision_score = overall['precision_score'] * 100
        precision_score = np.round(precision_score, 2)
        success_rate = overall['success_rate'] * 100
        success_rate = np.round(success_rate, 2)
        speed_fps = overall['speed_fps']
        speed_fps = np.round(speed_fps, 2)
        logger.info(f'copypaste: {precision_score},{success_score}')
        logger.info(f'success_score: {success_score}')
        logger.info(f'precision_score: {precision_score}')
        logger.info(f'success_rate: {success_rate}')
        logger.info(f'speed_fps: {speed_fps}')
        eval_res = dict(
            success_score=success_score,
            precision_score=precision_score,
            success_rate=success_rate)
        if wandb is not None:
            wandb.log(eval_res, step=epoch * train_iters, commit=False)
        if args.auto_resume:
            eval_info['last_epoch'] = epoch
            mmcv.dump(eval_info, json_path)
        if start_epoch == -1:
            break


if __name__ == '__main__':
    main()
