import argparse
import os
import os.path as osp
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmaction.apis import multi_gpu_test, single_gpu_test
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='automatically resume training')
    parser.add_argument(
        '--out', default=None, help='output result file in pickle format')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        default='tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom evaluation options')
    parser.add_argument(
        '--start-epoch', type=int, default=1, help='-1 for latest.pth')
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--skip', action='store_true')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--suffix', type=str, help='work_dir suffix')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

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
    if not osp.exists(cfg.work_dir):
        print('No work dir found, exiting')
        return
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'test_seq_davis-{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    # Overwrite output_config from args.out
    output_config = merge_configs(output_config, dict(out=args.out))

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    # Overwrite eval_config from args.eval
    eval_config = merge_configs(eval_config, dict(metrics=args.eval))
    # Add options from args.option
    eval_config = merge_configs(eval_config, args.eval_options)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()

    wandb = None
    for h in cfg.log_config.hooks:
        if rank == 0 and h.type == 'WandbLoggerHook':
            import wandb
            init_kwargs = h.init_kwargs.to_dict()
            suffix = 'davis'
            if args.suffix is not None:
                suffix = f'{args.suffix}-{suffix}'
            init_kwargs.update(
                dict(
                    name=h.init_kwargs.name + '-davis',
                    tags=[*h.init_kwargs.tags, 'davis'],
                    resume=args.auto_resume,
                    dir=f'wandb/{h.init_kwargs.name}-{suffix}'))
            mmcv.mkdir_or_exist(f'wandb/{h.init_kwargs.name}-{suffix}')
            wandb.init(**init_kwargs)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    train_len = len(build_dataset(cfg.data.train))
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    json_path = osp.join(cfg.work_dir, 'test_seq_davis.json')
    if osp.exists(json_path) and args.auto_resume:
        eval_info = mmcv.load(json_path)
        start_epoch = eval_info['last_epoch'] + 1
    else:
        eval_info = dict()
        start_epoch = args.start_epoch

    # build the model and load checkpoint
    train_iters = train_len // 256
    for epoch in range(start_epoch, cfg.total_epochs + 1,
                       cfg.checkpoint_config.interval):
        if start_epoch == -1:
            args.auto_resume = False
            ckpt_path = osp.realpath(osp.join(cfg.work_dir, 'latest.pth'))
            if not osp.exists(ckpt_path):
                print('latest.pth not found, exiting')
                return
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

        logger.info(f'test_cfg: {cfg.test_cfg}')
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, ckpt_path, map_location='cpu')
        if distributed:
            dist.barrier()

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)

        if rank == 0:
            if output_config:
                out = output_config['out']
                logger.info(f'\nwriting results to {out}')
                dataset.dump_results(outputs, **output_config)
            if eval_config:
                eval_res = dataset.evaluate(
                    outputs, **eval_config, logger=logger)
                for name, val in eval_res.items():
                    logger.info(f'{name}: {val:.04f}')
                logger.info(f'checkpoint: {ckpt_path}')
                if wandb is not None:
                    wandb.log(eval_res, step=epoch * train_iters, commit=False)
            if args.auto_resume:
                eval_info['last_epoch'] = epoch
                mmcv.dump(eval_info, json_path)
        if distributed:
            dist.barrier()
        if start_epoch == -1:
            break


if __name__ == '__main__':
    main()
