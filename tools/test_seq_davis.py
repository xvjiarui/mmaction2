import argparse
import os
import os.path as osp

import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmcv.utils import DictAction

from mmaction.apis import multi_gpu_test, single_gpu_test
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--out', default=None, help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
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
        '--average-clips',
        choices=['score', 'prob'],
        default='score',
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
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
    wandb = None
    for h in cfg.log_config.hooks:
        if h.type == 'WandbLoggerHook':
            import wandb
            wandb.init(**h.init_kwargs.to_dict())
            mmcv.mkdir_or_exist(
                osp.join('./wandb',
                         osp.splitext(osp.basename(args.config))[0]))

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

    if cfg.test_cfg is None:
        cfg.test_cfg = dict(average_clips=args.average_clips)
    else:
        cfg.test_cfg.average_clips = args.average_clips

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    for epoch in range(cfg.total_epochs):
        ckpt_path = osp.join(cfg.work_dir, f'epoch_{epoch+1}.pth')
        if not osp.exists(ckpt_path):
            print(f'skping {epoch+1}')
        load_checkpoint(model, ckpt_path, map_location='cpu')

        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)

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

        rank, _ = get_dist_info()
        if rank == 0:
            if output_config:
                out = output_config['out']
                print(f'\nwriting results to {out}')
                dataset.dump_results(outputs, **output_config)
            if eval_config:
                eval_res = dataset.evaluate(outputs, **eval_config)
                for name, val in eval_res.items():
                    print(f'{name}: {val:.04f}')
                if wandb is not None:
                    wandb.log(eval_res, step=epoch + 1)
        if distributed:
            dist.barrier()


if __name__ == '__main__':
    main()
