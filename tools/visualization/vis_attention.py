import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import ConnectionPatch
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         set_random_seed)
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmcv.utils import DictAction
from torch.nn.modules.utils import _pair

from mmaction.apis import multi_gpu_test, single_gpu_test
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.models.common import cat, spatial_neighbor, video2images
from mmaction.models.plugin import SelfAttention, SelfAttentionBlock
from mmaction.models.trackers import VanillaTracker


def get_top_diff_loc(imgs, ref_imgs, crop_size, grid_size, device, topk=20):
    """Randomly get a crop bounding box."""
    assert imgs.shape == ref_imgs.shape
    batches = imgs.size(0)
    img_size = imgs.shape[2:]
    crop_size = _pair(crop_size)
    grid_size = _pair(grid_size)
    stride_h = (img_size[0] - crop_size[0]) // (grid_size[0] - 1)
    stride_w = (img_size[1] - crop_size[1]) // (grid_size[1] - 1)
    diff_imgs = imgs - ref_imgs

    diff_list = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            crop_diff = diff_imgs[:, :,
                                  i * stride_h:i * stride_h + crop_size[0],
                                  j * stride_w:j * stride_w + crop_size[1]]
            diff_list.append(crop_diff.abs().sum(dim=(1, 2, 3)))
    # [batches, grid_size**2]
    diff_sum = torch.stack(diff_list, dim=1)
    diff_topk_idx = torch.argsort(diff_sum, dim=1, descending=True)[:, :topk]
    select_idx = diff_topk_idx
    idx_i = select_idx // grid_size[1]
    idx_j = select_idx % grid_size[1]

    crop_y1, crop_y2 = idx_i * stride_h, idx_i * stride_h + crop_size[0]
    crop_x1, crop_x2 = idx_j * stride_w, idx_j * stride_w + crop_size[1]
    center = torch.stack([(crop_x1 + crop_x2) * 0.5,
                          (crop_y1 + crop_y2) * 0.5],
                         dim=-1).float()

    return center


def get_att_map(self, query, key=None):
    q_h, q_w = query.shape[-2:]
    k_h, k_w = key.shape[-2:]
    if isinstance(self, SelfAttentionBlock):
        query = query.flatten(2)
        if key is None:
            key = query
        key = key.flatten(2)
        key = self.downsample_input(key)
        query = self.query_project(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key)
        key = key.reshape(*key.shape[:2], -1)

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
    elif isinstance(self, SelfAttention):
        if key is None:
            key = query
        key = self.downsample_input(key)
        query = query.flatten(2)
        key = key.flatten(2)
        if self.normalize:
            query = F.normalize(query, p=2, dim=1)
            key = F.normalize(key, p=2, dim=1)
        # [B, HxW, HxW]
        affinity = torch.einsum('bci, bcj->bij', key, query).contiguous()
        if self.matmul_norm:
            affinity = (query.shape[1]**-.5) * affinity
        affinity = affinity.softmax(dim=1)
        sim_map = affinity.transpose(1, 2)
    elif self is None:
        if key is None:
            key = query
        query = query.flatten(2)
        key = key.flatten(2)
        # [B, HxW, HxW]
        affinity = torch.einsum('bci, bcj->bij', query, key).contiguous()
        affinity = (query.shape[1]**-.5) * affinity
        sim_map = affinity.softmax(dim=-1)
    else:
        raise ValueError

    sim_map = sim_map.reshape(-1, q_h, q_w, k_h, k_w)
    return sim_map


class Visualizer(nn.Module):

    def __init__(self,
                 tracker,
                 query_idx,
                 key_indices,
                 feat_idx=0,
                 out_dir='./vis_att'):
        super().__init__()
        assert isinstance(tracker, VanillaTracker)
        self.tracker = tracker
        self.feat_idx = feat_idx
        self.query_idx = query_idx
        self.key_indices = key_indices
        self.out_dir = out_dir
        self.save_index = 0

    def forward(self,
                imgs,
                ref_seg_map=None,
                img_meta=None,
                label=None,
                return_loss=False):
        if hasattr(self.tracker.backbone, 'switch_strides'):
            self.tracker.backbone.switch_strides()

        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        dummy_faet = self.tracker.extract_feat_test(imgs[0:1, :, 0])
        if isinstance(dummy_faet, (list, tuple)):
            feat_shapes = [_.shape for _ in dummy_faet]
        else:
            feat_shapes = [dummy_faet.shape]
        feat_bank = self.tracker.get_feats(imgs, len(dummy_faet))

        feat_shape = feat_shapes[0]

        query_feat = feat_bank[self.feat_idx][:, :,
                                              self.query_idx].to(imgs.device)
        key_feat = feat_bank[self.feat_idx][:, :,
                                            self.key_indices].to(imgs.device)
        assert feat_shape[0] == 1

        # [N, H, W, H, W]
        sim_map = get_att_map(None, query_feat, key_feat)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.margins(0, 0)
        fig = self.show(imgs, sim_map)
        mmcv.mkdir_or_exist(self.out_dir)
        plt.savefig(
            osp.join(self.out_dir, f'{self.save_index:05d}.png'),
            bbox_inches='tight')
        self.save_index += 1
        # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        if self.save_index == 50:
            import ipdb
            ipdb.set_trace()
        return [0]

    def show(self, imgs, att_map):
        imgs = video2images(imgs).detach().cpu()
        att_map = att_map.detach().cpu().numpy()
        h, w = att_map.shape[1:3]
        ratio = h / imgs.size(2)
        diff_loc = get_top_diff_loc(
            imgs[0:1], imgs[1:2], 9, 28, imgs.device,
            topk=400).view(-1, 2).numpy()
        diff_loc = diff_loc[::100]
        mean = torch.tensor([123.675, 116.28,
                             103.53]).to(imgs).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).to(imgs).view(1, 3, 1, 1)
        imgs = (imgs * std + mean) / 255
        fig = plt.figure(figsize=(20, 5))
        assert len(imgs) == 2
        all_indices = [0] + [1] * len(diff_loc)
        axes = []
        for i in range(len(all_indices)):
            ax = fig.add_subplot(1, len(all_indices), i + 1)
            ax.axis('off')
            ax.imshow(imgs[all_indices[i]].permute(1, 2,
                                                   0).clamp(min=0.,
                                                            max=1.).numpy())
            axes.append(ax)
        for i, loc in enumerate(diff_loc):
            x = loc[0]
            y = loc[1]
            # axes[0].plot([x], [y], marker='s', color='r')
            axes[0].text(x, y, f'{i+1}', bbox=dict(facecolor='red', alpha=0.5))
            down_x = int(x * ratio)
            down_y = int(y * ratio)
            satt_map = att_map[0, down_y, down_x].copy()
            satt_map[satt_map < (np.percentile(satt_map, 80))] = satt_map.min()
            satt_map = mmcv.imresize(satt_map, imgs.shape[2:])
            axes[i + 1].imshow(
                satt_map, cmap='jet', interpolation='bilinear', alpha=0.3)
        return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
    parser.add_argument(
        '--eval-pretrained',
        action='store_true',
        help='whether to eval pretrained model instead of '
        'loaded one')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
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

    if args.seed is not None:
        print('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    dataset_type = 'VideoDataset'
    data_prefix = 'data/kinetics400/videos_train'
    ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False)
    train_pipeline = [
        dict(type='DecordInit'),
        dict(
            type='SampleFrames',
            clip_len=2,
            frame_interval=8,
            num_clips=1,
            test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, 224), keep_ratio=True),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]
    train_data_cfg = mmcv.ConfigDict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_prefix,
        pipeline=train_pipeline)
    # build the dataloader
    dataset = build_dataset(train_data_cfg, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=1,
        workers_per_gpu=0,
        dist=distributed,
        shuffle=True,
        seed=args.seed)

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if not args.eval_pretrained:
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model = Visualizer(
        model, query_idx=0, key_indices=[1], out_dir=args.out, feat_idx=0)

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


if __name__ == '__main__':
    main()
