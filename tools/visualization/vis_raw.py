import argparse
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import set_random_seed
from torchvision.utils import save_image

from mmaction.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.show or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--show" or "'
         '--show-dir"')

    # set random seeds
    if args.seed is not None:
        print('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed, deterministic=False)

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=4,
        workers_per_gpu=0,
        dist=distributed,
        shuffle=True)
    single_gpu_vis(data_loader, args.show, args.show_dir)


def single_gpu_vis(data_loader, show=False, out_dir=None):

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))

        if show or out_dir:
            img_tensor = data['imgs']
            img_norm_cfg = dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=False)

            img_tensor = img_tensor.reshape((-1, ) + img_tensor.shape[2:])

            imgs = []
            for _ in range(img_tensor.size(2)):
                img = mmcv.tensor2imgs(img_tensor[:, :, _], **img_norm_cfg)
                imgs.append(img)

            imgs_show = []
            for batch_idx in range(batch_size):
                for idx in range(len(imgs)):
                    imgs_show.append(
                        torch.from_numpy(imgs[idx][batch_idx].copy()))
            mmcv.mkdir_or_exist(out_dir)
            save_image(
                torch.stack(imgs_show).permute(0, 3, 1, 2) / 255.,
                osp.join(out_dir, f'{i:05d}_raw.png'),
                nrow=len(imgs))

        for _ in range(batch_size):
            prog_bar.update()


if __name__ == '__main__':
    main()
