import argparse
import os.path as osp

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('img_dir', help='img config directory')
    parser.add_argument('gt_dir', help='gt config directory')
    parser.add_argument('out_dir', help='output config directory')
    args = parser.parse_args()
    return args


# def main():
#     args = parse_args()
#     img_suffix = '_leftImg8bit.png'
#     seg_map_suffix = '_gtFine_color.png'
#     mmcv.mkdir_or_exist(args.out_dir)
#     for img_file in mmcv.scandir(args.img_dir, suffix=img_suffix):
#         seg_file = img_file.replace(img_suffix, seg_map_suffix)
#         img = mmcv.imread(osp.join(args.img_dir, img_file))
#         seg = mmcv.imread(osp.join(args.gt_dir, seg_file))
#         binded = img * 0.5 + seg * 0.5
#         mmcv.imwrite(binded, osp.join(args.out_dir, img_file))


def main():
    args = parse_args()
    img_suffix = '.jpg'
    seg_map_suffix = '.png'
    mmcv.mkdir_or_exist(args.out_dir)
    for img_file in mmcv.scandir(
            args.img_dir, suffix=img_suffix, recursive=True):
        seg_file = img_file.replace(img_suffix, seg_map_suffix)
        if not osp.exists(osp.join(args.gt_dir, seg_file)):
            continue
        img = mmcv.imread(osp.join(args.img_dir, img_file))
        seg = mmcv.imread(osp.join(args.gt_dir, seg_file))
        binded = img * 0.5 + seg * 0.5
        mmcv.imwrite(binded, osp.join(args.out_dir, img_file))


if __name__ == '__main__':
    main()
