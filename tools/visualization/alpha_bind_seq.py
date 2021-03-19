import argparse
import os.path as osp

import mmcv
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('img_dir', help='img config directory')
    parser.add_argument('gt_dir', help='gt config directory')
    parser.add_argument('out_dir', help='output config directory')
    parser.add_argument(
        '--gif', action='store_true', help='save the video into gif')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_suffix = '.jpg'
    seg_map_suffix = '.png'
    mmcv.mkdir_or_exist(args.out_dir)
    for video_name in mmcv.scandir(args.img_dir):
        video_binded = []
        for img_name in mmcv.scandir(
                osp.join(args.img_dir, video_name), suffix=img_suffix):
            img_path = osp.join(args.img_dir, video_name, img_name)
            seg_name = img_name.replace(img_suffix, seg_map_suffix)
            seg_path = osp.join(args.gt_dir, video_name, seg_name)
            out_path = osp.join(args.out_dir, video_name, img_name)
            if not osp.exists(seg_path):
                continue
            img = mmcv.imread(img_path)
            seg = mmcv.imread(seg_path)
            binded = img * 0.5 + seg * 0.5
            video_binded.append(binded)
            mmcv.imwrite(binded, out_path)
        mmcv.mkdir_or_exist(osp.join(args.out_dir, 'gifs'))
        gif = Image.fromarray(video_binded[0])
        gif.save(
            fp=osp.join(args.out_dir, f'gifs/{video_name}.gif'),
            format='GIF',
            append_images=[Image.fromarray(_) for _ in video_binded[1:]],
            save_all=True,
            duration=200,
            loop=0)


if __name__ == '__main__':
    main()
