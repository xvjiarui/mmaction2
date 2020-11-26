import os.path as osp
import os
import argparse
from shutil import copyfile

def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src imagenet path')
    parser.add_argument('dst', help='save path')
    parser.add_argument('subset_list', help='list of subset')
    args = parser.parse_args()

    with open(args.subset_list) as f:
        folder_list = [l.strip('\n') for l in f.readlines()]


    train_num = 0
    val_num = 0
    for folder in folder_list:
        num_imgs = len(os.listdir(osp.join(args.src, 'train', folder)))
        train_num += num_imgs
        num_imgs = len(os.listdir(osp.join(args.src, 'val', folder)))
        val_num += num_imgs
        os.makedirs(osp.join(args.dst, 'train'), exist_ok=True)
        os.makedirs(osp.join(args.dst, 'val'), exist_ok=True)
        os.symlink(osp.realpath(osp.join(args.src, 'train', folder)), osp.join(args.dst, 'train', folder), target_is_directory=True)
        os.symlink(osp.realpath(osp.join(args.src, 'val', folder)), osp.join(args.dst, 'val', folder), target_is_directory=True)
    for file in os.listdir(args.src):
        if file.endswith('.txt'):
            copyfile(osp.join(args.src, file), osp.join(args.dst, file))
    print(f'train: {train_num}, val: {val_num}')


if __name__ == '__main__':
    main()
