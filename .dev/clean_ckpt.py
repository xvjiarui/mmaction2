import argparse
import os
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('root', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--min-len', type=int, default=10)
    parser.add_argument('--match', type=str)

    args = parser.parse_args()

    return args


def clean_dir(input_dir, interval, min_len):
    epoch_nums = []
    for file in os.listdir(input_dir):
        if file.startswith('epoch_') and file.endswith('.pth'):
            ep = int(osp.splitext(file)[0].split('_')[-1])
            epoch_nums.append(ep)
    epoch_nums.sort()
    if len(epoch_nums) <= min_len:
        print('too less ckpts')
        return
    count = 0
    for i, ep in enumerate(epoch_nums[0:-1]):
        clean_path = osp.join(input_dir, f'epoch_{ep}.pth')
        if ep % interval != 0:
            print(f'cleaning {clean_path}')
            # os.remove(clean_path)
            count += 1
        else:
            print(f'skipping {clean_path}')
    print(f'cleaned {count} ckpts')


def main():
    args = parse_args()
    if args.work_dir is not None:
        clean_path = osp.join(args.root, args.work_dir)
        clean_dir(clean_path, args.interval, args.min_len)
    else:
        for directory in os.listdir(args.root):
            if osp.isdir(directory):
                if args.match is not None and args.match in directory:
                    print(f'skipping {directory}')
                    continue
                clean_path = osp.join(args.root, directory)
                clean_dir(clean_path, args.interval, args.min_len)


if __name__ == '__main__':
    main()
