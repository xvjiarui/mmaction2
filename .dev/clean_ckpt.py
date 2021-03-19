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
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--clean-aux', action='store_true')

    args = parser.parse_args()

    return args


def clean_dir(input_dir, interval, min_len, dry_run, clean_aux):
    epoch_nums = []
    for file in os.listdir(input_dir):
        if file.startswith('epoch_') and file.endswith('.pth'):
            ep = int(osp.splitext(file)[0].split('_')[-1])
            epoch_nums.append(ep)
        if clean_aux and 'backbone' in file:
            print(f'cleaning {osp.join(input_dir, file)}')
            if not dry_run:
                os.remove(osp.join(input_dir, file))

    epoch_nums.sort()
    if len(epoch_nums) <= min_len:
        print('too less ckpts')
        return
    count = 0
    for i, ep in enumerate(epoch_nums[0:-1]):
        clean_path = osp.join(input_dir, f'epoch_{ep}.pth')
        if ep % interval != 0:
            print(f'cleaning {clean_path}')
            if not dry_run:
                os.remove(clean_path)
            count += 1
        else:
            print(f'skipping {clean_path}')
    print(f'cleaned {count} ckpts')


def main():
    args = parse_args()
    if args.work_dir is not None:
        clean_path = osp.join(args.root, args.work_dir)
        clean_dir(clean_path, args.interval, args.min_len, args.dry_run,
                  args.clean_aux)
    else:
        for directory in os.listdir(args.root):
            if osp.isdir(osp.join(args.root, directory)):
                if args.match is None or args.match in directory:
                    clean_path = osp.join(args.root, directory)
                    clean_dir(clean_path, args.interval, args.min_len,
                              args.dry_run, args.clean_aux)
                else:
                    print(f'skipping {directory}')


if __name__ == '__main__':
    main()
