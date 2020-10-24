import argparse
import os
import os.path as osp
import re
import tempfile

WANDB_KEY = '18a953cf069a567c46b1e613f940e6eb8f878c3d'

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Submit to nautilus via kubectl')
    parser.add_argument('job', help='kubectl config path')
    parser.add_argument(
        '--branch', '-b', type=str, default='dev', help='git clone branch')
    parser.add_argument(
        '--ln-exp',
        '-l',
        action='store_true',
        help='link experiment directory')
    parser.add_argument(
        '-a',
        '--arch',
        metavar='ARCH',
        default='resnet18',
        choices=model_names,
        help='model architecture: ' + ' | '.join(model_names) +
        ' (default: resnet18)')
    parser.add_argument(
        '--batch-size',
        default=256,
        type=int,
        metavar='N',
        help='mini-batch size (default: 256), this is the total '
        'batch size of all GPUs on the current node when '
        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--version', '-v', default='v1', choices=['v1', 'v2'])
    parser.add_argument('--wandb', '-w', action='store_true', help='use wandb')
    parser.add_argument(
        '--gpus', type=int, default=2, help='number of gpus to use ')
    parser.add_argument(
        '--cpus', type=int, default=4, help='number of cpus to use')
    parser.add_argument(
        '--name-space', '-n', type=str, default='self-supervised-video')
    args, rest = parser.parse_known_args()

    return args, rest


def submit(args, rest):
    imgs_per_batch = args.batch_size * args.gpus
    multiplier = 256 * 8 // imgs_per_batch
    py_args = f'-a {args.arch} --lr {0.03 / multiplier} ' \
              f'--batch-size {args.batch_size} '
    if args.version == 'v2':
        py_args += '--mlp --moco-t 0.2 --aug-plus --cos '
    if args.wandb:
        py_args += '--wandb '
    py_args += ' '.join(rest)
    template_dict = dict(
        job_name=f'vanilla_moco{args.version}_'
        f'{args.arch}_{args.batch_size}x{args.gpus}',
        name_space=args.name_space,
        branch=args.branch,
        gpus=args.gpus,
        cpus=args.cpus,
        py_args=py_args,
        link='ln -s /exps/mmaction2/work_dirs; ' if args.ln_exp else '',
        wandb='mkdir -p /exps/mmaction2/wandb; '
        'ln -s /exps/mmaction2/wandb; '
        'pip install --upgrade wandb && wandb login '
        f'{WANDB_KEY} ;' if args.wandb else '')
    with open(args.job, 'r') as f:
        config_file = f.read()
    for key, value in template_dict.items():
        regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
        config_file = re.sub(regexp, str(value), config_file)
    temp_config_file = tempfile.NamedTemporaryFile(
        suffix=osp.splitext(args.job)[1])
    with open(temp_config_file.name, 'w') as tmp_config_file:
        tmp_config_file.write(config_file)
    # pprint.pprint(mmcv.load(temp_config_file.name))
    os.system(f'kubectl create -f {temp_config_file.name}')
    tmp_config_file.close()


def main():
    args, rest = parse_args()
    submit(args, rest)


if __name__ == '__main__':
    main()
