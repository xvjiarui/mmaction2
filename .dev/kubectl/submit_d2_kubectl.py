import argparse
import os
import os.path as osp
import re
import tempfile

import mmcv

WANDB_KEY = '18a953cf069a567c46b1e613f940e6eb8f878c3d'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Submit to nautilus via kubectl')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('job', help='kubectl config path')
    parser.add_argument(
        '--branch', '-b', type=str, default='dev', help='git clone branch')
    parser.add_argument(
        '--ln-exp',
        '-l',
        action='store_true',
        help='link experiment directory')
    parser.add_argument('--wandb', '-w', action='store_true', help='use wandb')
    parser.add_argument(
        '--gpus', type=int, default=2, help='number of gpus to use ')
    parser.add_argument(
        '--cpus', type=int, default=4, help='number of cpus to use')
    parser.add_argument(
        '--mem', type=int, default=30, help='amount of memory to use')
    parser.add_argument('--file', '-f', type=str, help='config txt file')
    parser.add_argument(
        '--name-space',
        '-n',
        type=str,
        default='self-supervised-video',
        choices=['self-supervised-video', 'ece3d-vision', 'image-model'])
    parser.add_argument('--epoch', type=str, default='latest.pth')
    parser.add_argument(
        '--reference-world-size',
        '-r',
        type=int,
        default=0,
        help='Detectron2 REFERENCE_WORLD_SIZE with DATALOADER.NUM_WORKERS',
    )
    args, rest = parser.parse_known_args()

    return args, rest


def submit(config, args, rest):
    work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config))[0])
    pretrained_ckpt = osp.join(work_dir, args.epoch)
    py_args = ' '.join(rest)
    py_args += f' OUTPUT_DIR work_dirs/{osp.splitext(osp.basename(config))[0]}/d2_voc '
    py_args += 'MODEL.RESNETS.DEPTH 18 ' \
                'MODEL.RESNETS.RES2_OUT_CHANNELS 64' if 'r18' in config else ''
    if args.reference_world_size != 0:
        py_args += f' SOLVER.REFERENCE_WORLD_SIZE {args.reference_world_size} '
    template_dict = dict(
        job_name=osp.splitext(osp.basename(config))[0].lower().replace(
            '_', '-') + '-',
        base_config=osp.splitext(osp.basename(config))[0],
        name_space=args.name_space,
        branch=args.branch,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=f'{args.mem}Gi',
        max_cpus=int(args.cpus * 1.2),
        max_mem=f'{int(args.mem * 1.2)}Gi',
        config=config,
        pretrained_ckpt=pretrained_ckpt,
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
    if osp.isdir(args.config):
        if args.file is not None:
            with open(args.file) as f:
                submit_cfg_names = [line.strip() for line in f.readlines()]
            for cfg in mmcv.scandir(args.config, recursive=True):
                if osp.basename(cfg) in submit_cfg_names:
                    submit(osp.join(args.config, cfg), args, rest)
        else:
            for cfg in mmcv.scandir(args.config, suffix='.py'):
                if 'playground' in cfg:
                    continue
                submit(osp.join(args.config, cfg), args, rest)
    else:
        submit(args.config, args, rest)


if __name__ == '__main__':
    main()
