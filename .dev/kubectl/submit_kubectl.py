import argparse
import os
import os.path as osp
import re
import tempfile

import mmcv


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
    parser.add_argument(
        '--gpus', type=int, default=2, help='number of gpus to use ')
    args, rest = parser.parse_known_args()

    return args, rest


def submit(config, args, rest):
    template_dict = dict(
        job_name=osp.splitext(osp.basename(config))[0].replace('_', '-') + '-',
        branch=args.branch,
        gpus=args.gpus,
        config=config,
        py_args=' '.join(rest),
        link='ln -s /exps/mmaction2/work_dirs; ' if args.ln_exp else '')
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
        for cfg in mmcv.scandir(args.config, suffix='.py'):
            submit(osp.join(args.config, cfg), args, rest)
    else:
        submit(args.config, args, rest)


if __name__ == '__main__':
    main()
