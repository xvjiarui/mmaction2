import os
import time

import mmcv


def test(model,
         L=20,
         K=10,
         T=0.07,
         R=12,
         opts=[],
         gpu=0,
         force=False,
         dryrun=False,
         data_path=None,
         data_list_path=None,
         config=None,
         disable_wandb=False):
    R = int(R)

    assert os.path.exists(model)
    model_type = 'scratch'

    model_name = '_'.join(model.split('/')[1:])  #.replace('/', '_')

    outdir = 'outdir'
    davis2017path = '../davis2017-evaluation/'

    model_name = '%s_L%s_K%s_T%s_R%s_opts%s_M%s' % \
                    (str(int(time.time()))[-4:], L, K, T, R, ''.join(opts), model_name)
    time.sleep(1)

    opts = ' '.join(opts)
    cmd = ''

    weight_path = os.path.realpath(model).replace(
        'epoch_',
        os.path.basename(model) + '_ep').replace('.pth', '-crw.pth')
    if not os.path.exists(weight_path):
        os.system(
            f'MKL_THREADING_LAYER=GNU python projects/videowalk/convert_to_crw.py {model} {weight_path}'
        )
    cfg = mmcv.Config.fromfile(config)
    for h in cfg.log_config.hooks:
        if h.type == 'WandbLoggerHook' and not disable_wandb:
            import wandb
            init_kwargs = h.init_kwargs.to_dict()
            mmcv.mkdir_or_exist(f'wandb/{os.path.basename(weight_path)}')
            init_kwargs.update(
                dict(
                    name=os.path.basename(weight_path),
                    resume=False,
                    dir=f'wandb/{os.path.basename(weight_path)}',
                    tags=[*h.init_kwargs.tags, 'crw'],
                    config=cfg.to_dict()))
            wandb.init(**init_kwargs)
    assert os.path.exists(weight_path)

    model_str = f'--model-type {model_type} --resume {weight_path}'

    cmd += f' python projects/videowalk/test.py --filelist {data_list_path} {model_str} \
            --topk {K} --radius {R}  --videoLen {L} --temperature {T} --save-path {outdir}/results_{model_name} \
            --workers 5  {opts} --gpu-id {gpu} && '

    convert_str = f'python projects/videowalk/eval/convert_davis.py --in_folder {outdir}/results_{model_name}/ \
            --out_folder {outdir}/converted_{model_name}/ --dataset {data_path}'

    eval_str = f'python {davis2017path}/evaluation_method.py --task semi-supervised \
            --results_path  {outdir}/converted_{model_name}/ --set val --davis_path {data_path}'

    cmd += f' {convert_str} && {eval_str}'
    print(cmd)

    if not dryrun:
        os.system(cmd)


def run(models,
        L,
        K,
        T,
        R,
        size,
        finetune,
        force=False,
        gpu=-1,
        dryrun=False,
        data_path=None,
        data_list_path=None,
        config=None,
        disable_wandb=False):
    import itertools

    base_opts = [
        '--cropSize',
        str(size),
    ]

    if finetune > 0:
        base_opts += [
            '--head-depth',
            str(0), '--use-res4', '--finetune',
            str(finetune)
        ]
    else:
        base_opts += ['--head-depth', str(-1)]

    opts = [base_opts]
    prod = list(itertools.product(models, L, K, T, R, opts))

    print(prod)
    for i in range(0, len(prod)):
        test(
            *prod[i],
            0 if gpu == -1 else gpu,
            force,
            dryrun=dryrun,
            data_path=data_path,
            data_list_path=data_list_path,
            config=config,
            disable_wandb=disable_wandb)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-path',
        default=[],
        type=str,
        nargs='+',
        help='(list of) paths of models to evaluate')
    parser.add_argument('--data-path', type=str, default='/data/davis/DAVIS/')
    parser.add_argument(
        '--data-list-path',
        type=str,
        default='projects/videowalk/eval/davis_vallist.txt')

    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--dryrun', default=False, action='store_true')

    parser.add_argument('--L', default=[20], type=int, nargs='+')
    parser.add_argument('--K', default=[10], type=int, nargs='+')
    parser.add_argument('--T', default=[0.05], type=float, nargs='+')
    parser.add_argument('--R', default=[12], type=float, nargs='+')
    parser.add_argument('--cropSize', default=-1, type=int)

    parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--config', type=str)
    parser.add_argument(
        '--disable-wandb', action='store_true', help='disable wandb')

    args = parser.parse_args()

    run(args.model_path,
        args.L,
        args.K,
        args.T,
        args.R,
        args.cropSize,
        args.finetune,
        force=args.force,
        gpu=args.gpu,
        dryrun=args.dryrun,
        data_path=args.data_path,
        data_list_path=args.data_list_path,
        config=args.config,
        disable_wandb=args.disable_wandb)
