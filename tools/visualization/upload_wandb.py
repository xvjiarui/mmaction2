import argparse
import json
import os.path as osp

import mmcv
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Upload Json Log to wandb')
    parser.add_argument('log_dir', help='train log file path')
    args = parser.parse_args()
    return args


def load_json_log(json_log):
    metrics_list = []
    with open(json_log, 'r') as log_file:
        last_epoch = 0
        epoch_iter = 0
        last_iter = 0
        for line in log_file:
            metrics = {}
            log = json.loads(line.strip())
            # skip lines without `epoch` field
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            _ = log.pop('memory')
            _ = log.pop('data_time')
            _ = log.pop('time')
            mode = log.pop('mode')
            iter = log.pop('iter')
            # switching epoch
            if epoch != last_epoch:
                epoch_iter += last_iter
            lr = log.pop('lr')
            for k, v in log.items():
                tag = f'{k}/{mode}'
                metrics[tag] = v
            metrics['learning_rate'] = lr
            metrics['momentum'] = 0.9
            metrics['step'] = epoch_iter + iter
            metrics_list.append(metrics)
            last_epoch = epoch
            last_iter = iter
    return metrics_list


def upload(project, config, log_path):

    metrics_list = load_json_log(log_path)
    cfg = mmcv.Config.fromfile(config)
    init_kwargs = dict(
        project=project,
        name=osp.splitext(osp.basename(config))[0],
        tags=['converted'],
        config=dict(
            model=cfg.model,
            train_cfg=cfg.train_cfg,
            test_cfg=cfg.test_cfg,
            data=cfg.data))
    wandb.init(**init_kwargs, reinit=True)
    for metrics in metrics_list:
        step = metrics.pop('step')
        wandb.log(metrics, step=step)


def main():
    args = parse_args()

    log_dir = args.log_dir
    for cfg in mmcv.scandir(log_dir, suffix='.py', recursive=True):
        if 'playground' in cfg:
            continue
        cfg_dir = osp.dirname(osp.join(log_dir, cfg))
        for log in mmcv.scandir(cfg_dir, suffix='.log.json'):
            print(f'uploading {cfg_dir}')
            upload('converted', osp.join(log_dir, cfg), osp.join(cfg_dir, log))


if __name__ == '__main__':
    main()
