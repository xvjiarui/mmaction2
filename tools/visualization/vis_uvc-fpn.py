import argparse
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, set_random_seed
from torchvision.utils import save_image

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.models.common import (bbox_overlaps, get_crop_grid, images2video,
                                    video2images)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.show or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--show" or "'
         '--show-dir"')

    # set random seeds
    if args.seed is not None:
        print('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed, deterministic=False)

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        videos_per_gpu=4,
        workers_per_gpu=0,
        dist=distributed,
        shuffle=True)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    assert not distributed
    model = MMDataParallel(model, device_ids=[0])
    single_gpu_vis(model, data_loader, args.show, args.show_dir)


hidden_outputs = {}


def uvc_hook(name):

    def hook(module, input, output):
        self = module
        imgs = input[0]
        assert imgs.size(1) == 2
        # imgs_k = imgs[:, 0].contiguous().reshape(-1, *imgs.shape[2:])
        imgs_q = imgs[:, 1].contiguous().reshape(-1, *imgs.shape[2:])
        batches, clip_len = imgs_q.size(0), imgs_q.size(2)
        if self.with_neck:
            x_q_full = self.encoder_q(video2images(imgs_q))
            track_x = images2video(self.neck(x_q_full), clip_len)
            # x_q = images2video(x_q_full[-1], clip_len)
        else:
            track_x = images2video(
                self.extract_feat(video2images(imgs_q)), clip_len)
            # x_q = track_x
        # assert x.size(1) == 512
        # assert clip_len == 2
        step = clip_len
        ref_frame = imgs_q[:, :, 0].contiguous()
        ref_x = track_x[:, :, 0].contiguous()
        # TODO: all bboxes are in feature space
        ref_bboxes = self.get_ref_crop_bbox(batches, imgs_q)
        ref_crop_x = self.crop_x_from_img(
            ref_frame,
            ref_x,
            ref_bboxes,
            self.extract_feat,
            crop_first=self.img_as_ref)
        ref_crop_grid = self.get_grid(ref_frame, ref_x, ref_bboxes)
        forward_hist = [(ref_bboxes, ref_crop_x)]
        for tar_idx in range(1, step):
            last_bboxes, last_crop_x = forward_hist[-1]
            tar_frame = imgs_q[:, :, tar_idx].contiguous()
            tar_x = track_x[:, :, tar_idx].contiguous()
            tar_bboxes, tar_crop_x = self.track(
                tar_frame, tar_x, last_crop_x, ref_bboxes=last_bboxes)
            forward_hist.append((tar_bboxes, tar_crop_x))
        assert len(forward_hist) == step

        backward_hist = [forward_hist[-1]]
        for tar_idx in reversed(range(step - 1)):
            last_bboxes, last_crop_x = backward_hist[-1]
            tar_frame = imgs_q[:, :, tar_idx].contiguous()
            tar_x = track_x[:, :, tar_idx].contiguous()
            tar_bboxes, tar_crop_x = self.track(
                tar_frame, tar_x, last_crop_x, ref_bboxes=last_bboxes)
            backward_hist.append((tar_bboxes, tar_crop_x))
        assert len(backward_hist) == step

        loss_step = dict()
        ref_pred_bboxes = backward_hist[-1][0]
        ref_pred_crop_grid = get_crop_grid(ref_frame,
                                           ref_pred_bboxes * self.stride,
                                           self.patch_img_size)
        loss_step['iou_bbox'] = bbox_overlaps(
            ref_pred_bboxes, ref_bboxes, is_aligned=True)
        loss_step['loss_bbox'] = self.cls_head.loss_bbox(
            ref_crop_grid, ref_pred_crop_grid)

        bboxes = []
        for idx in range(step):
            bboxes.append(forward_hist[idx][0] * self.stride)
        assert torch.allclose(forward_hist[-1][0], backward_hist[0][0])
        for idx in reversed(range(step - 1)):
            bboxes.append(backward_hist[idx][0] * self.stride)

        hidden_outputs[name] = dict(step=(bboxes, loss_step))
        if self.skip_cycle and step > 2:
            loss_skip = dict()
            tar_frame = imgs_q[:, :, step - 1].contiguous()
            tar_x = track_x[:, :, step - 1].contiguous()
            _, tar_crop_x = self.track(tar_frame, tar_x, ref_crop_x)
            ref_pred_bboxes, ref_pred_crop_x = self.track(
                ref_frame, ref_x, tar_crop_x)
            ref_pred_crop_grid = self.get_grid(ref_frame, ref_x,
                                               ref_pred_bboxes)
            loss_skip['iou_bbox'] = bbox_overlaps(
                ref_pred_bboxes, ref_bboxes, is_aligned=True)
            loss_skip['loss_bbox'] = self.cls_head.loss_bbox(
                ref_crop_grid, ref_pred_crop_grid)
            hidden_outputs[name].update(
                dict(
                    skip=([
                        ref_bboxes * self.stride, tar_bboxes *
                        self.stride, ref_pred_bboxes * self.stride
                    ], loss_skip)))

    return hook


def register_uvc_hook(model):
    model.module.register_forward_hook(uvc_hook('UVC'))
    # for module_name, module in model.module.named_modules():
    #     if 'UVCTrackerV2' in str(
    #             module.__class__):
    #         module.register_forward_hook(uvc_hook(module_name))
    #         print(f'{module_name} is registered')


def single_gpu_vis(model, data_loader, show=False, out_dir=None):
    model.eval()
    # TODO: check switch success
    # model.module.backbone.switch_strides()
    # model.module.backbone.switch_out_indices()
    register_uvc_hook(model)

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        # use the first key as main key to calculate the batch size
        with torch.no_grad():
            model(data['imgs'], data['label'], return_loss=True)
        batch_size = len(next(iter(data.values())))

        if show or out_dir:
            img_tensor = data['imgs'][:, 0]
            img_norm_cfg = dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=False)
            # img_norm_cfg = dict(
            #     mean=[50, 0, 0], std=[50, 127, 127], to_rgb=False)

            # img_tensor = img_tensor.reshape((-1, ) + img_tensor.shape[2:])
            # assert img_tensor.size(2) == 2, img_tensor.shape

            imgs = []
            for _ in range(img_tensor.size(2)):
                img = mmcv.tensor2imgs(img_tensor[:, :, _], **img_norm_cfg)
                # for k in range(len(img)):
                #     img[k] = mmcv.imconvert(img[k], 'lab', 'rgb')
                imgs.append(img)

            assert len(hidden_outputs) == 1

            hidden_dict = list(hidden_outputs.values())[0]
            bboxes, loss_step = hidden_dict['step']
            assert len(bboxes) == 2 * len(imgs) - 1
            print(loss_step)
            imgs_show = []
            for batch_idx in range(batch_size):
                for idx in range(len(imgs)):
                    imgs_show.append(
                        torch.from_numpy(
                            mmcv.imshow_bboxes(
                                imgs[idx][batch_idx].copy(),
                                bboxes[idx][batch_idx].unsqueeze(
                                    0).detach().cpu().numpy(),
                                show=False)))
                for idx in reversed(range(len(imgs) - 1)):
                    imgs_show.append(
                        torch.from_numpy(
                            mmcv.imshow_bboxes(
                                imgs[idx][batch_idx].copy(),
                                bboxes[idx + len(imgs)][batch_idx].unsqueeze(
                                    0).detach().cpu().numpy(),
                                show=False)))
            mmcv.mkdir_or_exist(out_dir)
            save_image(
                torch.stack(imgs_show).permute(0, 3, 1, 2) / 255.,
                osp.join(out_dir, f'{i:05d}_step.png'),
                nrow=len(bboxes))

            if 'skip' in hidden_dict:
                bboxes, loss_skip = hidden_dict['skip']
                imgs_skip = [imgs[0], imgs[-1]]
                imgs_show = []
                for batch_idx in range(batch_size):
                    for idx in range(len(imgs_skip)):
                        imgs_show.append(
                            torch.from_numpy(
                                mmcv.imshow_bboxes(
                                    imgs_skip[idx][batch_idx].copy(),
                                    bboxes[idx][batch_idx].unsqueeze(
                                        0).detach().cpu().numpy(),
                                    show=False)))
                    for idx in reversed(range(len(imgs_skip) - 1)):
                        imgs_show.append(
                            torch.from_numpy(
                                mmcv.imshow_bboxes(
                                    imgs_skip[idx][batch_idx].copy(),
                                    bboxes[idx + len(imgs_skip)][batch_idx].
                                    unsqueeze(0).detach().cpu().numpy(),
                                    show=False)))
                mmcv.mkdir_or_exist(out_dir)
                save_image(
                    torch.stack(imgs_show).permute(0, 3, 1, 2) / 255.,
                    osp.join(out_dir, f'{i:05d}_skip.png'),
                    nrow=len(bboxes))

        hidden_outputs.clear()
        for _ in range(batch_size):
            prog_bar.update()


if __name__ == '__main__':
    main()
