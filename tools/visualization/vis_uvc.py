import argparse
import os.path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from torchvision.utils import save_image

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.models.common import get_crop_grid, images2video, video2images


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
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

    cfg = Config.fromfile(args.config)
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
        shuffle=False)

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
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        batches, clip_len = imgs.size(0), imgs.size(2)
        # x = images2video(
        #     self.extract_feat(self.aug(video2images(imgs))), clip_len)
        x = images2video(self.extract_feat(video2images(imgs)), clip_len)
        assert x.size(1) == 512
        assert clip_len == 2
        step = 2
        ref_frame = imgs[:, :, 0]
        ref_x = x[:, :, 0]
        # all bboxes are in feature space
        # ref_bboxes, is_center_crop = get_random_crop_bbox(
        #     batches,
        #     self.patch_x_size,
        #     ref_x.shape[2:],
        #     device=x.device,
        #     center_ratio=self.train_cfg.center_ratio,
        #     border=self.border//self.stride)
        # is_center_crop = False
        # ref_bboxes = get_top_diff_crop_bbox(imgs[:, :, 0], imgs[:, :, -1],
        #                                     self.patch_img_size,
        #                                     self.grid_size,
        #                                     device=x.device)/self.stride
        ref_bboxes, is_center_crop = self.get_ref_crop_bbox(batches, imgs)
        ref_crop_x = self.crop_x_from_img(ref_frame, ref_x, ref_bboxes,
                                          self.train_cfg.img_as_ref)
        ref_crop_grid = get_crop_grid(ref_frame, ref_bboxes * self.stride,
                                      self.patch_img_size)
        forward_hist = [(ref_bboxes, ref_crop_x)]
        for tar_idx in range(1, step):
            last_bboxes, last_crop_x = forward_hist[-1]
            tar_frame = imgs[:, :, tar_idx]
            tar_x = x[:, :, tar_idx]
            tar_bboxes, tar_crop_x = self.track(
                tar_frame,
                tar_x,
                last_crop_x,
                tar_bboxes=ref_bboxes if is_center_crop else None)
            forward_hist.append((tar_bboxes, tar_crop_x))
        assert len(forward_hist) == step

        backward_hist = [forward_hist[-1]]
        for last_idx in reversed(range(1, step)):
            tar_idx = last_idx - 1
            last_bboxes, last_crop_x = backward_hist[-1]
            tar_frame = imgs[:, :, tar_idx]
            tar_x = x[:, :, tar_idx]
            tar_bboxes, tar_crop_x = self.track(
                tar_frame,
                tar_x,
                last_crop_x,
                tar_bboxes=ref_bboxes if is_center_crop else None)
            backward_hist.append((tar_bboxes, tar_crop_x))
        assert len(backward_hist) == step

        loss_step = dict()
        ref_pred_bboxes = backward_hist[-1][0]
        ref_pred_crop_grid = get_crop_grid(ref_frame,
                                           ref_pred_bboxes * self.stride,
                                           self.patch_img_size)
        loss_step['dist_bbox'] = self.cls_head.loss_bbox(
            ref_pred_bboxes / self.patch_x_size[0],
            ref_bboxes / self.patch_x_size[0])
        loss_step['loss_bbox'] = self.cls_head.loss_bbox(
            ref_crop_grid, ref_pred_crop_grid)

        bbox1 = forward_hist[0][0] * self.stride
        bbox2 = forward_hist[1][0] * self.stride
        bbox3 = backward_hist[0][0] * self.stride
        bbox4 = backward_hist[1][0] * self.stride
        assert torch.allclose(bbox2, bbox3)

        hidden_outputs[name] = (bbox1, bbox2, bbox4, loss_step)

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
            img_tensor = data['imgs']
            # img_norm_cfg = dict(
            #     mean=[123.675, 116.28, 103.53],
            #     std=[58.395, 57.12, 57.375],
            #     to_rgb=False)
            img_norm_cfg = dict(
                mean=[50, 0, 0], std=[50, 127, 127], to_rgb=False)

            img_tensor = img_tensor.reshape((-1, ) + img_tensor.shape[2:])
            assert img_tensor.size(2) == 2, img_tensor.shape

            imgs = []
            for _ in range(img_tensor.size(2)):
                img = mmcv.tensor2imgs(img_tensor[:, :, _], **img_norm_cfg)
                for k in range(len(img)):
                    img[k] = mmcv.imconvert(img[k], 'lab', 'rgb')
                imgs.append(img)

            assert len(hidden_outputs) == 1

            bbox1, bbox2, bbox4, loss_step = list(hidden_outputs.values())[0]
            print(loss_step)
            imgs_show = []
            for idx in range(batch_size):
                imgs_show.append(
                    torch.from_numpy(
                        mmcv.imshow_bboxes(
                            imgs[0][idx].copy(),
                            bbox1[idx].unsqueeze(0).detach().cpu().numpy(),
                            show=False)))
                imgs_show.append(
                    torch.from_numpy(
                        mmcv.imshow_bboxes(
                            imgs[1][idx].copy(),
                            bbox2[idx].unsqueeze(0).detach().cpu().numpy(),
                            show=False)))
                imgs_show.append(
                    torch.from_numpy(
                        mmcv.imshow_bboxes(
                            imgs[0][idx].copy(),
                            bbox4[idx].unsqueeze(0).detach().cpu().numpy(),
                            show=False)))
            mmcv.mkdir_or_exist(out_dir)
            save_image(
                torch.stack(imgs_show).permute(0, 3, 1, 2) / 255.,
                osp.join(out_dir, f'{i:05d}.png'),
                nrow=3)

        hidden_outputs.clear()
        for _ in range(batch_size):
            prog_bar.update()


if __name__ == '__main__':
    main()
