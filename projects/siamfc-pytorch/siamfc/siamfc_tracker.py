import datetime
import os.path as osp
import time

import kornia.augmentation as K
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from got10k.trackers import Tracker
from mmcv.runner import load_checkpoint
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import DataLoader

from mmaction.models import build_backbone
from . import ops
from .datasets import Pair
from .heads import SiamConvFC, SiamFC
from .losses import BalancedLoss, FocalLoss
from .transforms import SiamFCTransforms


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, cfg, logger):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = cfg
        self.logger = logger

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        backbone = build_backbone(cfg.model.backbone)
        backbone = _convert_batchnorm(backbone)
        backbone.init_weights()
        if cfg.extra_conv:
            self.net = Net(
                backbone=backbone,
                head=SiamConvFC(
                    cfg.out_channels, 256, out_scale=self.cfg.out_scale))
        else:
            self.net = Net(
                backbone=backbone, head=SiamFC(out_scale=self.cfg.out_scale))
        logger.info(f'Model: {str(self.net)}')
        if cfg.checkpoint is not None:
            load_checkpoint(self.net, cfg.checkpoint, map_location='cpu')

        self.net = self.net.to(self.device)

        # setup criterion
        if cfg.loss == 'balance':
            self.criterion = BalancedLoss()
        elif cfg.loss == 'focal':
            self.criterion = FocalLoss()
        else:
            raise NotImplementedError

        # setup optimizer
        if cfg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.net.parameters(),
                lr=self.cfg.initial_lr,
                weight_decay=self.cfg.weight_decay
                if cfg.model.backbone.frozen_stages < 4 else 0,
                momentum=self.cfg.momentum)
        elif cfg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.net.parameters(),
                lr=self.cfg.initial_lr,
                weight_decay=self.cfg.weight_decay
                if cfg.model.backbone.frozen_stages < 4 else 0)
        else:
            raise NotImplementedError

        # for #param
        self.net.train()
        num_trainable_params = len(
            [p for p in self.net.parameters() if p.requires_grad])
        num_params = len([p for p in self.net.parameters()])
        logger.info(f'Number of trainable parameters: {num_trainable_params}')
        logger.info(f'Number of total parameters: {num_params}')

        # setup lr scheduler
        if cfg.lr_schedule == 'exp':
            gamma = np.power(self.cfg.ultimate_lr / self.cfg.initial_lr,
                             1.0 / self.cfg.epoch_num)
            self.lr_scheduler = ExponentialLR(self.optimizer, gamma)
        elif cfg.lr_schedule == 'step':
            self.lr_scheduler = StepLR(self.optimizer, cfg.lr_step_size)
        else:
            raise NotImplementedError

        self.normalize = K.Normalize(
            mean=torch.tensor([123.675, 116.28, 103.53]),
            std=torch.tensor([58.395, 57.12, 57.375]))

    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2, box[0] - 1 +
            (box[2] - 1) / 2, box[3], box[2]
        ],
                       dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz), np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step**np.linspace(
            -(self.cfg.scale_num // 2), self.cfg.scale_num // 2,
            self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img,
            self.center,
            self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # exemplar features
        z = torch.from_numpy(z).to(self.device).permute(
            2, 0, 1).unsqueeze(0).float()
        z = self.normalize(z)
        self.kernel = self.net.backbone(z)

    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [
            ops.crop_and_resize(
                img,
                self.center,
                self.x_sz * f,
                out_size=self.cfg.instance_sz,
                border_value=self.avg_color) for f in self.scale_factors
        ]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float()
        x = self.normalize(x)

        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        # responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        # responses = np.stack([
        #     cv2.resize(
        #         u, (self.upscale_sz, self.upscale_sz),
        #         interpolation=cv2.INTER_CUBIC) for u in responses
        # ])
        responses = F.interpolate(
            responses,
            size=(self.upscale_sz, self.upscale_sz),
            mode='bicubic',
            align_corners=False)
        responses = responses.squeeze(1).cpu().numpy()
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]
        ])

        return box

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)
        prog_bar = mmcv.ProgressBar(len(img_files))

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])
            prog_bar.update()

        return boxes, times

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            z = self.normalize(z)
            x = self.normalize(x)
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs):
        # set to train mode
        self.net.train()

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms,
            pairs_per_seq=self.cfg.pairs_per_seq)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # loop over epochs
        start_time = time.perf_counter()
        last_iter = 0
        total_iters = self.cfg.epoch_num * len(dataloader)
        for epoch in range(self.cfg.epoch_num):
            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                if (it + 1) % self.cfg.log_config.interval == 0 or it == (
                        len(dataloader) - 1):
                    passed_iters = it + epoch * len(dataloader) + 1
                    total_seconds_per_img = (time.perf_counter() -
                                             start_time) / (
                                                 passed_iters - last_iter)
                    eta = datetime.timedelta(
                        seconds=int(total_seconds_per_img *
                                    (total_iters - passed_iters - 1)))
                    self.logger.info(
                        f'Epoch: {epoch+1} [{it+1}/{len(dataloader)}]'
                        f' ETA: {str(eta)}'
                        f' lr: {self.current_lr()[0]:.5f}'
                        f' Loss: {loss:.5f}')
                    last_iter = passed_iters
                    start_time = time.perf_counter()
            # update lr at each epoch
            self.lr_scheduler.step()

            if epoch == self.cfg.epoch_num - 1:
                save_dir = osp.join(self.cfg.work_dir, self.cfg.suffix)
                # save checkpoint
                mmcv.mkdir_or_exist(save_dir)
                net_path = osp.join(save_dir, f'epoch_{epoch + 1}.pth')
                torch.save(self.net.state_dict(), net_path)
                self.logger.info(f'{net_path} saved')
                dst_file = osp.join(save_dir, 'latest.pth')
                mmcv.symlink(osp.basename(net_path), dst_file)

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(
                dist <= r_pos, np.ones_like(x),
                np.where(dist < r_neg,
                         np.ones_like(x) * 0.5, np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()

        return self.labels
