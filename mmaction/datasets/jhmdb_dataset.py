import copy
import os
import os.path as osp

import mmcv
import numpy as np
import scipy.io as sio
from mmcv.utils import print_log

from mmaction.utils import add_prefix, terminal_is_available
from .rawframe_dataset import RawframeDataset
from .registry import DATASETS


@DATASETS.register_module()
class JHMDBDataset(RawframeDataset):

    NUM_KEYPOINTS = 15

    PALETTE = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
               [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
               [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
               [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255]]

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 anno_prefix=None,
                 test_mode=False,
                 split='val',
                 data_root='data/davis2017',
                 task='semi-supervised'):
        assert split in ['train', 'val']
        assert task in ['semi-supervised']
        self.split = split
        self.data_root = data_root
        self.task = task
        self.anno_prefix = anno_prefix
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            filename_tmpl='{:05}.png',
            with_offset=False,
            multi_class=False,
            num_classes=None,
            start_index=1,
            modality='RGB')

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        ann_frame_dir = results['frame_dir'].replace(self.data_prefix,
                                                     self.anno_prefix)
        # "pos_img" has key points, shape [2, 15, clip_len]
        pose_path = osp.join(
            ann_frame_dir.replace('Frames', 'joint_positions'),
            'joint_positions.mat')
        pose_mat = sio.loadmat(pose_path)
        # magic -1
        results['pose_coord'] = pose_mat['pos_img'][..., 0] - 1
        return self.pipeline(results)

    @staticmethod
    def compute_pck(distAll, distThresh):

        pckAll = np.zeros((len(distAll), ))
        for pidx in range(len(distAll)):
            idxs = np.argwhere(distAll[pidx] <= distThresh)
            pck = 100.0 * len(idxs) / len(distAll[pidx])
            pckAll[pidx] = pck

        return pckAll

    def img2coord(self, imgs, topk=5):
        clip_len = len(imgs)
        height, width = imgs.shape[2:]
        assert imgs.shape[:2] == (clip_len, self.NUM_KEYPOINTS)
        coords = np.zeros((2, self.NUM_KEYPOINTS, clip_len), dtype=np.float)
        imgs = imgs.reshape(clip_len, self.NUM_KEYPOINTS, -1)
        assert imgs.shape[-1] == height * width
        # [clip_len, NUM_KEYPOINTS, topk]
        topk_indices = np.argsort(imgs, axis=-1)[..., -topk:]
        topk_values = np.take_along_axis(imgs, topk_indices, axis=-1)
        topk_values = topk_values / np.sum(topk_values, keepdims=True, axis=-1)
        topk_x = topk_indices % width
        topk_y = topk_indices // width
        # [clip_len, NUM_KEYPOINTS]
        coords[0] = np.sum(topk_x * topk_values, axis=-1).T
        coords[1] = np.sum(topk_y * topk_values, axis=-1).T
        coords[:, np.sum(imgs.transpose(1, 0, 2), axis=-1) == 0] = -1

        return coords

    def pck_evaluate(self, results, output_dir, logger=None):

        dist_all = [np.zeros((0, 0)) for _ in range(self.NUM_KEYPOINTS)]
        if terminal_is_available():
            prog_bar = mmcv.ProgressBar(len(self))
        for vid_idx in range(len(results)):
            cur_results = results[vid_idx]
            if isinstance(cur_results, str):
                file_path = cur_results
                cur_results = np.load(file_path)
                os.remove(file_path)
            pose_path = osp.join(
                self.video_infos[vid_idx]['frame_dir'].replace(
                    self.data_prefix,
                    self.anno_prefix).replace('Frames', 'joint_positions'),
                'joint_positions.mat')
            gt_poses = sio.loadmat(pose_path)['pos_img'] - 1

            # get predict poses
            clip_len = self.video_infos[vid_idx]['total_frames']
            # truncate according to GT
            clip_len = min(clip_len, gt_poses.shape[-1])
            cur_results = cur_results[:clip_len]

            assert len(
                cur_results) == clip_len, f'{len(cur_results)} vs {clip_len}'
            # for img_idx in range(clip_len):
            #     result = cur_results[img_idx]
            #     assert result.ndim == 3
            #     assert result.shape[0] == self.NUM_KEYPOINTS
            #     coord = np.zeros((2, self.NUM_KEYPOINTS))
            #     top_ids = 5
            #     for t in range(self.NUM_KEYPOINTS):
            #         pred = result[t]
            #         pred = pred.reshape(-1)
            #         if pred.sum() == 0:
            #             coord[:, t] = -1
            #             continue
            #         ids = np.argsort(pred)[-top_ids:]
            #         vals = np.sort(pred)[-top_ids:]
            #
            #         vals = vals / vals.sum()
            #
            #         for i in range(len(ids)):
            #             nowid = ids[i]
            #             nowx = nowid % result.shape[2]
            #             nowy = nowid // result.shape[2]
            #             coord[0, t] += nowx * vals[i]
            #             coord[1, t] += nowy * vals[i]
            #     pred_poses.append(coord)
            # # [2, 15, clip_len]
            # pred_poses = np.stack(pred_poses, axis=-1)
            # [2, 15, clip_len]
            pred_poses = self.img2coord(cur_results)
            assert pred_poses.shape == gt_poses.shape, \
                f'{pred_poses.shape} vs {gt_poses.shape}'
            # no prediction
            # [15, clip_len]
            joint_visible = pred_poses[0] > 0
            # TODO verctorlized is slow or not? fast
            valid_max_gt_poses = gt_poses.copy()
            valid_max_gt_poses[:, ~joint_visible] = -1
            valid_min_gt_poses = gt_poses.copy()
            valid_min_gt_poses[:, ~joint_visible] = 1e6
            boxes = np.stack((valid_max_gt_poses[0].max(axis=0) -
                              valid_min_gt_poses[0].min(axis=0),
                              valid_max_gt_poses[1].max(axis=0) -
                              valid_min_gt_poses[1].min(axis=0)),
                             axis=0)
            # [clip_len]
            boxes = 0.6 * np.linalg.norm(boxes, axis=0)
            # TODO check img_idx=0 pose: small
            # boxes = [None] * clip_len
            # for img_idx in range(clip_len):
            #     minx = 1e6
            #     maxx = -1
            #     miny = 1e6
            #     maxy = -1
            #     for t in range(self.NUM_KEYPOINTS):
            #         if not joint_visible[t, img_idx]:
            #             continue
            #         minx = np.min([minx, gt_poses[0, t, img_idx]])
            #         miny = np.min([miny, gt_poses[1, t, img_idx]])
            #         maxx = np.max([maxx, gt_poses[0, t, img_idx]])
            #         maxy = np.max([maxy, gt_poses[1, t, img_idx]])
            #     cur_bbox = 0.6 * np.linalg.norm(
            #         np.subtract([maxx, maxy], [minx, miny]))
            #     boxes[img_idx] = cur_bbox
            for img_idx in range(clip_len):
                for t in range(self.NUM_KEYPOINTS):
                    if not joint_visible[t, img_idx]:
                        continue
                    predx = pred_poses[0, t, img_idx]
                    predy = pred_poses[1, t, img_idx]
                    gtx = gt_poses[0, t, img_idx]
                    gty = gt_poses[1, t, img_idx]
                    dist = np.linalg.norm(
                        np.subtract([predx, predy], [gtx, gty]))
                    dist = dist / boxes[img_idx]

                    dist_all[t] = np.append(dist_all[t], [[dist]])
            if terminal_is_available():
                prog_bar.update()
        pck_ranges = (0.1, 0.2, 0.3, 0.4, 0.5)
        pck_all = []
        for pck_range in pck_ranges:
            pck_all.append(self.compute_pck(dist_all, pck_range))
        eval_results = {}
        for alpha, pck in zip(pck_ranges, pck_all):
            eval_results[f'PCK@{alpha}'] = np.mean(pck)

        return eval_results

    def evaluate(self, results, metrics='pck', output_dir=None, logger=None):
        print('evaluate')
        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['pck']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        eval_results = dict()
        if mmcv.is_seq_of(results, list):
            num_feats = len(results[0])
            for feat_idx in range(num_feats):
                cur_results = [result[feat_idx] for result in results]
                eval_results.update(
                    add_prefix(
                        self.pck_evaluate(cur_results, output_dir, logger),
                        prefix=f'feat_{feat_idx}'))
        else:
            eval_results.update(self.pck_evaluate(results, output_dir, logger))
        copypaste = []
        for k, v in list(eval_results.items())[:2]:
            copypaste.append(f'{float(v):.2f}')
        print_log(f'Results copypaste  {",".join(copypaste)}', logger=logger)
        return eval_results
