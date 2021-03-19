# dataset settings
data_root = 'data/jhmdb/JHMDB'
dataset_type_val = 'JHMDBDataset'
data_prefix_val = 'data/jhmdb/JHMDB/Frames'
anno_prefix_val = 'data/jhmdb/JHMDB/joint_positions'
ann_file_val = 'data/jhmdb/JHMDB/jhmdb_val_split_1_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
# CRW MEAN STD
# img_norm_cfg = dict(
#     mean=[0.4914 * 255, 0.4822 * 255, 0.4465 * 255],
#     std=[0.2023 * 255, 0.1994 * 255, 0.2010 * 255], to_bgr=False)
val_pipeline = [
    dict(type='SequentialSampleFrames', frame_interval=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 320), keep_ratio=True),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'ref_seg_map'],
        meta_keys=('frame_dir', 'frame_inds', 'original_shape', 'pose_coord')),
    dict(type='ImageToTensor', keys=['ref_seg_map']),
    dict(type='ToTensor', keys=['imgs', 'ref_seg_map'])
]
sigma = 0.5 * 8
data = dict(
    workers_per_gpu=0,
    val=dict(
        type=dataset_type_val,
        ann_file=ann_file_val,
        data_prefix=data_prefix_val,
        data_root=data_root,
        anno_prefix=anno_prefix_val,
        pipeline=val_pipeline,
        test_mode=True,
        sigma=sigma),
    test=dict(
        type=dataset_type_val,
        ann_file=ann_file_val,
        data_prefix=data_prefix_val,
        data_root=data_root,
        anno_prefix=anno_prefix_val,
        pipeline=val_pipeline,
        test_mode=True,
        sigma=sigma))
# test_cfg = dict(
#     precede_frames=8,
#     topk=20,
#     temperature=1.,
#     strides=(1, 2, 1, 1),
#     out_indices=(2, ),
#     neighbor_range=None,
#     with_first=True,
#     with_first_neighbor=True,
#     output_dir='eval_results')
test_cfg = dict(
    precede_frames=4,
    topk=20,
    temperature=0.2,
    strides=(1, 2, 1, 1),
    out_indices=(2, ),
    neighbor_range=None,
    with_first=True,
    with_first_neighbor=True,
    output_dir='eval_results')
