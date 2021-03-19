# dataset settings
data_root = 'data/vip/VIP_Fine'
dataset_type_val = 'VIPDataset'
data_prefix_val = 'data/vip/VIP_Fine/Images'
anno_prefix_val = 'data/vip/VIP_Fine/Annotations/Category_ids'
ann_file_val = 'data/vip/vip_val_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
# CRW MEAN STD
# img_norm_cfg = dict(
#     mean=[0.4914 * 255, 0.4822 * 255, 0.4465 * 255],
#     std=[0.2023 * 255, 0.1994 * 255, 0.2010 * 255], to_bgr=False)
val_pipeline = [
    dict(type='SequentialSampleFrames', frame_interval=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 560), keep_ratio=True),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'ref_seg_map'],
        meta_keys=('frame_dir', 'frame_inds', 'original_shape', 'seg_map')),
    dict(type='ToTensor', keys=['imgs', 'ref_seg_map'])
]
data = dict(
    workers_per_gpu=0,
    val=dict(
        type=dataset_type_val,
        ann_file=ann_file_val,
        data_prefix=data_prefix_val,
        data_root=data_root,
        anno_prefix=anno_prefix_val,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type_val,
        ann_file=ann_file_val,
        data_prefix=data_prefix_val,
        data_root=data_root,
        anno_prefix=anno_prefix_val,
        pipeline=val_pipeline,
        test_mode=True))
test_cfg = dict(
    precede_frames=1,
    topk=10,
    temperature=0.2,
    strides=(1, 2, 1, 1),
    out_indices=(2, ),
    neighbor_range=24,
    with_first=True,
    with_first_neighbor=False,
    output_dir='eval_results')
