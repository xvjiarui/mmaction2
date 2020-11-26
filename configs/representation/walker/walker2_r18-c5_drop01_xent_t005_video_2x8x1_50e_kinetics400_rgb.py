# model settings
model = dict(
    type='SpaceTimeWalker',
    backbone=dict(
        type='ResNet',
        pretrained=None,
        depth=18,
        strides=(1, 2, 1, 1),
        out_indices=(3, ),
        norm_eval=False,
        zero_init_residual=True),
    cls_head=dict(
        type='WalkerHeadV2',
        num_classes=400,
        in_channels=512,
        channels=128,
        temperature=0.05,
        loss_cls=dict(type='CrossEntropyLoss'),
        walk_len=7,
        dropout_ratio=0.1))
# model training and testing settings
train_cfg = dict(patch_size=64, patch_stride=32)
test_cfg = dict(
    precede_frames=20,
    topk=10,
    temperature=0.2,
    strides=(1, 2, 1, 1),
    out_indices=(2, 3),
    neighbor_range=24,
    with_first=True,
    with_first_neighbor=True,
    output_dir='eval_results')
# dataset settings
dataset_type = 'VideoDataset'
dataset_type_val = 'DavisDataset'
data_prefix = 'data/kinetics400/videos_train'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
data_prefix_val = 'data/davis/DAVIS/JPEGImages/480p'
anno_prefix_val = 'data/davis/DAVIS/Annotations/480p'
data_root_val = 'data/davis/DAVIS'
ann_file_val = 'data/davis/DAVIS/ImageSets/davis2017_val_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=2, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='SequentialSampleFrames', frame_interval=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 480), keep_ratio=True),
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
    videos_per_gpu=24,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_prefix,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type_val,
        ann_file=ann_file_val,
        data_prefix=data_prefix_val,
        data_root=data_root_val,
        anno_prefix=anno_prefix_val,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type_val,
        ann_file=ann_file_val,
        data_prefix=data_prefix_val,
        data_root=data_root_val,
        anno_prefix=anno_prefix_val,
        pipeline=val_pipeline,
        test_mode=True))
# optimizer
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='CosineAnnealing', min_lr=0)
lr_config = dict(policy='Fixed')
total_epochs = 50
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1, metrics='davis', save_best=False, key_indicator=None)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mmaction2',
                name='{{fileBasenameNoExtension}}',
                resume=True,
                tags=['walker2'],
                dir='wandb/{{fileBasenameNoExtension}}',
                config=dict(
                    model=model,
                    train_cfg=train_cfg,
                    test_cfg=test_cfg,
                    data=data))),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = False
