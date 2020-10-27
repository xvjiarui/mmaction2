# model settings
temperature = 0.01
with_norm = True
query_dim = 128
model = dict(
    type='UVCNeckMoCoTrackerV2',
    queue_dim=query_dim,
    patch_queue_size=256 * 144 * 5,
    backbone=dict(
        type='ResNet',
        pretrained=None,
        depth=18,
        out_indices=(3, ),
        # strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        zero_init_residual=True),
    cls_head=dict(
        type='UVCHead',
        loss_feat=None,
        loss_aff=dict(
            type='ConcentrateLoss',
            win_len=8,
            stride=8,
            temperature=temperature,
            with_norm=with_norm,
            loss_weight=1.),
        loss_bbox=dict(type='L1Loss', loss_weight=10.),
        in_channels=256,
        channels=128,
        temperature=temperature,
        with_norm=with_norm,
        init_std=0.01,
        track_type='center'),
    patch_head=None,
    img_head=dict(
        type='MoCoHead',
        loss_feat=dict(type='MultiPairNCE', loss_weight=1.),
        in_channels=512,
        # num_convs=2,
        # kernel_size=3,
        # norm_cfg=dict(type='BN'),
        # act_cfg=dict(type='ReLU'),
        channels=query_dim,
        temperature=temperature,
        with_norm=with_norm))
# model training and testing settings
train_cfg = dict(
    patch_size=96,
    patch_size_moco=256,
    img_as_ref=True,
    img_as_tar=False,
    img_as_embed=True,
    mix_full_imgs=True,
    img_geo_aug=False,
    diff_crop=True,
    skip_cycle=True,
    center_ratio=0.,
    shuffle_bn=True)
test_cfg = dict(
    precede_frames=7,
    topk=5,
    temperature=temperature,
    strides=(1, 2, 1, 1),
    out_indices=(2, 3),
    neighbor_range=40,
    with_norm=with_norm,
    output_dir='eval_results')
# dataset settings
dataset_type = 'ImageDataset'
dataset_type_val = 'DavisDataset'
data_prefix = 'data/imagenet/2012/train'
ann_file_train = 'data/imagenet/2012/train_map.txt'
data_prefix_val = 'data/davis/DAVIS/JPEGImages/480p'
anno_prefix_val = 'data/davis/DAVIS/Annotations/480p'
data_root_val = 'data/davis/DAVIS'
ann_file_val = 'data/davis/DAVIS/ImageSets/davis2017_val_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=8, num_clips=1),
    dict(type='DuplicateFrames', times=2),
    dict(type='RawImageDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='RandomResizedCrop',
        area_range=(0.2, 1.),
        same_across_clip=False,
        same_on_clip=False),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        same_across_clip=False,
        same_on_clip=False),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
        p=0.8,
        same_across_clip=False,
        same_on_clip=False),
    dict(
        type='RandomGrayScale',
        p=0.2,
        same_across_clip=False,
        same_on_clip=False),
    dict(
        type='RandomGaussianBlur',
        p=0.5,
        same_across_clip=False,
        same_on_clip=False),
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
    videos_per_gpu=96,
    workers_per_gpu=16,
    val_workers_per_gpu=1,
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
# optimizer = dict(type='Adam', lr=1e-4)
optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
# lr_config = dict(policy='Fixed')
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=100,
#     warmup_ratio=0.001,
#     step=[1, 2])
total_epochs = 30
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1,
    metrics='davis',
    key_indicator='feat_1.J&F-Mean',
    rule='greater')
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
                tags=['moco2'],
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
