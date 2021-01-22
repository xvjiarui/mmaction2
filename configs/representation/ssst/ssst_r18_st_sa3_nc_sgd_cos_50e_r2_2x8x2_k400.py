# model settings
temperature = 0.2
with_norm = True
query_dim = 128
model = dict(
    type='SimSiamBaseSTSNTracker',
    backbone=dict(
        type='ResNet',
        pretrained=None,
        depth=18,
        out_indices=(3, ),
        # strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        zero_init_residual=True),
    # cls_head=None,
    # patch_head=None,
    att_plugin=dict(
        type='SelfAttentionBlock',
        query_in_channels=512,
        channels=256,
        value_out_norm=True,
        with_out=True,
        norm_cfg=dict(type='SyncBN'),
        # norm_cfg=dict(type='LN', normalized_shape=(128, 14*14)),
        zero_init=True),
    img_head=dict(
        type='SimSiamHead',
        in_channels=512,
        norm_cfg=dict(type='SyncBN'),
        num_projection_fcs=3,
        projection_mid_channels=512,
        projection_out_channels=512,
        num_predictor_fcs=2,
        predictor_mid_channels=128,
        predictor_out_channels=512,
        with_norm=True,
        loss_feat=dict(type='CosineSimLoss', negative=False),
        spatial_type='avg'))
# model training and testing settings
train_cfg = dict(
    att_indices=(3, ),
    self_as_value=True,
    pred_frame_index=1,
    target_frame_index=-2,
    target_att=True)
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
    dict(type='SampleFrames', clip_len=2, frame_interval=8, num_clips=2),
    # dict(type='DuplicateFrames', times=2, as_clip=False),
    # dict(type='Frame2Clip'),
    # dict(
    #     type='AppendFrames',
    #     num_frames=1,
    #     frame_interval=8,
    #     temporal_jitter=False),
    dict(type='DecordDecode'),
    dict(
        type='RandomResizedCrop',
        area_range=(0.2, 1.),
        same_across_clip=False,
        same_on_clip=False,
        same_frame_indices=(1, )),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        same_across_clip=False,
        same_on_clip=False,
        same_frame_indices=(1, )),
    # dict(
    #     type='ColorJitter',
    #     brightness=0.4,
    #     contrast=0.4,
    #     saturation=0.4,
    #     hue=0.1,
    #     p=0.8,
    #     same_across_clip=False,
    #     same_on_clip=False),
    # dict(
    #     type='RandomGrayScale',
    #     p=0.2,
    #     same_across_clip=False,
    #     same_on_clip=False),
    # dict(
    #     type='RandomGaussianBlur',
    #     p=0.5,
    #     same_across_clip=False,
    #     same_on_clip=False),
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
    videos_per_gpu=128,
    workers_per_gpu=16,
    val_workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            data_prefix=data_prefix,
            pipeline=train_pipeline)),
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
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
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
total_epochs = 50
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
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
                tags=['ssst'],
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
