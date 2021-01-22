# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='ImageNetAutoAug'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']), # normalize to [0, 1]
    dict(type='ColorJitterLighting'),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data_root = '/home/xiangli/DataSet/'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_prefix=data_root+'imagenet1k/val',
        ann_file=data_root+'imagenet1k/meta/val.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=data_root+'imagenet1k/val',
        ann_file=data_root+'imagenet1k/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=data_root+'imagenet1k/val',
        ann_file=data_root+'imagenet1k/meta/val.txt',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')

# model settings
model = dict(
    type='MixUpImageClassifier',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
    ),
    #backbone=dict(
    #    type='SKResNet',
    #    depth=50,
    #    num_stages=4,
    #    avg_down_stride=True,
    #    avg_down_after=False,
    #    deep_stem=False,
    #    avg_down=False,
    #    out_indices=(3, ),
    #    style='pytorch',
    #),
    #backbone=dict(
    #    type='SACResNet',
    #    depth=50,
    #    num_stages=4,
    #    avg_down_after=False,
    #    deep_stem=False,
    #    avg_down=False,
    #    out_indices=(3, ),
    #    style='pytorch',
    #),
    #backbone=dict(
    #    type='SK2ResNet',
    #    depth=50,
    #    num_stages=4,
    #    attn_channel=False,
    #    avg_down_after=False,
    #    deep_stem=False,
    #    avg_down=False,
    #    out_indices=(3, ),
    #    style='pytorch',
    #),
    #backbone=dict(
    #    type='SKResNeSt',
    #    depth=50,
    #    num_stages=4,
    #    avg_down_stride=False,
    #    deep_stem=False,
    #    avg_down=False,
    #    out_indices=(3, ),
    #    style='pytorch',
    #),
    #backbone=dict(
    #    type='SK2ResNeSt',
    #    depth=50,
    #    num_stages=4,
    #    attn_channel=False,
    #    avg_down_stride=True,
    #    deep_stem=True,
    #    avg_down=True,
    #    out_indices=(3, ),
    #    style='pytorch',
    #),
    #backbone=dict(
    #    type='RepVGGNet',
    #    num_blocks=[2, 4, 14, 1],
    #    #num_blocks=[4, 6, 16, 1],
    #    width_multiplier=[2, 2, 2, 4],
    #    #single_path=False,

    #    #type='SK2Net',
    #    #type='ResNetV1d',
    #    #type='ResNet',
    #    #type='SK2ResNeSt',
    #    #depth=101,
    #    #num_stages=4,
    #    #attn_channel=False,

    #    #scales=4, 
    #    #base_width=26, 
    #    #cardinality=1,
    #    #sk_groups=13,
    #    #ratio=8,
    #    #deep_stem=True,

    #    #reduction_factor=4,
    #    #down_stride_after=True,
    #    out_indices=(3, ),
    #    style='pytorch'
    #    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        topk=(1, 5),
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            num_classes=1000),
    ))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001,
                paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0)
    #warmup='linear',
    #warmup_iters=2500,
    #warmup_ratio=0.25)
runner = dict(type='EpochBasedRunner', max_epochs=100)

# checkpoint saving
checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
