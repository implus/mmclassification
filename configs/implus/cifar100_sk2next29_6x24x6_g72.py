# model settings
model = dict(
    type='CutMixImageClassifier',
    #type='ImageClassifier',
    cutmix_prob = 0.5,
    cutmix_beta = 1.,
    backbone=dict(
        type='SK2NeXt29',
        depth=29,
        cardinality=6,
        base_width=24,
        scale=6,
        groups=72,
        #cardinality=8, 
        #bottleneck_width=64,
        #depth=50,
        #num_stages=4,
        #base_width=26,
        #cardinality=1,
        #reduction_factor=4,
        #down_stride_after=True,
        #out_indices=(3, ),
        #style='pytorch'
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=1024,
        topk=(1, 5),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# dataset settings
dataset_type = 'CIFAR100'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=True)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data_prefix = '/home/xiangli/DataSet/cifar/cifar100'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix=data_prefix,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type, data_prefix=data_prefix, pipeline=test_pipeline),
    test=dict(
        type=dataset_type, data_prefix=data_prefix, pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005,
                paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='step', step=[150, 225])
lr_config = dict(policy='CosineAnnealing', min_lr=0,
    warmup='linear',
    warmup_iters=391*5,
    warmup_ratio=0.125)
runner = dict(type='EpochBasedRunner', max_epochs=300)

# checkpoint saving
checkpoint_config = dict(interval=100)
evaluation = dict(interval=1, metric='accuracy')

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
