import os

# DATA_ROOT environment variable points to your dataset root (or set manually).
_data_root = os.environ.get('DATA_ROOT')

# tidy namespace
del os

# Inherit runtime and schedule defaults from base configs.
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=50000)
checkpoint_config = dict(by_epoch=False, interval=2500)
evaluation = dict(interval=2500, metric='mIoU')


# Model settings (backbone + decode head). Uses SyncBN for multi-GPU training.
# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    # Encoder-decoder model that accepts a coarse mask for refinement.
    type='EncoderDecoderRefine',
    # Pretrained HRNet weights from the OpenMMLab model zoo for faster
    # convergence (optional).
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        # HRNet-based backbone configured to W48-like widths.
        type='HRNetRefine',
        norm_cfg=norm_cfg,
        # Keep BN in train mode (norm_eval=False).
        norm_eval=False,
        # `extra` sets stage/branch/block/channel details (HRNet-W48).
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
    # Light-weight FCN head that concat-resizes branch outputs.
        type='FCNHead',
        in_channels=[48, 96, 192, 384],
        in_index=(0, 1, 2, 3),
    # `channels` equals the concatenated channel dimension of branches.
        channels=sum([48, 96, 192, 384]),
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,  # disable dropout
        num_classes=2,     # binary segmentation
        norm_cfg=norm_cfg,
        align_corners=False,
        # CrossEntropyLoss (softmax). For single-channel binary, use sigmoid.
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))
# Training/testing behavior. `mode='whole'` runs full-image inference.
train_cfg = dict()
test_cfg = dict(mode='whole')

# Dataset settings. `RefineDataset` should provide 'coarse_mask' with each sample.
dataset_type = 'RefineDataset'
data_root = _data_root
# Image normalization settings.
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)

# Fixed input size (256x256) and data pipelines for train/val/test.
crop_size = (256, 256)

# Training pipeline: loads image, gt, and coarse mask; applies augmentations.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadCoarseMask'),
    # Fixed resize; no random scaling.
    dict(type='Resize', img_scale=crop_size, ratio_range=(1.0, 1.0)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'coarse_mask']),
]
# Test/validation pipeline (single-scale, no flip).
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadCoarseMask'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=True,
        flip_direction=['horizontal', 'vertical'],
        transforms=[
            dict(type='Resize', img_scale=crop_size, keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'coarse_mask']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        mask_dir='mask_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        mask_dir='mask_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        mask_dir='mask_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline))
