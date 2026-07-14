# configs/bpr/hrnet48_256_fct_noflip.py
_base_ = ['./hrnet48_256.py']

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)

# Training pipeline WITHOUT RandomFlip
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadCoarseMask'),
    dict(type='Resize', img_scale=crop_size, ratio_range=(1.0, 1.0)),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'coarse_mask']),
]

# Override SyncBN (inherited from hrnet48_256.py) with plain BN. SyncBN
# requires a distributed process group and crashes non-distributed / single-GPU
# runs; the polyp baseline (poly_hrnet48_256.py) uses BN.
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoderRefineFCT',
    fct_weight=0.3,       # back to original since no longer competing with flip aug
    fct_warmup_iters=300,
    fct_use_vflip=True,
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
)

data = dict(
    samples_per_gpu=6,
    train=dict(pipeline=train_pipeline),
)
