"""BPR HRNet-48 + Flip-Consistency Training (FCT), wired to the POLYP task.

Unlike configs/bpr/hrnet48_256_fct.py -- which inherits the *general*
hrnet48_256.py (SyncBN, PhotoMetricDistortion, a 160k-iter/16k-eval schedule,
and a flip=True eval pipeline) -- this config inherits the polyp-specific
poly_hrnet48_256.py, so FCT actually applies to the polyp benchmark:
  - BN (single-GPU friendly), the 50k-iter / 2.5k-eval schedule,
  - RefineDataset + PraNet coarse masks via the DATA_ROOT env var.

It only overrides:
  1. the segmentor -> EncoderDecoderRefineFCT (+ FCT hyperparameters),
  2. samples_per_gpu, reduced so per-iter compute stays close to the baseline
     under FCT's internal batch expansion, and
  3. the val/test pipeline -> flip=False (single view).

Why (3): the poly base test pipeline uses MultiScaleFlipAug(flip=True), which
routes evaluation through the segmentor's aug_test(). EncoderDecoderRefine only
implements simple_test(img, img_meta, coarse_mask); the inherited
EncoderDecoder.aug_test(imgs, img_metas, rescale) has NO coarse_mask parameter,
so in-training evaluation would raise `TypeError: aug_test() got an unexpected
keyword argument 'coarse_mask'`. Forcing flip=False makes num_augs==1, which
routes evaluation through simple_test (correct). Real test-time 4-view TTA is
done separately and correctly by tools/test_bpr_tta_manual.py.

Train:
  python tools/train_poly.py configs/bpr/poly_hrnet48_256_fct.py \
      --gpus 1 --work-dir work_dirs/poly_hrnet48_256_fct
  (set DATA_ROOT to your RefineDataset patch root first)
Test-time TTA:
  bash run_bpr_tta_eval.sh   # uses tools/test_bpr_tta_manual.py
"""

_base_ = ['./poly_hrnet48_256.py']

# The poly base already uses BN; restate for clarity/robustness.
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoderRefineFCT',
    # ----- FCT hyperparameters (tunable) -----
    # Peak weight of the consistency (MSE) loss.
    fct_weight=0.05,
    # Linear ramp 0 -> fct_weight over the first N iters (short warmup).
    fct_warmup_iters=300,
    # Polyps are invariant to vertical flip, and the test-time TTA averages
    # H/V/HV views, so enforcing vertical consistency at train time matches
    # inference. Set False to only use H-flip (2x batch instead of 3x).
    fct_use_vflip=True,
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
)

# ----- Evaluation pipeline: single view (flip=False) -----
# See the module docstring for why flip=True would crash aug_test().
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadCoarseMask'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=crop_size, keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'coarse_mask']),
        ])
]

# FCT with fct_use_vflip=True triples each batch internally (orig + H + V).
# The poly base uses samples_per_gpu=16; drop to 6 so the effective per-iter
# batch (~18) stays close to the baseline. Use 8 if fct_use_vflip=False (2x).
data = dict(
    samples_per_gpu=6,
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)
