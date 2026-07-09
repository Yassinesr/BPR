"""BPR HRNet-48 with Flip-Consistency Training (FCT).

Inherits all settings from hrnet48_256.py and overrides:
  - Segmentor type -> EncoderDecoderRefineFCT
  - FCT hyperparameters
  - Batch size (halved to compensate for the 3x internal expansion)
"""

_base_ = ['./hrnet48_256.py']

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoderRefineFCT',
    # FCT-specific args
    fct_weight=0.05,
    fct_warmup_iters=300,
    fct_use_vflip=False,
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
)

# Reduce batch size because FCT internally triples each batch.
# hrnet48_256.py uses samples_per_gpu=24. We drop to 8 so the
# effective tripled-batch size is 24 (same compute as the baseline).
data = dict(samples_per_gpu=8)

# Keep all other settings from the base config (optimizer, schedule,
# checkpoint settings, etc).
