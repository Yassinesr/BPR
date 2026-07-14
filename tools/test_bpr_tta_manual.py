"""Manual 4-view TTA inference for BPR on all 5 polyp test datasets.

Bypasses mmseg's pipeline. Loads images and PraNet coarse masks directly,
applies BPR's preprocessing manually, runs the model with optional 4-view
TTA, and saves binary PNGs for external evaluation.
"""

import argparse
import os
import os.path as osp
import sys

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from tqdm import tqdm

# Make sure the BPR repo is on the path (we're running from tools/)
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from mmseg.models import build_segmentor


DATASETS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']

IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD  = np.array([58.395, 57.12, 57.375],  dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Manual TTA inference for BPR on polyp test datasets')
    parser.add_argument('--config', required=True,
                        help='BPR config file path')
    parser.add_argument('--checkpoint', required=True,
                        help='Trained BPR checkpoint .pth')
    parser.add_argument('--test-root', required=True,
                        help='Folder containing per-dataset subfolders '
                             '(Kvasir, CVC-ClinicDB, ...)')
    parser.add_argument('--out-root', required=True,
                        help='Where to save refined-mask PNGs')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable TTA, single forward pass only')
    parser.add_argument('--crop-size', type=int, nargs=2, default=[256, 256],
                        help='Input crop size (H W). Default: 256 256')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Subset of datasets to process. Default: all 5.')
    return parser.parse_args()


def preprocess(img_bgr, coarse_gray, crop_size):
    """Apply BPR's preprocessing manually.

    Replicates mmseg's Resize(img_scale=crop_size, keep_ratio=True), which
    resizes so the LONGER edge equals max(crop_size) and the shorter edge
    is scaled proportionally. The output is NOT square unless the input
    happened to be square.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = img_rgb.shape[:2]

    # mmseg's keep_ratio=True logic: scale = min(crop_size) / min(H, W)
    # then both dimensions get multiplied by this scale factor.
    # Source: mmcv.image.geometric.imrescale -> _scale_size
    max_long_edge = max(crop_size)
    max_short_edge = min(crop_size)
    scale_factor = min(max_long_edge / max(H_orig, W_orig),
                       max_short_edge / min(H_orig, W_orig))
    new_W = int(W_orig * scale_factor + 0.5)
    new_H = int(H_orig * scale_factor + 0.5)

    img_resized = cv2.resize(img_rgb, (new_W, new_H),
                             interpolation=cv2.INTER_LINEAR).astype(np.float32)
    coarse_resized = cv2.resize(coarse_gray, (new_W, new_H),
                                interpolation=cv2.INTER_NEAREST).astype(np.float32)

    # Normalize image
    img_resized = (img_resized - IMG_MEAN) / IMG_STD

    # Binarize coarse mask -> {0, 1}. Match training's LoadCoarseMask, which
    # binarizes with `> 0` (any nonzero pixel is foreground); using a >127
    # midpoint here would be a train/test mismatch for non-binary coarse masks.
    coarse_bin = (coarse_resized > 0).astype(np.float32)

    # To tensors
    img_t = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).cuda()
    coarse_t = torch.from_numpy(coarse_bin).unsqueeze(0).cuda()
    return img_t, coarse_t


def flip_t(t, direction):
    if direction == 'h':  return torch.flip(t, dims=[-1])
    if direction == 'v':  return torch.flip(t, dims=[-2])
    if direction == 'hv': return torch.flip(t, dims=[-2, -1])
    raise ValueError(direction)


def single_view_prob(model, img, coarse, img_meta, use_coarse_mask):
    """Returns sigmoid prob of foreground class. (1, 1, H, W)."""
    if use_coarse_mask:
        coarse_normalized = (coarse - 0.5) / 0.5
        x = torch.cat([img, coarse_normalized[:, None, ...]], dim=1)
    else:
        x = img
    # inference() (parent EncoderDecoder) ALREADY applies F.softmax and returns
    # per-class PROBABILITIES of shape (1, num_classes, H, W) -- not logits.
    # Do NOT softmax again: a second softmax squashes probs toward 0.5 (a true
    # 0.9 -> ~0.69) and corrupts the multi-view TTA average. Take the
    # foreground channel directly.
    prob_map = model.module.inference(x, [img_meta], rescale=False)
    if prob_map.shape[1] == 1:
        prob = prob_map                     # single-channel (already a prob)
    else:
        prob = prob_map[:, 1:2]             # foreground-class probability
    return prob


@torch.no_grad()
def tta_predict(model, img_t, coarse_t, img_meta, use_coarse_mask):
    p_o = single_view_prob(model, img_t, coarse_t, img_meta, use_coarse_mask)

    img_h, coarse_h = flip_t(img_t, 'h'), flip_t(coarse_t, 'h')
    p_h = single_view_prob(model, img_h, coarse_h, img_meta, use_coarse_mask)
    p_h = flip_t(p_h, 'h')

    img_v, coarse_v = flip_t(img_t, 'v'), flip_t(coarse_t, 'v')
    p_v = single_view_prob(model, img_v, coarse_v, img_meta, use_coarse_mask)
    p_v = flip_t(p_v, 'v')

    img_hv, coarse_hv = flip_t(img_t, 'hv'), flip_t(coarse_t, 'hv')
    p_hv = single_view_prob(model, img_hv, coarse_hv, img_meta, use_coarse_mask)
    p_hv = flip_t(p_hv, 'hv')

    return (p_o + p_h + p_v + p_hv) / 4.0


@torch.no_grad()
def no_tta_predict(model, img_t, coarse_t, img_meta, use_coarse_mask):
    return single_view_prob(model, img_t, coarse_t, img_meta, use_coarse_mask)


def main():
    args = parse_args()
    crop_size = tuple(args.crop_size)

    # Build model and load weights
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    use_coarse_mask = getattr(model.module, 'use_coarse_mask', True)
    print(f"Config:              {args.config}")
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Crop size:           {crop_size}")
    print(f"use_coarse_mask:     {use_coarse_mask}")
    print(f"TTA enabled:         {not args.no_tta}")
    print(f"Output root:         {args.out_root}")

    datasets = args.datasets if args.datasets else DATASETS

    for ds in datasets:
        img_dir = osp.join(args.test_root, ds, 'images')
        coarse_dir = osp.join(args.test_root, ds, 'pranet_masks')
        out_dir = osp.join(args.out_root, ds)

        if not osp.isdir(img_dir) or not osp.isdir(coarse_dir):
            print(f"SKIP {ds}: missing {img_dir} or {coarse_dir}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        filenames = sorted(os.listdir(img_dir))
        print(f"\n=== {ds}: {len(filenames)} images ===")

        for fname in tqdm(filenames):
            img_path = osp.join(img_dir, fname)
            coarse_path = osp.join(coarse_dir, fname)
            if not osp.exists(coarse_path):
                print(f"  Missing coarse mask for {fname}, skipping")
                continue

            img_bgr = cv2.imread(img_path)
            coarse_gray = cv2.imread(coarse_path, cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or coarse_gray is None:
                print(f"  Failed to load {fname}, skipping")
                continue

            orig_H, orig_W = img_bgr.shape[:2]
            img_t, coarse_t = preprocess(img_bgr, coarse_gray, crop_size)
            resized_H, resized_W = img_t.shape[-2], img_t.shape[-1]
            img_meta = {
                'img_shape':   (resized_H, resized_W, 3),
                'ori_shape':   (orig_H, orig_W, 3),
                'pad_shape':   (resized_H, resized_W, 3),
                'scale_factor': 1.0,
                'flip': False,
                'flip_direction': None,
            }

            if args.no_tta:
                prob = no_tta_predict(model, img_t, coarse_t, img_meta,
                                      use_coarse_mask)
            else:
                prob = tta_predict(model, img_t, coarse_t, img_meta,
                                   use_coarse_mask)

            mask_bin = (prob.squeeze() > 0.5).cpu().numpy().astype(np.uint8) * 255
            mask_full = cv2.resize(mask_bin, (orig_W, orig_H),
                                   interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(osp.join(out_dir, fname), mask_full)

    print(f"\nDone. Refined masks saved to {args.out_root}")


if __name__ == '__main__':
    main()
