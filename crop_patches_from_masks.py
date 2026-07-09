import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from mmcv.ops import roi_align
from mmcv.ops.nms import nms

# ========================
# CONFIGURATION
# ========================
root_dir = "./TrainDataset"       
pred_dir = "./TrainDataset"           
# NOTE about prediction inputs:
# - This script currently constructs `pred_path` from `root_dir` as
#     pred_path = os.path.join(root_dir, "predtrain")
#   That means predictions are expected in the `predtrain` subfolder of
#   `root_dir` by default.
# - If you'd like to use a different folder for predictions, either:
#     1) set `pred_dir` to the folder you want and then replace the
#        `pred_path = os.path.join(root_dir, "predtrain")` line below with
#        `pred_path = pred_dir`
#     2) or update the `pred_path` assignment directly.
# - Currently `pred_dir` is defined but not used; the script still uses
#   the `predtrain` subfolder under `root_dir` unless you change `pred_path`.
output_dir = root_dir.rstrip("/\\") + "_patches"

boundary_width = 3
# NMS IoU threshold used to suppress overlapping patch proposals
iou_thresh = 0.55
patch_size = 64
out_size = 64

# ========================
# Helper functions
# ========================
def _force_move_back(sdets, H, W, patch_size):
    # Make sure patch boxes stay inside the image.
    # "sdets" are boxes like [x1,y1,x2,y2] (may also include a score column).
    # If a box would go outside the image, move it so the whole patch fits.
    s = sdets[:, 0] < 0
    sdets[s, 0] = 0
    sdets[s, 2] = patch_size

    s = sdets[:, 1] < 0
    sdets[s, 1] = 0
    sdets[s, 3] = patch_size

    s = sdets[:, 2] >= W
    sdets[s, 0] = W - 1 - patch_size
    sdets[s, 2] = W - 1

    s = sdets[:, 3] >= H
    sdets[s, 1] = H - 1 - patch_size
    sdets[s, 3] = H - 1
    return sdets

def _to_rois(boxes):
    # Prepare boxes for roi_align. roi_align expects each box to start with a
    # batch index like [batch_idx, x1, y1, x2, y2]. We only have one image,
    # so batch index is 0 for every box.
    idx = torch.zeros((boxes.size(0), 1), device=boxes.device)
    return torch.cat([idx, boxes], dim=1)

def find_float_boundary(maskdt, width):
    # Turn hard masks (0 or 1) into a "soft" boundary map.
    # Think: where the mask changes from 0->1 or 1->0 is the boundary.
    # This returns a float map (same shape) that is larger near edges.
    N, H, W = maskdt.shape
    maskdt = maskdt.view(N, 1, H, W)
    boundary_finder = maskdt.new_ones((1, 1, width, width))
    boundary_mask = F.conv2d(maskdt, boundary_finder, stride=1, padding=width//2)
    # bml measures distance from full patch (interior), bms from zero (exterior)
    bml = torch.abs(boundary_mask - width*width)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width*width/2)
    return fbmask.view(N, H, W)

def get_dets(fbmask, patch_size, iou_thresh):
    # From the soft boundary map, create candidate square patches centered on
    # boundary pixels. Use NMS to remove overlapping boxes. The result is a
    # set of boxes with a score (how strong the boundary was there).
    ys, xs = torch.nonzero(fbmask, as_tuple=True)
    scores = fbmask[ys, xs]
    ys = ys.float()
    xs = xs.float()
    dets = torch.stack([xs - patch_size//2, ys - patch_size//2,
                        xs + patch_size//2, ys + patch_size//2, scores]).T
    _, inds = nms(dets[:, :4].contiguous(), dets[:, 4].contiguous(), iou_thresh)
    sdets = dets[inds]
    H, W = fbmask.shape
    return _force_move_back(sdets, H, W, patch_size)


def boxes_to_clockwise_corners(boxes):
    """
    Convert axis-aligned boxes [x1,y1,x2,y2] into corner coordinates
    ordered clockwise starting at top-left: (x1,y1),(x2,y1),(x2,y2),(x1,y2).

    Input: boxes tensor shape (N,4)
    Output: tensor shape (N,8) with ordering [x1,y1,x2,y1,x2,y2,x1,y2]
    """
    if boxes.numel() == 0:
        return boxes.new_empty((0, 8))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    corners = torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], dim=1)
    return corners

# ========================
# Split function
# ========================
def split(img, maskdts, pred_t, boundary_width=3, iou_thresh=0.55, patch_size=64, out_size=64):
    """
    Extract patches around mask boundaries.

    Inputs:
      - img: HxWx3 numpy image (RGB)
      - maskdts: tensor (K,H,W) of binary masks (float 0/1)
      - pred_t: tensor (K,H,W) of predicted masks (float 0/1) or similar
    Returns:
      - detss: list of tensors of detections for each instance
      - img_patches: tensor (N, C, patch_size, patch_size)
      - dt_patches: tensor (N, 1, patch_size, patch_size) (ground-truth crops)
      - pred_patches: tensor (N, 1, patch_size, patch_size) (prediction crops)

        Notes:
            - If no detections are found returns empty list and empty tensors.
    """
    # 1) Convert each predicted instance mask into a soft boundary map.
    #    We use predicted masks (pred_t) to generate proposals as requested.
    fbmasks = find_float_boundary(pred_t, boundary_width)
    # 2) For each instance, find patch boxes around strong boundary pixels.
    detss = []
    for i in range(fbmasks.size(0)):
        # dets shape: (M,5) -> keep first 4 cols (x1,y1,x2,y2)
        dets = get_dets(fbmasks[i], patch_size, iou_thresh=iou_thresh)[:, :4]
        detss.append(dets)

    # If no boundary boxes were found, return empty tensors so the caller can
    # skip this image without errors.
    if len(detss) == 0 or all([d.size(0) == 0 for d in detss]):
        return [], torch.empty(0), torch.empty(0), torch.empty(0)

    all_dets = torch.cat(detss, dim=0)

    # For each detection compute the 4 corner coordinates in clockwise order.
    # det_corners_list mirrors detss (list per instance), where each element
    # is a tensor shape (num_boxes, 8) containing [x1,y1,x2,y1,x2,y2,x1,y2].
    det_corners_list = [boxes_to_clockwise_corners(d) for d in detss]

    # 3) Crop RGB image patches using roi_align. roi_align needs tensors in
    # (B,C,H,W) format. We have one image so batch size is 1.
    img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).float().contiguous()
    img_patches = roi_align(img_t, _to_rois(all_dets), patch_size)

    _detss = [torch.cat([i * _.new_ones((_.size(0), 1)), _], dim=1) for i, _ in enumerate(detss)]
    _detss = torch.cat(_detss)
    # 4) Crop corresponding ground-truth and predicted mask patches. These are
    # single-channel (1,H,W) so we add a channel dim before roi_align.
    dt_patches = roi_align(maskdts[:, None, :, :], _detss, patch_size)
    pred_patches = roi_align(pred_t[:, None, :, :], _detss, patch_size)

    # Return values (easy terms):
    #  - detss: list of box tensors, one list per ground-truth instance.
    #  - img_patches: RGB patch images ready for saving or further model input.
    #  - dt_patches: the corresponding ground-truth mask crops (useful for
    #                training or evaluation).
    #  - pred_patches: the predicted mask crops (what the coarse model predicted).
    # Return the original detss plus the clockwise corner coordinates as a
    # supplementary result (det_corners_list). The function now returns:
    # detss, img_patches, dt_patches, pred_patches, det_corners_list
    return detss, img_patches, dt_patches, pred_patches, det_corners_list

# ========================
# Main loop
# ========================

# Input paths
img_path = os.path.join(root_dir, "images")
gt_path = os.path.join(root_dir, "masks")
pred_path = os.path.join(root_dir, "predtrain")

file_list = [f for f in os.listdir(gt_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Output paths
out_img_dir = os.path.join(output_dir, "img_dir", "train")
out_gt_dir = os.path.join(output_dir, "ann_dir", "train")
out_pred_dir = os.path.join(output_dir, "mask_dir", "train")
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_gt_dir, exist_ok=True)
os.makedirs(out_pred_dir, exist_ok=True)

for filename in file_list:
    base = os.path.splitext(filename)[0]
    img_file = os.path.join(img_path, filename)
    gt_file = os.path.join(gt_path, filename)
    pred_file = os.path.join(pred_path, filename)

    if not os.path.exists(img_file) or not os.path.exists(pred_file):
        print(f"Skipping {filename}: missing image or prediction file.")
        continue

    img = cv2.imread(img_file)[:, :, ::-1].copy()  # BGR -> RGB
    gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)

    gt_t = torch.from_numpy(gt / 255.0).float().unsqueeze(0)      # (1,H,W)
    pred_t = torch.from_numpy(pred / 255.0).float().unsqueeze(0)

    detss, img_patches, gt_patches, pred_patches, det_corners_list = split(
        img, gt_t, pred_t, boundary_width, iou_thresh, patch_size, out_size
    )

    print(filename)
    print('img_patches.shape =', img_patches.shape)
    # Print corner coordinates for each instance. Each element in
    # det_corners_list is a tensor shape (num_boxes, 8) with ordering:
    # [x1,y1,x2,y1,x2,y2,x1,y2] (clockwise corners).
    for inst_idx, corners in enumerate(det_corners_list):
        print(f' instance {inst_idx} corners:')
        print(corners)
    if img_patches.numel() == 0:
        continue
    break
    # img_patches_np = img_patches.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    # for idx in range(img_patches_np.shape[0]):
    #     print(os.path.join(out_img_dir, f"{base}_patch_{idx:04d}.png"))
    #     cv2.imwrite(
    #         os.path.join(out_img_dir, f"{base}_patch_{idx:04d}.png"),
    #         img_patches_np[idx][:, :, ::-1]  # RGB -> BGR
    #     )

    #     gt_patch_np = (gt_patches[idx].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    #     cv2.imwrite(
    #         os.path.join(out_gt_dir, f"{base}_patch_{idx:04d}.png"),
    #         gt_patch_np
    #     )

    #     pred_patch_np = (pred_patches[idx].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    #     cv2.imwrite(
    #         os.path.join(out_pred_dir, f"{base}_patch_{idx:04d}.png"),
    #         pred_patch_np
    #     )

print("\n All images processed using ROI-based patch extraction.")
