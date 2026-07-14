#!/bin/bash
# 4-view flip TTA evaluation for BPR on the 5 polyp test sets.
#
# This delegates to tools/test_bpr_tta_manual.py, which performs the ACTUAL
# test-time augmentation (identity + H-flip + V-flip + HV-flip, each un-flipped
# and averaged in probability space). The previous version of this script
# called tools/test.py with configs/bpr/hrnet48_256.py, whose test pipeline has
# flip=False -> it performed NO TTA (and --show-dir only writes visualizations,
# not the binary masks needed for Dice/IoU scoring).
#
# The manual tool loops over the datasets internally and expects, per dataset:
#   <TEST_ROOT>/<dataset>/images/         RGB inputs
#   <TEST_ROOT>/<dataset>/pranet_masks/   coarse (PraNet) masks to refine
# and writes refined binary PNGs to <OUT_ROOT>/<dataset>/.
#
# Override any of these via environment variables.
set -e

CONFIG=${CONFIG:-configs/bpr/poly_hrnet48_256_fct.py}
CHECKPOINT=${CHECKPOINT:-work_dirs/poly_hrnet48_256_fct/latest.pth}
TEST_ROOT=${TEST_ROOT:-dataset/TestDataset_bpr_layout}
OUT_ROOT=${OUT_ROOT:-results_cl/BPR_TTA}

echo "=========================================="
echo "  BPR 4-view TTA"
echo "  config     : $CONFIG"
echo "  checkpoint : $CHECKPOINT"
echo "  test root  : $TEST_ROOT"
echo "  out root   : $OUT_ROOT"
echo "=========================================="

python tools/test_bpr_tta_manual.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --test-root "$TEST_ROOT" \
    --out-root "$OUT_ROOT"

echo ""
echo "TTA-refined masks saved under $OUT_ROOT (per-dataset subfolders)."
echo "Now run your polyp eval script (Dice/IoU vs GT) on that folder."
echo "Tip: pass --no-tta to tools/test_bpr_tta_manual.py to compare against the"
echo "single-view (no-TTA) baseline."
