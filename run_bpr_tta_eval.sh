#!/bin/bash
set -e

CONFIG=configs/bpr/hrnet48_256.py
CHECKPOINT=work_dirs/hrnet48_256/latest.pth
OUTPUT_BASE=results_cl/BPR_TTA

DATA_ROOT_BASE=~/projects/BPR/dataset/TestDataset_bpr_layout
DATASETS=("Kvasir" "CVC-ClinicDB" "CVC-ColonDB" "CVC-300" "ETIS-LaribPolypDB")

mkdir -p "$OUTPUT_BASE"

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Running BPR + TTA on $DS"
    echo "=========================================="

    export DATA_ROOT="$DATA_ROOT_BASE/$DS"

    OUT_DIR="$OUTPUT_BASE/$DS"
    mkdir -p "$OUT_DIR"

    python tools/test.py "$CONFIG" "$CHECKPOINT" \
        --show-dir "$OUT_DIR" \
        --options data.test.data_root="$DATA_ROOT"

    echo "Predictions saved to $OUT_DIR"
done

echo ""
echo "All BPR+TTA predictions saved under $OUTPUT_BASE"
echo "Now run your existing eval script on this folder."
