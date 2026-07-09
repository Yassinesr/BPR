#!/bin/bash
set -e

SRC=~/projects/BPR/dataset/TestDataset
DST=~/projects/BPR/dataset/TestDataset_bpr_layout

DATASETS=("Kvasir" "CVC-ClinicDB" "CVC-ColonDB" "CVC-300" "ETIS-LaribPolypDB")

for DS in "${DATASETS[@]}"; do
    echo "Setting up $DS..."
    mkdir -p "$DST/$DS/img_dir/val"
    mkdir -p "$DST/$DS/mask_dir/val"
    mkdir -p "$DST/$DS/ann_dir/val"

    # Clear any previous symlinks
    rm -f "$DST/$DS/img_dir/val"/* 2>/dev/null
    rm -f "$DST/$DS/mask_dir/val"/* 2>/dev/null
    rm -f "$DST/$DS/ann_dir/val"/* 2>/dev/null

    # Symlink images, coarse masks (PraNet), and ground truth
    ln -sf "$SRC/$DS/images/"*           "$DST/$DS/img_dir/val/"
    ln -sf "$SRC/$DS/pranet_masks/"*     "$DST/$DS/mask_dir/val/"
    ln -sf "$SRC/$DS/gts/"*              "$DST/$DS/ann_dir/val/"

    echo "  Images:        $(ls "$DST/$DS/img_dir/val/" | wc -l) files"
    echo "  Coarse masks:  $(ls "$DST/$DS/mask_dir/val/" | wc -l) files"
    echo "  Ground truth:  $(ls "$DST/$DS/ann_dir/val/" | wc -l) files"
done

echo "Done. BPR-compatible test layout created at: $DST"
