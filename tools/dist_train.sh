#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG

#--launcher pytorch ${@:3}
#
#DATA_ROOT=../UACANet-main/dataset/pranet-traindataset/PatchesDataset-IOU-fusion \
#bash tools/dist_train.sh \
#  configs/bpr/poly_hrnet48_256.py \
#  1

