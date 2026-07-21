#!/bin/bash
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=1,4

export MASTER_PORT=29500

LOG_DIR="./serverlog"

mkdir -p $LOG_DIR

CURRENT_TIME=$(date +%Y%m%d_%H%M%S)

TARGET_LOG_FILE="$LOG_DIR/generate_i2v_server_multigpu_${CURRENT_TIME}.log"

echo "GPU devices: $CUDA_VISIBLE_DEVICES"

torchrun --nproc_per_node=2 --nnodes=1 --master_port=$MASTER_PORT \
    ./generate_i2v_server_multigpu.py \
    2>&1 | tee $TARGET_LOG_FILE
