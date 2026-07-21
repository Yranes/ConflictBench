#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: bash batch_evalMulAgent.sh <MODEL_NAME> <SERVER_URL> <EPISODE>"
    echo "Example: bash batch_evalMulAgent.sh qwen3-vl-plus http://gpu09:8021/generate EP1"
    exit 1
fi

MODEL_NAME="$1"
SERVER_URL="$2"
EPISODE="$3"

LOG_DIR="$(dirname "$0")/bashlog"
mkdir -p $LOG_DIR

CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
TARGET_LOG_FILE="$LOG_DIR/evalMulAgent_${MODEL_NAME}_${EPISODE}_${SLURM_JOB_ID}_${CURRENT_TIME}.log"

echo "Job started at $CURRENT_TIME. Log redirected to: $TARGET_LOG_FILE" >&2
python "$(dirname "$0")/evalMulAgent.py" -m $MODEL_NAME -s $SERVER_URL -e $EPISODE > $TARGET_LOG_FILE 2>&1
