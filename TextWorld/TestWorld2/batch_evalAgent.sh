
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash batch_evalAgent.sh <MODEL_NAME> <EPISODE>"
    echo "Example: bash batch_evalAgent.sh deepseek-v3 EP1"
    exit 1
fi

MODEL_NAME="$1"
EPISODE="$2"

LOG_DIR="$(dirname "$0")/bashlog"
mkdir -p $LOG_DIR

CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
TARGET_LOG_FILE="$LOG_DIR/evalAgent_${MODEL_NAME}_${EPISODE}_${SLURM_JOB_ID}_${CURRENT_TIME}.log"

echo "Job started at $CURRENT_TIME. Log redirected to: $TARGET_LOG_FILE" >&2
python "$(dirname "$0")/evalAgent.py" -m $MODEL_NAME -e $EPISODE > $TARGET_LOG_FILE 2>&1
