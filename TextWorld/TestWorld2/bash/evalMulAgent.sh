#!/usr/bin/env bash

set -e

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <SEED_ID> <MODEL> <SERVER_URL> [ --resume=True | --resume=False ]"
    echo "Example: $0 EP1-023 qwen3-VL-plus http://gpu09:8021/generate --resume=False"
    exit 1
fi

ULX_PATH_DIR="./TextWorld/TestWorld2/results/ulxEnv/$1"
ULX_PATH_DIR_EP2="./TextWorld/TestWorld2/results/EP2-Result/ulxEnv/$1"
ULX_PATH_DIR_EP3="./TextWorld/TestWorld2/results/EP3-Result/ulxEnv/$1"

if [[ "$1" == EP2* ]]; then
    ULX_PATH_DIR="$ULX_PATH_DIR_EP2"
elif [[ "$1" == EP3* ]]; then
    ULX_PATH_DIR="$ULX_PATH_DIR_EP3"
fi

LAST_ULX_FILE=$(find "$ULX_PATH_DIR" -maxdepth 1 -type f -name "*.ulx" -printf '%T@ %p\n' | sort -n -r | head -n 1 | cut -d' ' -f2-)

echo "Evaluating agent in environment: $LAST_ULX_FILE with model: $2"

python ./TextWorld/TestWorld2/testEnv/Mulagent.py "$LAST_ULX_FILE" "$2" "$3" "${@:4}"