#!/bin/bash
cd "$(dirname "$0")"

prompt=$1

python generate.py  \
    --task t2v-A14B \
    --size 832*480 \
    --ckpt_dir Wan-AI/Wan2.2-T2V-A14B \
    --offload_model False \
    --frame_num 49 \
    --convert_model_dtype \
    --sample_steps 40\
    --prompt "${prompt}"