#!/bin/bash
cd "$(dirname "$0")"

image_path=$1
prompt=$2

python generate.py \
  --task i2v-A14B \
  --size 832*480 \
  --ckpt_dir Wan-AI/Wan2.2-I2V-A14B \
  --offload_model False \
  --convert_model_dtype \
  --frame_num 49 \
  --sample_steps 20 \
  --base_seed 6689 \
  --image ${image_path} \
  --prompt "${prompt}"