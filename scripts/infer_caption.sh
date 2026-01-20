#!/usr/bin/env bash
set -e

CKPT=$1
IMAGE=$2
PROMPT=${3:-"Describe the image in one sentence."}

python -m src.infer.generate \
  --ckpt "$CKPT" \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --num_beams 5
