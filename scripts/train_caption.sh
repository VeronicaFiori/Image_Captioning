#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/flickr8k_caption.yaml}
DATA_ROOT=${2:-data/flickr8k}
OUT_DIR=${3:-outputs/flickr8k_stageA}

python -m src.train.trainer \
  --config "$CONFIG" \
  --data_root "$DATA_ROOT" \
  --out_dir "$OUT_DIR"
