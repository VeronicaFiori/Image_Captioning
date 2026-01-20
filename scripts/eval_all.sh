#!/usr/bin/env bash
set -e
CONFIG=${1:-configs/flickr8k_caption.yaml}
DATA_ROOT=${2:-data/flickr8k}
CKPT=${3:-outputs/flickr8k_stageA/checkpoints/best.pt}
OUT_DIR=${4:-outputs/flickr8k_stageA}
mkdir -p "$OUT_DIR"
python -m src.infer.generate --ckpt "$CKPT" --config "$CONFIG" --data_root "$DATA_ROOT" --split test --save_json "$OUT_DIR/preds_test.json"
python -m src.eval.caption_metrics --config "$CONFIG" --data_root "$DATA_ROOT" --preds "$OUT_DIR/preds_test.json" --split test
