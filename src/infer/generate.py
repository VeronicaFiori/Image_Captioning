# src/infer/generate.py
from __future__ import annotations

# IMPORTANTISSIMO: env PRIMA di importare transformers/open_clip
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.models.build import build_model
from src.utils.io import load_yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/flickr8k_caption.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Describe the image. Use only visible details.")
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_cuda", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()

    # device
    device = "cpu" if args.no_cuda else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # prints "a prova di colab": flush=True
    print(">> starting generate.py", flush=True)
    print(f">> device = {device}", flush=True)
    print(f">> ckpt = {args.ckpt}", flush=True)
    print(f">> image = {args.image}", flush=True)

    # load cfg
    cfg = load_yaml(args.config)
    print(">> config loaded", flush=True)

    # build model
    print(">> building model...", flush=True)
    model = build_model(cfg)
    model.eval()
    model.to(device)
    print(">> model built", flush=True)

    # load ckpt (strict!)
    print(">> loading checkpoint...", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f">> ckpt loaded. missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if len(missing) > 0:
        print(">> missing keys (first 20):", missing[:20], flush=True)
    if len(unexpected) > 0:
        print(">> unexpected keys (first 20):", unexpected[:20], flush=True)

    # load image
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    print(">> loading image...", flush=True)
    image = Image.open(str(img_path)).convert("RGB")

    # IMPORTANT: usa lo stesso preprocess del vision encoder
    # build_model(cfg) nel mio progetto espone model.preprocess se presente
    if hasattr(model, "preprocess") and model.preprocess is not None:
        x = model.preprocess(image).unsqueeze(0)
    else:
        # fallback: resize semplice (meglio di niente, ma consiglio avere preprocess vero)
        image = image.resize((224, 224))
        x = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0)

    x = x.to(device)

    # generate
    print(">> running generation...", flush=True)
    with torch.no_grad():
        out_text = model.generate(
            images=x,
            prompt_text=args.prompt,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )

    print("\n=== CAPTION ===", flush=True)
    print(out_text, flush=True)


if __name__ == "__main__":
    main()
