from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.io import load_yaml
from src.models.build import build_model
from src.data.flickr8k import Flickr8kDataset
from src.data.transforms import build_transform
from src.data.collate import collate_flickr8k


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    ap.add_argument("--save_json", default=None)
    ap.add_argument("--prompt", default="Describe the image in one sentence.")
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_model(ckpt_path: str, cfg: Optional[dict], device: str):
    if cfg is None:
        cfg = load_yaml(Path(__file__).resolve().parents[2] / "configs" / "flickr8k_caption.yaml")
    model = build_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def caption_single(model, image_path: str, prompt: str, num_beams: int, max_new_tokens: int, device: str):
    tfm = build_transform(train=False)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    return model.generate_caption(x, prompt_text=prompt, num_beams=num_beams, max_new_tokens=max_new_tokens)[0]


@torch.no_grad()
def caption_split(model, data_root: str, split: str, prompt: str, num_beams: int, max_new_tokens: int, device: str, save_json: str, batch_size: int):
    ds = Flickr8kDataset(data_root, split=split, transform=build_transform(train=False))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.startswith("cuda"),
        collate_fn=lambda b: collate_flickr8k(b, split=split),
    )
    out = {}
    for batch in tqdm(loader, desc=f"infer-{split}"):
        images = batch["images"].to(device)
        names = batch["image_names"]
        preds = model.generate_caption(images, prompt_text=prompt, num_beams=num_beams, max_new_tokens=max_new_tokens)
        for n, p in zip(names, preds):
            out[n] = p
    Path(save_json).parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    cfg = load_yaml(args.config) if args.config else None
    model = load_model(args.ckpt, cfg, device)

    if args.image:
        print(caption_single(model, args.image, args.prompt, args.num_beams, args.max_new_tokens, device))
        return

    if not (args.data_root and args.split and args.save_json):
        raise SystemExit("Provide --image OR (--data_root --split --save_json)")

    caption_split(model, args.data_root, args.split, args.prompt, args.num_beams, args.max_new_tokens, device, args.save_json, args.batch_size)


if __name__ == "__main__":
    main()
