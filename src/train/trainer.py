from __future__ import annotations
import sys, os 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.flickr8k import Flickr8kDataset
from src.data.transforms import build_transform
from src.data.collate import collate_flickr8k
from src.models.build import build_model
from src.utils.io import load_yaml, ensure_dir
from src.utils.seed import set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def train_one_epoch(model, loader, optimizer, scaler, cfg, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(pbar, start=1):
        if batch is None:
            continue
        images = batch["images"].to(device, non_blocking=True)
        captions = batch["captions"]
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=bool(cfg["train"].get("fp16", True)) and device.startswith("cuda")):
            out = model(images=images, target_text=captions, prompt_text=cfg["train"].get("prompt"), max_text_len=cfg["train"].get("max_text_len", 64))
            loss = out.loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        if step % int(cfg.get("logging", {}).get("log_every", 50)) == 0:
            pbar.set_postfix(loss=running_loss / step)

    return running_loss / max(1, len(loader))


@torch.no_grad()
def eval_one_epoch(model, loader, cfg, device):
    model.eval()
    running_loss = 0.0
    pbar = tqdm(loader, desc="val", leave=False)
    for step, batch in enumerate(pbar, start=1):
        if batch is None:
           continue
        images = batch["images"].to(device, non_blocking=True)
        # For val we have list-of-lists; pick first ref for loss
        captions = [refs[0] if isinstance(refs, list) and len(refs) > 0 else "" for refs in batch["captions"]]
        out = model(images=images, target_text=captions, prompt_text=cfg["train"].get("prompt"), max_text_len=cfg["train"].get("max_text_len", 64))
        loss = out.loss
        running_loss += float(loss.item())
        pbar.set_postfix(loss=running_loss / step)
    return running_loss / max(1, len(loader))


def save_checkpoint(model, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out_path)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "checkpoints")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # Data
    train_ds = Flickr8kDataset(args.data_root, split="train", transform=build_transform(train=True))
    val_ds = Flickr8kDataset(args.data_root, split="val", transform=build_transform(train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.startswith("cuda"),
        collate_fn=lambda b: collate_flickr8k(b, split="train"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("val", {}).get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg.get("val", {}).get("num_workers", 4)),
        pin_memory=device.startswith("cuda"),
        collate_fn=lambda b: collate_flickr8k(b, split="val"),
    )

    # Model
    model = build_model(cfg).to(device)

    # Optimizer: ONLY params that require grad (Q-Former + proj)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.05)),
    )

    use_fp16 = bool(cfg["train"].get("fp16", True)) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    best_val = 1e9
    history = []

    
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device)
        val_loss = eval_one_epoch(model, val_loader, cfg, device)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row))

        # Save last
        save_checkpoint(model, out_dir / "checkpoints" / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, out_dir / "checkpoints" / "best.pt")

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
