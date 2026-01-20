from __future__ import annotations

import random
from typing import Any, Dict, List

import torch

def collate_flickr8k(batch, split="train"):
    # 1) rimuovi i sample None (immagini mancanti/corrotte)
    batch = [b for b in batch if b is not None]

    # 2) se il batch è vuoto, ritorna None (il trainer deve skipparlo)
    if len(batch) == 0:
        return None

    images = torch.stack([b["image"] for b in batch], dim=0)

    if split == "train":
        captions = [b["caption"] for b in batch]  # stringhe
    else:
        # se nel val/test tu ritorni liste di refs, metti qui la tua logica
        captions = [b["caption"] for b in batch]

    return {"images": images, "captions": captions}

"""
def collate_flickr8k(batch: List[Dict[str, Any]], split: str = "train") -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    names = [b["image_name"] for b in batch]

    if split == "train":
        caps = [random.choice(b["captions"]) for b in batch]
    else:
        # Keep all references for evaluation
        caps = [b["captions"] for b in batch]

    return {
        "images": images,
        "image_names": names,
        "captions": caps,
    }
"""