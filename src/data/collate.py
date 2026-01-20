from __future__ import annotations

import random
from typing import Any, Dict, List

import torch


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
