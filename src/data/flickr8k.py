from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


def parse_captions(token_file: Path) -> Dict[str, List[str]]:
    """Parse Flickr8k.token.txt

    Each line: <image>#<idx>\t<caption>
    """
    caps: Dict[str, List[str]] = {}
    with token_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Sometimes separator may be \t
            if "\t" in line:
                key, cap = line.split("\t", 1)
            else:
                key, cap = line.split(" ", 1)
            img = key.split("#", 1)[0]
            caps.setdefault(img, []).append(cap.strip())
    return caps


def read_image_list(list_file: Path) -> List[str]:
    with list_file.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


@dataclass
class Flickr8kPaths:
    root: Path

    @property
    def images_dir(self) -> Path:
        return self.root / "Images"

    @property
    def token_file(self) -> Path:
        return self.root / "Flickr8k.token.txt"

    @property
    def train_list(self) -> Path:
        return self.root / "Flickr_8k.trainImages.txt"

    @property
    def val_list(self) -> Path:
        return self.root / "Flickr_8k.devImages.txt"

    @property
    def test_list(self) -> Path:
        return self.root / "Flickr_8k.testImages.txt"


class Flickr8kDataset(Dataset):
    def __init__(self, root: str | Path, split: str, transform=None):
        self.paths = Flickr8kPaths(Path(root))
        self.captions = parse_captions(self.paths.token_file)

        if split == "train":
            self.image_names = read_image_list(self.paths.train_list)
        elif split in ("val", "dev"):
            self.image_names = read_image_list(self.paths.val_list)
        elif split == "test":
            self.image_names = read_image_list(self.paths.test_list)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        img_path = self.paths.images_dir / name
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        caps = self.captions.get(name, [])
        if len(caps) == 0:
            # Flickr8k should always have captions, but be robust
            caps = [""]

        return {
            "image": image,
            "image_name": name,
            "captions": caps,
        }
