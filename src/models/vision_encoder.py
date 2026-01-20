from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import open_clip


@dataclass
class VisionCfg:
    name: str
    pretrained: str
    freeze: bool = True


class OpenCLIPVisionEncoder(nn.Module):
    def __init__(self, cfg: VisionCfg):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(cfg.name, pretrained=cfg.pretrained)
        self.visual = model.visual
        self.out_dim = getattr(self.visual, "output_dim", None)
        if self.out_dim is None:
            # fallback for some OpenCLIP backbones
            self.out_dim = self.visual.proj.shape[1] if hasattr(self.visual, "proj") else 768

        if cfg.freeze:
            for p in self.visual.parameters():
                p.requires_grad = False
            self.visual.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-image embedding (B, D)."""
        feats = self.visual(images)
        return feats
