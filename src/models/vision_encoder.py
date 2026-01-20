from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import open_clip


@dataclass
class VisionCfg:
    name: str
    pretrained: str
    freeze: bool = True


class OpenCLIPVisionEncoder(nn.Module):
    """
    OpenCLIP vision encoder that returns token-level features when possible.

    - Stores the OpenCLIP preprocess transform (must be used for correct normalization).
    - Returns (B, N, D) where N is number of visual tokens (patches + cls if present).
      If token extraction is not available, returns (B, 1, D) using global embedding.
    """
    def __init__(self, cfg: VisionCfg):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.name, pretrained=cfg.pretrained
        )
        self.visual = model.visual
        self.preprocess = preprocess  # <-- IMPORTANTISSIMO

        # infer output dim robustly
        self.out_dim = getattr(self.visual, "output_dim", None)
        if self.out_dim is None:
            if hasattr(self.visual, "proj") and self.visual.proj is not None:
                # proj can be (D, D) or similar
                self.out_dim = self.visual.proj.shape[1]
            else:
                self.out_dim = 768

        if cfg.freeze:
            for p in self.visual.parameters():
                p.requires_grad = False
            self.visual.eval()

    def _forward_tokens(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Try to extract token-level features from OpenCLIP visual backbone.
        Different OpenCLIP backbones expose different APIs; we try a few common ones.
        Returns:
          tokens: (B, N, D) or None if not supported.
        """
        # Many OpenCLIP ViT visual backbones have `forward_features`
        if hasattr(self.visual, "forward_features"):
            feats = self.visual.forward_features(images)

            # Some implementations return dicts
            if isinstance(feats, dict):
                # common keys: 'x', 'last_hidden_state', 'tokens'
                for k in ("tokens", "last_hidden_state", "x"):
                    if k in feats and isinstance(feats[k], torch.Tensor):
                        feats = feats[k]
                        break

            if isinstance(feats, torch.Tensor):
                # Could be (B, N, D) already, or (B, D)
                if feats.ndim == 3:
                    return feats
                if feats.ndim == 2:
                    return feats.unsqueeze(1)  # (B, 1, D)

        # Fallback: no token method found
        return None

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Return visual features as tokens:
          (B, N, D) preferred; fallback (B, 1, D).
        """
        tokens = self._forward_tokens(images)
        if tokens is not None:
            return tokens

        # ultimate fallback: global embedding
        feats = self.visual(images)  # often (B, D)
        if feats.ndim == 2:
            feats = feats.unsqueeze(1)  # (B, 1, D)
        return feats
