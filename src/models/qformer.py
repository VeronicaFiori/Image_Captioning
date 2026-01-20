from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class QFormerCfg:
    num_query_tokens: int = 32
    d_q: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_q: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_q, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_q)
        self.ff = nn.Sequential(
            nn.Linear(d_q, 4 * d_q),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_q, d_q),
        )
        self.ln2 = nn.LayerNorm(d_q)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, Nq, d)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        q = self.ln1(q + self.drop(attn_out))
        ff_out = self.ff(q)
        q = self.ln2(q + self.drop(ff_out))
        return q


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_q: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_q, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_q)
        self.ff = nn.Sequential(
            nn.Linear(d_q, 4 * d_q),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_q, d_q),
        )
        self.ln2 = nn.LayerNorm(d_q)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(query=x, key=x, value=x, need_weights=False)
        x = self.ln1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.drop(ff_out))
        return x


class QFormer(nn.Module):
    """A minimal BLIP2-like Q-Former.

    We keep it simple: learnable query tokens (Nq) attend to a single image embedding.
    If you want per-patch tokens, you can swap the vision encoder to output patches.
    """

    def __init__(self, cfg: QFormerCfg, vision_dim: int):
        super().__init__()
        self.cfg = cfg
        self.query_tokens = nn.Parameter(torch.randn(1, cfg.num_query_tokens, cfg.d_q) * 0.02)
        self.kv_proj = nn.Linear(vision_dim, cfg.d_q)

        layers = []
        for _ in range(cfg.num_layers):
            layers.append(CrossAttentionBlock(cfg.d_q, cfg.num_heads, cfg.dropout))
            layers.append(SelfAttentionBlock(cfg.d_q, cfg.num_heads, cfg.dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, image_emb: torch.Tensor) -> torch.Tensor:
        # image_emb: (B, Dv)
        kv = self.kv_proj(image_emb).unsqueeze(1)  # (B, 1, d_q)
        q = self.query_tokens.expand(image_emb.size(0), -1, -1)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, CrossAttentionBlock):
                q = layer(q, kv)
            else:
                q = layer(q)
        return q
