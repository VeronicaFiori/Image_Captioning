from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .vision_encoder import OpenCLIPVisionEncoder, VisionCfg
from .qformer import QFormer, QFormerCfg


@dataclass
class CaptionModelCfg:
    vision: VisionCfg
    qformer: QFormerCfg
    llm_name: str
    llm_freeze: bool = True


class BLIP2StyleCaptioner(nn.Module):
    def __init__(self, cfg: CaptionModelCfg):
        super().__init__()
        self.vision = OpenCLIPVisionEncoder(cfg.vision)
        self.qformer = QFormer(cfg.qformer, vision_dim=self.vision.out_dim)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name, use_fast=True)
        self.llm = T5ForConditionalGeneration.from_pretrained(cfg.llm_name)

        if cfg.llm_freeze:
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.eval()

        # Project Q-Former hidden to T5 d_model
        self.proj_to_t5 = nn.Linear(cfg.qformer.d_q, self.llm.config.d_model)

    def encode_prompt(self, prompt_text: Optional[str], device: torch.device):
        if prompt_text is None or len(prompt_text.strip()) == 0:
            return None, None
        tok = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=64,
        )
        input_ids = tok.input_ids.to(device)
        attn_mask = tok.attention_mask.to(device)
        # Get embeddings from T5 encoder embed tokens
        prompt_embeds = self.llm.get_input_embeddings()(input_ids)
        return prompt_embeds, attn_mask

    def forward(
        self,
        images: torch.Tensor,
        target_text: Optional[list[str]] = None,
        prompt_text: Optional[str] = None,
        max_text_len: int = 64,
    ):
        device = images.device
        with torch.no_grad():
            img_emb = self.vision(images)  # (B, Dv)
        q = self.qformer(img_emb)  # (B, Nq, d_q)
        vis_embeds = self.proj_to_t5(q)  # (B, Nq, d_t5)

        prompt_embeds, prompt_mask = self.encode_prompt(prompt_text, device)

        if prompt_embeds is not None:
            enc_embeds = torch.cat([prompt_embeds.expand(images.size(0), -1, -1), vis_embeds], dim=1)
            enc_mask = torch.cat(
                [prompt_mask.expand(images.size(0), -1), torch.ones(images.size(0), vis_embeds.size(1), device=device, dtype=torch.long)],
                dim=1,
            )
        else:
            enc_embeds = vis_embeds
            enc_mask = torch.ones(images.size(0), vis_embeds.size(1), device=device, dtype=torch.long)

        if target_text is None:
            # inference-only forward not used; call generate_caption instead
            raise ValueError("target_text is required for forward(). Use generate_caption() for inference.")

        tok = self.tokenizer(
            target_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_len,
        )
        labels = tok.input_ids.to(device)
        labels[labels == self.tokenizer.pad_token_id] = -100

        out = self.llm(
            inputs_embeds=enc_embeds,
            attention_mask=enc_mask,
            labels=labels,
        )
        return out

    @torch.no_grad()
    def generate_caption(
        self,
        images: torch.Tensor,
        prompt_text: str,
        num_beams: int = 5,
        max_new_tokens: int = 40,
    ) -> list[str]:
        self.eval()
        device = images.device
        img_emb = self.vision(images)
        q = self.qformer(img_emb)
        vis_embeds = self.proj_to_t5(q)

        prompt_embeds, prompt_mask = self.encode_prompt(prompt_text, device)
        if prompt_embeds is not None:
            enc_embeds = torch.cat([prompt_embeds.expand(images.size(0), -1, -1), vis_embeds], dim=1)
            enc_mask = torch.cat(
                [prompt_mask.expand(images.size(0), -1), torch.ones(images.size(0), vis_embeds.size(1), device=device, dtype=torch.long)],
                dim=1,
            )
        else:
            enc_embeds = vis_embeds
            enc_mask = torch.ones(images.size(0), vis_embeds.size(1), device=device, dtype=torch.long)

        gen_ids = self.llm.generate(
            inputs_embeds=enc_embeds,
            attention_mask=enc_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
