from __future__ import annotations

from typing import Any, Dict

from .captioner_t5 import BLIP2StyleCaptioner, CaptionModelCfg
from .vision_encoder import VisionCfg
from .qformer import QFormerCfg


def build_model(cfg: Dict[str, Any]) -> BLIP2StyleCaptioner:
    m = cfg["model"]
    vision = VisionCfg(**m["vision"])
    qformer = QFormerCfg(**m["qformer"])
    llm_name = m["llm"]["name"]
    llm_freeze = bool(m["llm"].get("freeze", True))

    model_cfg = CaptionModelCfg(
        vision=vision,
        qformer=qformer,
        llm_name=llm_name,
        llm_freeze=llm_freeze,
    )
    return BLIP2StyleCaptioner(model_cfg)
