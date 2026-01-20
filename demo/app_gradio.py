from __future__ import annotations

import argparse

import gradio as gr
import torch
from PIL import Image

from src.utils.io import load_yaml
from src.models.build import build_model
from src.data.transforms import build_transform


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/flickr8k_caption.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_captioner(ckpt: str, config: str, device: str):
    cfg = load_yaml(config)
    model = build_model(cfg)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    tfm = build_transform(train=False)
    return model, tfm


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model, tfm = load_captioner(args.ckpt, args.config, device)

    def run(image: Image.Image, style: str, num_beams: int, max_new_tokens: int):
        if image is None:
            return ""
        prompt_map = {
            "brief": "Describe the image in one short sentence.",
            "detailed": "Describe the image in detail.",
            "factual": "Describe only what is clearly visible in the image. Avoid guessing.",
        }
        prompt = prompt_map.get(style, prompt_map["brief"])
        x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            cap = model.generate_caption(x, prompt_text=prompt, num_beams=num_beams, max_new_tokens=max_new_tokens)[0]
        return cap

    with gr.Blocks() as demo:
        gr.Markdown("# Flickr8k Captioning (BLIP2-style)")
        with gr.Row():
            inp = gr.Image(type="pil", label="Image")
            out = gr.Textbox(label="Caption")
        with gr.Row():
            style = gr.Dropdown(["brief", "detailed", "factual"], value="brief", label="Style")
            beams = gr.Slider(1, 8, value=5, step=1, label="Beams")
            max_tok = gr.Slider(10, 120, value=40, step=1, label="Max new tokens")
        btn = gr.Button("Generate")
        btn.click(run, inputs=[inp, style, beams, max_tok], outputs=[out])

    demo.launch()


if __name__ == "__main__":
    main()
