from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--preds", required=True, help="JSON {image_name: caption}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


SYSTEM_PROMPT = (
    "You are a strict visual fact-checker. "
    "Given an image and a caption, decide whether every claim in the caption is supported by visible evidence. "
    "Return ONLY valid JSON with keys: supported (bool), unsupported_claims (list of strings), "
    "missing_details (list of strings), score_0_1 (float). "
    "Be conservative: if uncertain, mark unsupported."
)


def main():
    args = parse_args()

    with open(args.preds, "r", encoding="utf-8") as f:
        preds = json.load(f)

    # Lazy imports to avoid hard dependency for users who skip judge
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    kwargs = {}
    if args.load_in_4bit:
        kwargs.update({"load_in_4bit": True, "device_map": "auto"})
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        **kwargs,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    data_root = Path(args.data_root)
    images_dir = data_root / "Images"

    results = {}
    for img_name, caption in tqdm(preds.items(), desc="qwen_judge"):
        img_path = images_dir / img_name
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"CAPTION: {caption}"},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if not args.load_in_4bit:
            inputs = inputs.to(args.device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        # Extract JSON (best effort)
        j = None
        try:
            start = out_text.find("{")
            end = out_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                j = json.loads(out_text[start : end + 1])
        except Exception:
            j = {"parse_error": True, "raw": out_text}

        results[img_name] = {"caption": caption, "judge": j}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
