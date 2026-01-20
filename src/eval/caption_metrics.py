from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu

from src.data.flickr8k import Flickr8kDataset, parse_captions, Flickr8kPaths
from src.utils.io import load_yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--preds", required=True, help="JSON {image_name: prediction}")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    return ap.parse_args()


def main():
    args = parse_args()

    # Load predictions
    with open(args.preds, "r", encoding="utf-8") as f:
        preds = json.load(f)

    paths = Flickr8kPaths(Path(args.data_root))
    caps = parse_captions(paths.token_file)

    if args.split == "train":
        names = [ln.strip() for ln in paths.train_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    elif args.split == "val":
        names = [ln.strip() for ln in paths.val_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        names = [ln.strip() for ln in paths.test_list.read_text(encoding="utf-8").splitlines() if ln.strip()]

    sys_hyps = []
    refs = [[] for _ in range(5)]  # Flickr8k has 5 refs per image

    missing = 0
    for n in names:
        if n not in preds:
            missing += 1
            continue
        sys_hyps.append(preds[n])
        r = caps.get(n, [""])
        # pad to 5
        r = (r + [""] * 5)[:5]
        for i in range(5):
            refs[i].append(r[i])

    bleu = sacrebleu.corpus_bleu(sys_hyps, refs)

    out = {
        "split": args.split,
        "num_images": len(sys_hyps),
        "missing": missing,
        "bleu": {
            "score": bleu.score,
            "precisions": bleu.precisions,
            "bp": bleu.bp,
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
