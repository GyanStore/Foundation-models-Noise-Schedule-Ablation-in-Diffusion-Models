# ADDED: this whole file is new code written by the group (test/dev photo download).
"""Download held-out real face photos (FFHQ, 256x256) for testing and inference-setting selection.

FFHQ (Karras et al., 2019; https://github.com/NVlabs/ffhq-dataset) is a Flickr photo dataset that does
not overlap CelebA-HQ (the pretraining data of google/ddpm-celebahq-256). MetFaces was aligned and
cropped with the same procedure as FFHQ, so these photos match the training images in framing.

Images come from the Hugging Face mirror `bitmind/ffhq-256` (one parquet shard). A fixed seed picks
`--n-test + --n-dev` distinct images; the test and dev sets are disjoint.
"""
from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "bitmind/ffhq-256"
SHARD = "data/train-00000-of-00016.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--dev-dir", type=Path, required=True)
    p.add_argument("--sources-file", type=Path, required=True)
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--n-dev", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shard = hf_hub_download(REPO_ID, SHARD, repo_type="dataset")
    df = pd.read_parquet(shard)
    img_col = next(c for c in df.columns if c.lower().startswith("image"))

    rng = random.Random(args.seed)
    idx = rng.sample(range(len(df)), args.n_test + args.n_dev)
    lines = [f"source: https://huggingface.co/datasets/{REPO_ID} ({SHARD}), seed={args.seed}", "set,file,row_index"]
    for k, i in enumerate(idx):
        cell = df.iloc[i][img_col]
        data = cell["bytes"] if isinstance(cell, dict) else cell
        img = Image.open(io.BytesIO(data)).convert("RGB")
        is_test = k < args.n_test
        out_dir = args.test_dir if is_test else args.dev_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"ffhq_{'test' if is_test else 'dev'}_{(k if is_test else k - args.n_test):02d}.png"
        img.save(out_dir / name)
        lines.append(f"{'test' if is_test else 'dev'},{name},{i}")
    args.sources_file.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.n_test} test -> {args.test_dir}, {args.n_dev} dev -> {args.dev_dir}")


if __name__ == "__main__":
    main()
