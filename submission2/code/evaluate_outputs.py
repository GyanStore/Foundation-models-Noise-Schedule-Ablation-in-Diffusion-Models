# ADDED: this whole file is new code written by the group (KID / FID / LPIPS evaluation).
"""Quantitative evaluation of Run 0 / A / B photo -> MetFaces-style translations.

Metrics
-------
* KID (Kernel Inception Distance) between each run's outputs and the held-out
  MetFaces test split: how close the outputs are to the target artistic domain
  (lower = more "Old Masters"-like). KID is used instead of FID as the primary
  score because it is unbiased for small sample sizes; FID is reported too.
* LPIPS between each output and its own input photo: how much of the subject's
  content/structure is preserved (lower = closer to the input).

The raw input photos are also scored as a reference row ("input"), i.e. the
"no translation at all" point.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reference-manifest", type=Path, required=True,
                   help="text file listing the MetFaces test-split images (run_A_test_split.txt)")
    p.add_argument("--input-dir", type=Path, required=True, help="original input photos")
    p.add_argument("--run", action="append", required=True, metavar="NAME=DIR",
                   help="run outputs, e.g. --run 0=.../run_0_outputs (repeatable)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--kid-subsets", type=int, default=100)
    return p.parse_args()


def list_images(root: Path) -> list[Path]:
    return sorted(p for ext in IMG_EXTS for p in root.glob(ext))


def load_batch(paths: list[Path], size: int) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),  # [0, 1]
    ])
    return torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths])


def domain_scores(real: torch.Tensor, fake: torch.Tensor, kid_subsets: int) -> dict[str, float]:
    subset = min(50, len(real), len(fake))
    kid = KernelInceptionDistance(subsets=kid_subsets, subset_size=subset, normalize=True)
    fid = FrechetInceptionDistance(normalize=True)
    for m in (kid, fid):
        m.update(real, real=True)
        m.update(fake, real=False)
    kid_mean, kid_std = kid.compute()
    return {"kid": kid_mean.item(), "kid_std": kid_std.item(), "fid": fid.compute().item()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    ref_paths = [Path(line.strip()) for line in args.reference_manifest.read_text().splitlines() if line.strip()]
    real = load_batch(ref_paths, args.image_size)
    input_paths = list_images(args.input_dir)
    inputs = load_batch(input_paths, args.image_size)
    print(f"reference MetFaces test images: {len(real)} | input photos: {len(inputs)}")

    lpips_fn = lpips.LPIPS(net="alex", verbose=False)

    rows = []
    rows.append({"run": "input", **domain_scores(real, inputs, args.kid_subsets), "lpips_to_input": 0.0, "n": len(inputs)})

    for spec in args.run:
        name, run_dir = spec.split("=", 1)
        run_dir = Path(run_dir)
        pairs = []
        for ip in input_paths:
            matches = list_images_matching(run_dir, ip.stem)
            if matches:
                pairs.append((ip, matches[0]))
        if not pairs:
            raise ValueError(f"No outputs in {run_dir} match inputs in {args.input_dir}")
        outs = load_batch([o for _, o in pairs], args.image_size)
        ins = load_batch([i for i, _ in pairs], args.image_size)
        with torch.no_grad():
            lp = lpips_fn(outs * 2 - 1, ins * 2 - 1).flatten().mean().item()
        rows.append({"run": name, **domain_scores(real, outs, args.kid_subsets), "lpips_to_input": lp, "n": len(outs)})
        print(rows[-1])

    csv_path = args.output_dir / "eval_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    names = [r["run"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(names, [r["kid"] for r in rows], yerr=[r["kid_std"] for r in rows], capsize=4, color="#4C72B0")
    axes[0].set_title("KID to MetFaces test split (lower = more artistic)")
    axes[1].bar(names, [r["lpips_to_input"] for r in rows], color="#DD8452")
    axes[1].set_title("LPIPS to input photo (lower = more identity kept)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "eval_metrics.png", dpi=150)
    print(f"wrote {csv_path} and eval_metrics.png")


def list_images_matching(run_dir: Path, stem: str) -> list[Path]:
    return sorted(p for ext in IMG_EXTS for p in run_dir.glob(f"{stem}_*{ext[1:]}"))


if __name__ == "__main__":
    np.set_printoptions(precision=5)
    main()
