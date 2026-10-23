from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import DDPMPipeline


def load_image(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tfm = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return tfm(image).unsqueeze(0)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    tensor = (tensor + 1.0) / 2.0
    arr = (tensor[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def step_indices(n: int, k: int = 5) -> set[int]:
    if n <= k:
        return set(range(n))
    return {int(round(x)) for x in np.linspace(0, n - 1, k)}


@torch.no_grad()
def sdedit_sample(
    pipe: DDPMPipeline,
    input_tensor: torch.Tensor,
    num_steps: int,
    strength: float,
    generator: torch.Generator,
    trajectory_dir: Path | None,
    image_stem: str,
) -> torch.Tensor:
    unet = pipe.unet
    scheduler = pipe.scheduler
    device = next(unet.parameters()).device

    scheduler.set_timesteps(num_steps, device=device)
    start_idx = max(0, min(num_steps - 1, int(round(strength * (num_steps - 1)))))
    timesteps = scheduler.timesteps[start_idx:]

    x0 = input_tensor.to(device)
    noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)
    start_timestep = timesteps[0].unsqueeze(0)
    xt = scheduler.add_noise(x0, noise, start_timestep)

    capture_steps = step_indices(len(timesteps), 5)
    for i, t in enumerate(timesteps):
        if trajectory_dir is not None and i in capture_steps:
            image = to_pil(xt)
            image.save(trajectory_dir / f"{image_stem}_step_{i:03d}_t_{int(t.item()):04d}.png")

        pred_noise = unet(xt, t).sample
        xt = scheduler.step(pred_noise, t, xt).prev_sample

    return xt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--strength", type=float, default=0.60)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-trajectory", action="store_true")
    args = parser.parse_args()

    if args.num_steps < 2:
        raise ValueError("--num-steps must be >= 2")
    if not (0.0 <= args.strength <= 1.0):
        raise ValueError("--strength must be in [0, 1]")

    return args


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    pipe = DDPMPipeline.from_pretrained(args.model_path)
    pipe = pipe.to(device)

    input_paths = sorted(
        [
            p
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
            for p in args.input_dir.rglob(ext)
        ]
    )
    if not input_paths:
        raise ValueError(f"No images found in {args.input_dir}")

    out_dir = args.output_dir / f"run_{args.run_name}_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_dir = None
    if args.save_trajectory:
        traj_dir = args.output_dir / f"run_{args.run_name}_trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)

    for path in tqdm(input_paths, desc=f"Run {args.run_name} inference"):
        x0 = load_image(path, args.image_size)
        out = sdedit_sample(
            pipe=pipe,
            input_tensor=x0,
            num_steps=args.num_steps,
            strength=args.strength,
            generator=generator,
            trajectory_dir=traj_dir,
            image_stem=path.stem,
        )
        to_pil(out).save(out_dir / f"{path.stem}_{args.run_name}.png")


if __name__ == "__main__":
    main()
