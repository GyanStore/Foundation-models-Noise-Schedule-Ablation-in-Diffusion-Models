# ADDED: this whole file is new code written by the group (base model code in diffusers is used unchanged).
# Marker convention: lines that change the native noise schedule beta_t -> beta_t^x are tagged # MODIFIED;
# every other block in this file is group-written training code (# ADDED by file).
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler

EXPONENT_X = 2.00  # MODIFIED: exponent x applied to the native noise parameter theta_t = beta_t (Run B)
DEFAULT_MODEL_ID = "google/ddpm-celebahq-256"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleEMA:
    """Exponential moving average of model weights, with the usual (1+n)/(10+n) warmup."""

    def __init__(self, params, decay: float) -> None:
        self.decay = decay
        self.num_updates = 0
        self.shadow = [p.detach().clone() for p in params]
        self.backup: list[torch.Tensor] = []

    @torch.no_grad()
    def step(self, params) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for s, p in zip(self.shadow, params):
            s.mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def store(self, params) -> None:
        self.backup = [p.detach().clone() for p in params]

    @torch.no_grad()
    def copy_to(self, params) -> None:
        for s, p in zip(self.shadow, params):
            p.copy_(s)

    @torch.no_grad()
    def restore(self, params) -> None:
        for b, p in zip(self.backup, params):
            p.copy_(b)
        self.backup = []


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, image_size: int, hflip: bool = False) -> None:
        self.paths = sorted(
            [
                p
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
                for p in root.rglob(ext)
            ]
        )
        if not self.paths:
            raise ValueError(f"No images found in {root}")

        tfms: list = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
        if hflip:
            tfms.append(transforms.RandomHorizontalFlip(p=0.5))
        tfms += [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = transforms.Compose(tfms)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


@dataclass
class TrainConfig:
    run_name: str
    dataset_dir: Path
    output_dir: Path
    model_id: str
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    save_every: int
    train_split: float
    val_split: float
    test_split: float
    val_every: int
    checkpoint_input_dir: Path | None
    checkpoint_num_images: int
    checkpoint_num_steps: int
    checkpoint_strength: float
    lr_scheduler: str
    lr_warmup_steps: int
    grad_accum_steps: int
    grad_clip: float
    ema_decay: float
    hflip: bool
    val_repeats: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", choices=["A", "B"], required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--train-split", type=float, default=0.80)
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--test-split", type=float, default=0.10)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--checkpoint-input-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-num-images", type=int, default=3)
    parser.add_argument("--checkpoint-num-steps", type=int, default=100)
    parser.add_argument("--checkpoint-strength", type=float, default=0.60)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.0, help="0 disables clipping")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="0 disables EMA")
    parser.add_argument("--hflip", action="store_true", help="random horizontal flip on train split")
    parser.add_argument("--val-repeats", type=int, default=1, help="passes over val set with fixed noise")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.val_every < 1:
        raise ValueError("--val-every must be >= 1")
    if args.checkpoint_num_images < 1:
        raise ValueError("--checkpoint-num-images must be >= 1")
    if args.checkpoint_num_steps < 2:
        raise ValueError("--checkpoint-num-steps must be >= 2")
    if not (0.0 <= args.checkpoint_strength <= 1.0):
        raise ValueError("--checkpoint-strength must be in [0, 1]")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if not (0.0 <= args.ema_decay < 1.0):
        raise ValueError("--ema-decay must be in [0, 1)")
    if args.val_repeats < 1:
        raise ValueError("--val-repeats must be >= 1")

    split_sum = args.train_split + args.val_split + args.test_split
    if abs(split_sum - 1.0) > 1e-8:
        raise ValueError("train/val/test splits must sum to 1.0")

    return TrainConfig(
        run_name=args.run_name,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        save_every=args.save_every,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        val_every=args.val_every,
        checkpoint_input_dir=args.checkpoint_input_dir,
        checkpoint_num_images=args.checkpoint_num_images,
        checkpoint_num_steps=args.checkpoint_num_steps,
        checkpoint_strength=args.checkpoint_strength,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        hflip=args.hflip,
        val_repeats=args.val_repeats,
    )


def build_scheduler_for_run(base_scheduler: DDPMScheduler, run_name: str) -> DDPMScheduler:
    if run_name == "A":
        return DDPMScheduler.from_config(base_scheduler.config)  # Run A: native beta_t, unchanged

    base_betas = base_scheduler.betas.detach().cpu()  # MODIFIED: read native beta_t (linear 1e-4..0.02, T=1000)
    transformed_betas = torch.pow(base_betas, EXPONENT_X).numpy()  # MODIFIED: theta_t -> theta_t^x
    return DDPMScheduler.from_config(base_scheduler.config, trained_betas=transformed_betas)  # MODIFIED: Run B schedule


def selected_checkpoint_epochs(total_epochs: int) -> set[int]:
    if total_epochs < 5:
        return {total_epochs}
    raw = np.linspace(1, total_epochs, num=5)
    return {int(round(x)) for x in raw}


def save_pipeline(unet: nn.Module, scheduler: DDPMScheduler, model_id: str, save_path: Path) -> None:
    pipe = DDPMPipeline.from_pretrained(model_id, unet=unet, scheduler=scheduler)
    pipe.save_pretrained(save_path)


def compute_split_lengths(n: int, train_split: float, val_split: float, test_split: float) -> tuple[int, int, int]:
    raw = np.array([train_split, val_split, test_split], dtype=np.float64) * n
    lengths = np.floor(raw).astype(int)
    remainder = n - int(lengths.sum())
    if remainder > 0:
        order = np.argsort(-(raw - lengths))
        for idx in order[:remainder]:
            lengths[idx] += 1
    train_len, val_len, test_len = map(int, lengths.tolist())
    if train_len < 1:
        raise ValueError("Train split is empty. Increase --train-split or dataset size.")
    return train_len, val_len, test_len


def write_split_manifest(paths: Sequence[Path], subset: Subset[torch.Tensor], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        for idx in subset.indices:
            f.write(f"{paths[int(idx)]}\n")


def list_images(root: Path) -> list[Path]:
    return sorted([p for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp") for p in root.rglob(ext)])


def load_infer_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tfm = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
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


@torch.no_grad()
def sdedit_infer(
    pipe: DDPMPipeline,
    x0: torch.Tensor,
    num_steps: int,
    strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    unet = pipe.unet
    scheduler = pipe.scheduler
    device = next(unet.parameters()).device

    scheduler.set_timesteps(num_steps, device=device)
    # Standard img2img convention: strength=1.0 starts from pure noise, 0.0 leaves the input untouched.
    start_idx = min(num_steps - 1, int(round(num_steps * (1.0 - strength))))
    timesteps = scheduler.timesteps[start_idx:]
    # Skip timesteps whose noise level underflows to 0 in float32 (beta_t^2 schedule) -> avoids NaN.
    noise_level = 1.0 - scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
    timesteps = timesteps[noise_level > 0]

    x0 = x0.to(device)
    noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)
    start_timestep = timesteps[0].unsqueeze(0)
    xt = scheduler.add_noise(x0, noise, start_timestep)

    for t in timesteps:
        pred_noise = unet(xt, t).sample
        xt = scheduler.step(pred_noise, t, xt).prev_sample

    return xt


@torch.no_grad()
def generate_checkpoint_samples(
    cfg: TrainConfig,
    unet: nn.Module,
    scheduler: DDPMScheduler,
    epoch: int,
    device: torch.device,
    input_paths: list[Path],
) -> None:
    if not input_paths:
        return

    sample_dir = cfg.output_dir / f"run_{cfg.run_name}_epoch_{epoch:03d}_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    pipe = DDPMPipeline.from_pretrained(cfg.model_id, unet=unet, scheduler=scheduler).to(device)
    pipe.unet.eval()
    generator = torch.Generator(device=device).manual_seed(cfg.seed + epoch)

    for path in input_paths[: cfg.checkpoint_num_images]:
        x0 = load_infer_image(path, cfg.image_size)
        out = sdedit_infer(
            pipe,
            x0,
            num_steps=cfg.checkpoint_num_steps,
            strength=cfg.checkpoint_strength,
            generator=generator,
        )
        to_pil(out).save(sample_dir / f"{path.stem}_epoch_{epoch:03d}_{cfg.run_name}.png")


def validate_epoch(
    unet: nn.Module,
    dataloader: DataLoader,
    scheduler: DDPMScheduler,
    device: torch.device,
    seed: int,
    repeats: int = 1,
) -> float:
    """Validation MSE with a fixed noise/timestep draw, so epochs (and runs) are comparable."""
    if len(dataloader) == 0:
        return float("nan")

    unet.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    count = 0
    with torch.no_grad():
        for _ in range(repeats):
            for batch in dataloader:
                images = batch.to(device)
                batch_size = images.shape[0]
                noise = torch.randn(images.shape, generator=gen).to(device)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch_size,), generator=gen
                ).to(device)
                noisy = scheduler.add_noise(images, noise, timesteps)
                pred = unet(noisy, timesteps).sample
                total += F.mse_loss(pred, noise).item() * batch_size
                count += batch_size

    return total / count


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolderDataset(cfg.dataset_dir, cfg.image_size)
    train_source = ImageFolderDataset(cfg.dataset_dir, cfg.image_size, hflip=True) if cfg.hflip else dataset
    if len(dataset) < cfg.batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} images, smaller than batch size {cfg.batch_size}. "
            "Use a smaller --batch-size."
        )

    train_len, val_len, test_len = compute_split_lengths(
        len(dataset), cfg.train_split, cfg.val_split, cfg.test_split
    )
    split_gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=split_gen)

    dataloader = DataLoader(
        Subset(train_source, train_ds.indices),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    base_pipe = DDPMPipeline.from_pretrained(cfg.model_id)
    unet = base_pipe.unet.to(device)
    noise_scheduler = build_scheduler_for_run(base_pipe.scheduler, cfg.run_name)

    optimizer = AdamW(unet.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(dataloader) / cfg.grad_accum_steps)
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=steps_per_epoch * cfg.epochs,
    )
    ema = SimpleEMA(unet.parameters(), decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

    @contextlib.contextmanager
    def eval_weights():
        """Temporarily swap in EMA weights (if enabled) for validation/saving/sampling."""
        if ema is None:
            yield
            return
        ema.store(unet.parameters())
        ema.copy_to(unet.parameters())
        try:
            yield
        finally:
            ema.restore(unet.parameters())

    metadata = {
        "run_name": cfg.run_name,
        "model_id": cfg.model_id,
        "exponent_x": EXPONENT_X,
        "image_size": cfg.image_size,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler": cfg.lr_scheduler,
        "lr_warmup_steps": cfg.lr_warmup_steps,
        "grad_accum_steps": cfg.grad_accum_steps,
        "effective_batch_size": cfg.batch_size * cfg.grad_accum_steps,
        "grad_clip": cfg.grad_clip,
        "ema_decay": cfg.ema_decay,
        "hflip": cfg.hflip,
        "val_repeats": cfg.val_repeats,
        "seed": cfg.seed,
        "dataset_dir": str(cfg.dataset_dir),
        "split": {
            "train_split": cfg.train_split,
            "val_split": cfg.val_split,
            "test_split": cfg.test_split,
            "train_count": train_len,
            "val_count": val_len,
            "test_count": test_len,
        },
        "checkpoint_sampling": {
            "enabled": cfg.checkpoint_input_dir is not None,
            "input_dir": str(cfg.checkpoint_input_dir) if cfg.checkpoint_input_dir is not None else None,
            "num_images": cfg.checkpoint_num_images,
            "num_steps": cfg.checkpoint_num_steps,
            "strength": cfg.checkpoint_strength,
        },
    }
    with (cfg.output_dir / f"run_{cfg.run_name}_config.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    write_split_manifest(dataset.paths, train_ds, cfg.output_dir / f"run_{cfg.run_name}_train_split.txt")
    write_split_manifest(dataset.paths, val_ds, cfg.output_dir / f"run_{cfg.run_name}_val_split.txt")
    write_split_manifest(dataset.paths, test_ds, cfg.output_dir / f"run_{cfg.run_name}_test_split.txt")

    log_path = cfg.output_dir / f"run_{cfg.run_name}_train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "lr"])

    val_log_path = cfg.output_dir / f"run_{cfg.run_name}_val_log.csv"
    with val_log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "val_loss"])

    checkpoint_epochs = selected_checkpoint_epochs(cfg.epochs)

    checkpoint_input_paths: list[Path] = []
    if cfg.checkpoint_input_dir is not None:
        checkpoint_input_paths = list_images(cfg.checkpoint_input_dir)
        if not checkpoint_input_paths:
            raise ValueError(f"No checkpoint sampling images found in {cfg.checkpoint_input_dir}")

    best_val = float("inf")
    best_epoch = None
    val_seed = 1234  # same fixed validation draw for every epoch and for both runs

    for epoch in range(1, cfg.epochs + 1):
        unet.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(dataloader, desc=f"Run {cfg.run_name} | Epoch {epoch}/{cfg.epochs}")
        for step_idx, batch in enumerate(progress, start=1):
            images = batch.to(device)
            batch_size = images.shape[0]

            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )

            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            noise_pred = unet(noisy_images, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

            (loss / cfg.grad_accum_steps).backward()
            if step_idx % cfg.grad_accum_steps == 0 or step_idx == len(dataloader):
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.step(unet.parameters())

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.6f}")

        mean_loss = epoch_loss / len(dataloader)
        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{mean_loss:.8f}", f"{lr_scheduler.get_last_lr()[0]:.3e}"])

        should_save_ckpt = epoch in checkpoint_epochs
        should_save_periodic = cfg.save_every > 0 and (epoch % cfg.save_every == 0)

        with eval_weights():
            if len(val_ds) > 0 and (epoch % cfg.val_every == 0):
                val_loss = validate_epoch(
                    unet, val_loader, noise_scheduler, device, seed=val_seed, repeats=cfg.val_repeats
                )
                with val_log_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, f"{val_loss:.8f}"])
                if val_loss < best_val:
                    best_val, best_epoch = val_loss, epoch
                    save_pipeline(unet, noise_scheduler, cfg.model_id, cfg.output_dir / f"run_{cfg.run_name}_best")
                    with (cfg.output_dir / f"run_{cfg.run_name}_best.json").open("w") as f:
                        json.dump({"best_epoch": best_epoch, "best_val_loss": best_val}, f, indent=2)

            if should_save_ckpt or should_save_periodic:
                ckpt_dir = cfg.output_dir / f"run_{cfg.run_name}_epoch_{epoch:03d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_pipeline(unet, noise_scheduler, cfg.model_id, ckpt_dir)

                if should_save_ckpt and checkpoint_input_paths:
                    generate_checkpoint_samples(
                        cfg=cfg,
                        unet=unet,
                        scheduler=noise_scheduler,
                        epoch=epoch,
                        device=device,
                        input_paths=checkpoint_input_paths,
                    )
        unet.train()

    final_dir = cfg.output_dir / f"run_{cfg.run_name}_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    with eval_weights():
        save_pipeline(unet, noise_scheduler, cfg.model_id, final_dir)


if __name__ == "__main__":
    train(parse_args())
