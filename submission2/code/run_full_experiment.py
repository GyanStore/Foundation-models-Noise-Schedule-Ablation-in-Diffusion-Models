# ADDED: this whole file is new code written by the group (orchestrates analyze -> train A/B -> infer 0/A/B).
# No line here changes the noise schedule; the beta_t -> beta_t^x change is in train_ddpm_metfaces.py (# MODIFIED).
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("\n$", " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="google/ddpm-celebahq-256")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--strength", type=float, default=0.60)
    parser.add_argument("--checkpoint-num-images", type=int, default=3)
    parser.add_argument("--checkpoint-num-steps", type=int, default=100)
    parser.add_argument("--checkpoint-strength", type=float, default=0.60)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--val-repeats", type=int, default=1)
    parser.add_argument("--infer-checkpoint", choices=["final", "best"], default="final")
    parser.add_argument("--skip-schedule", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_ddpm_metfaces.py"
    infer_script = script_dir / "run_ddpm_img2img.py"
    schedule_script = script_dir / "analyze_schedule.py"

    schedule_out = args.output_dir / "schedule"
    train_out = args.output_dir / "train"
    infer_out = args.output_dir / "inference"

    py = sys.executable

    if not args.skip_schedule:
        run_cmd([
            py,
            str(schedule_script),
            "--model-id",
            args.model_id,
            "--output-dir",
            str(schedule_out),
        ])

    extra_train = [
        "--lr-scheduler", args.lr_scheduler,
        "--lr-warmup-steps", str(args.lr_warmup_steps),
        "--grad-accum-steps", str(args.grad_accum_steps),
        "--grad-clip", str(args.grad_clip),
        "--ema-decay", str(args.ema_decay),
        "--val-repeats", str(args.val_repeats),
    ] + (["--hflip"] if args.hflip else [])

    for run_name in ["A", "B"]:
        run_cmd([
            py,
            str(train_script),
            "--run-name",
            run_name,
            "--dataset-dir",
            str(args.dataset_dir),
            "--output-dir",
            str(train_out),
            "--model-id",
            args.model_id,
            "--image-size",
            str(args.image_size),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--checkpoint-input-dir",
            str(args.input_dir),
            "--checkpoint-num-images",
            str(args.checkpoint_num_images),
            "--checkpoint-num-steps",
            str(args.checkpoint_num_steps),
            "--checkpoint-strength",
            str(args.checkpoint_strength),
        ] + extra_train)

    run_cmd([
        py,
        str(infer_script),
        "--model-path",
        args.model_id,
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(infer_out),
        "--run-name",
        "0",
        "--num-steps",
        str(args.num_steps),
        "--strength",
        str(args.strength),
        "--image-size",
        str(args.image_size),
        "--seed",
        str(args.seed),
        "--save-trajectory",
    ])

    for run_name in ["A", "B"]:
        run_cmd([
            py,
            str(infer_script),
            "--model-path",
            str(train_out / f"run_{run_name}_{args.infer_checkpoint}"),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(infer_out),
            "--run-name",
            run_name,
            "--num-steps",
            str(args.num_steps),
            "--strength",
            str(args.strength),
            "--image-size",
            str(args.image_size),
            "--seed",
            str(args.seed),
            "--save-trajectory",
        ])


if __name__ == "__main__":
    main()
