# ADDED: new file written by the group; plots theta_t = beta_t and theta_t^x and verifies saved schedulers.
from __future__ import annotations

import argparse
import json  # ADDED
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler  # ADDED

EXPONENT_X = 2.00  # MODIFIED: exponent x (same constant as train_ddpm_metfaces.py)
DEFAULT_MODEL_ID = "google/ddpm-celebahq-256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-a-model", type=Path, default=None, help="saved Run A pipeline to verify")  # ADDED
    parser.add_argument("--run-b-model", type=Path, default=None, help="saved Run B pipeline to verify")  # ADDED
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = DDPMPipeline.from_pretrained(args.model_id)
    betas = pipe.scheduler.betas.detach().cpu().numpy()
    betas_x = np.power(betas, EXPONENT_X)  # MODIFIED: theta_t -> theta_t^x

    if not (np.all((betas > 0.0) & (betas < 1.0)) and np.all((betas_x > 0.0) & (betas_x < 1.0))):
        raise ValueError("Schedule bounds check failed: expected all values in (0,1).")

    t = np.arange(1, len(betas) + 1)
    df = pd.DataFrame({"t": t, "beta_t": betas, f"beta_t_pow_{EXPONENT_X:.2f}": betas_x})
    df.to_csv(args.output_dir / "schedule_values.csv", index=False)

    plt.figure(figsize=(9, 4.8))
    plt.plot(t, betas, label="beta_t (Run A)")
    plt.plot(t, betas_x, label=f"beta_t^{EXPONENT_X:.2f} (Run B)")
    plt.xlabel("timestep t")
    plt.ylabel("noise variance parameter")
    plt.title("Noise schedule ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "schedule_plot.png", dpi=200)

    # ADDED: cumulative signal alpha_bar_t and log-SNR for both schedules (float64: no underflow).
    curves = {}  # ADDED
    for name, b in (("Run A: beta_t", betas.astype(np.float64)), (f"Run B: beta_t^{EXPONENT_X:.2f}", betas_x.astype(np.float64))):  # ADDED
        abar = np.cumprod(1.0 - b)  # ADDED
        curves[name] = (abar, np.log(abar) - np.log1p(-abar))  # ADDED
    df["alpha_bar_A"], df["log_snr_A"] = list(curves.values())[0]  # ADDED
    df["alpha_bar_B"], df["log_snr_B"] = list(curves.values())[1]  # ADDED
    df.to_csv(args.output_dir / "schedule_values.csv", index=False)  # ADDED
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # ADDED
    for name, (abar, lsnr) in curves.items():  # ADDED
        axes[0].plot(t, abar, label=name)  # ADDED
        axes[1].plot(t, lsnr, label=name)  # ADDED
    axes[0].set(xlabel="timestep t", ylabel="alpha_bar_t (signal kept)", title="Cumulative signal alpha_bar_t")  # ADDED
    axes[1].set(xlabel="timestep t", ylabel="log SNR = log(alpha_bar/(1-alpha_bar))", title="Log signal-to-noise ratio")  # ADDED
    for ax in axes:  # ADDED
        ax.grid(alpha=0.3)  # ADDED
        ax.legend()  # ADDED
    fig.tight_layout()  # ADDED
    fig.savefig(args.output_dir / "schedule_alphabar_snr.png", dpi=200)  # ADDED
    summary = {name: {"alpha_bar_T": float(a[-1]), "log_snr_T": float(s[-1]), "log_snr_1": float(s[0])}  # ADDED
               for name, (a, s) in curves.items()}  # ADDED

    # ADDED: verify that the saved fine-tuned schedulers really use beta_t (A) and beta_t^x (B).
    lines = []  # ADDED
    for label, path, expected in (("Run A", args.run_a_model, betas), ("Run B", args.run_b_model, betas_x)):  # ADDED
        if path is None:  # ADDED
            continue  # ADDED
        saved = DDPMScheduler.from_pretrained(path, subfolder="scheduler").betas.detach().cpu().numpy()  # ADDED
        err_a = float(np.max(np.abs(saved - betas)))  # ADDED
        err_b = float(np.max(np.abs(saved - betas_x)))  # ADDED
        ok = bool(np.allclose(saved, expected, rtol=1e-5, atol=1e-12))  # ADDED
        lines.append(f"{label} ({path}): betas[0]={saved[0]:.3e} betas[-1]={saved[-1]:.3e} "  # ADDED
                     f"max|diff vs beta_t|={err_a:.3e} max|diff vs beta_t^{EXPONENT_X:.2f}|={err_b:.3e} "  # ADDED
                     f"-> matches expected schedule: {ok}")  # ADDED
    lines.append(json.dumps(summary, indent=2))  # ADDED
    (args.output_dir / "schedule_verification.txt").write_text("\n".join(lines) + "\n")  # ADDED
    print("\n".join(lines))  # ADDED


if __name__ == "__main__":
    main()
