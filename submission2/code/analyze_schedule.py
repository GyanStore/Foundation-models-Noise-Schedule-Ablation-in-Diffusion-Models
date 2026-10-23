from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diffusers import DDPMPipeline

EXPONENT_X = 2.00
DEFAULT_MODEL_ID = "google/ddpm-celebahq-256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = DDPMPipeline.from_pretrained(args.model_id)
    betas = pipe.scheduler.betas.detach().cpu().numpy()
    betas_x = np.power(betas, EXPONENT_X)

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


if __name__ == "__main__":
    main()
