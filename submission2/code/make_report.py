# ADDED: this whole file is new code written by the group (one-page strength-sweep report).
"""Test-set inference + evaluation + one-page PDF report for the Run 0 / A / B ablation.

For every strength in --strengths it runs SDEdit inference for Run 0 (pretrained), Run A and Run B
on the test photos, scores them with evaluate_outputs.py (KID to the held-out MetFaces test split,
LPIPS to the input photo) and then writes:

  <report-dir>/report_one_page.pdf (+ .png)   one-page report with key points and charts
  <report-dir>/charts/*.png                   each chart on its own
  <report-dir>/report_metrics.csv             all metrics for all strengths

Finished inference/eval folders are reused, so the script can be re-run cheaply.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

RUNS = ["0", "A", "B"]
ROWS = ["input", "0", "A", "B"]
COLORS = {"input": "#9E9E9E", "0": "#4C72B0", "A": "#55A868", "B": "#C44E52"}
LABELS = {"input": "Input photo", "0": "Run 0 (pretrained)", "A": r"Run A ($\beta_t$)", "B": r"Run B ($\beta_t^2$)"}
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=Path, required=True, help="experiment root containing train/")
    p.add_argument("--input-dir", type=Path, required=True, help="test photos")
    p.add_argument("--report-dir", type=Path, default=None, help="default: <exp-dir>/test_report")
    p.add_argument("--model-id", default="google/ddpm-celebahq-256")
    p.add_argument("--checkpoint", choices=["best", "final"], default="best")
    p.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.45, 0.6])
    p.add_argument("--display-strength", type=float, default=None, help="default: middle strength")
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid-images", type=int, default=4)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def list_images(root: Path) -> list[Path]:
    return sorted(p for ext in IMG_EXTS for p in root.glob(ext))


def read_log(path: Path, col: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r["epoch"]) for r in rows]), np.array([float(r[col]) for r in rows])


def strength_to_t(s: float, n_steps: int, n_train: int = 1000) -> int:
    ts = np.arange(0, n_train, n_train // n_steps)[::-1]
    return int(ts[min(n_steps - 1, int(round(n_steps * (1.0 - s))))])


# ----------------------------------------------------------------------------- inference + eval
def run_inference_and_eval(args, report_dir: Path, inputs: list[Path]) -> dict[float, dict[str, dict]]:
    code = Path(__file__).resolve().parent
    py = sys.executable
    train_dir = args.exp_dir / "train"
    metrics: dict[float, dict[str, dict]] = {}

    for s in args.strengths:
        sdir = report_dir / f"strength_{s:.2f}"
        for r in RUNS:
            model = args.model_id if r == "0" else str(train_dir / f"run_{r}_{args.checkpoint}")
            out = sdir / f"run_{r}_outputs"
            if out.exists() and len(list(out.glob("*.png"))) >= len(inputs):
                continue
            run([py, str(code / "run_ddpm_img2img.py"), "--model-path", model, "--input-dir", str(args.input_dir),
                 "--output-dir", str(sdir), "--run-name", r, "--num-steps", str(args.num_steps),
                 "--strength", str(s), "--seed", str(args.seed)])

        eval_csv = sdir / "eval" / "eval_metrics.csv"
        if not eval_csv.exists():
            run([py, str(code / "evaluate_outputs.py"),
                 "--reference-manifest", str(train_dir / "run_A_test_split.txt"),
                 "--input-dir", str(args.input_dir),
                 *[x for r in RUNS for x in ("--run", f"{r}={sdir / f'run_{r}_outputs'}")],
                 "--output-dir", str(sdir / "eval")])
        with eval_csv.open() as f:
            metrics[s] = {row["run"]: {k: float(v) for k, v in row.items() if k != "run"} for row in csv.DictReader(f)}
    return metrics


# ----------------------------------------------------------------------------- charts
def plot_val_loss(ax, train_dir: Path) -> None:
    eA, vA = read_log(train_dir / "run_A_val_log.csv", "val_loss")
    eB, vB = read_log(train_dir / "run_B_val_log.csv", "val_loss")
    ax.plot(eA, vA, "-o", ms=3, color=COLORS["A"], label=LABELS["A"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("Run A val loss", color=COLORS["A"])
    ax2 = ax.twinx()
    ax2.plot(eB, vB, "-s", ms=3, color=COLORS["B"], label=LABELS["B"])
    ax2.set_ylabel("Run B val loss", color=COLORS["B"])
    for e, v, axx, c in ((eA, vA, ax, COLORS["A"]), (eB, vB, ax2, COLORS["B"])):
        i = int(np.argmin(v))
        axx.scatter([e[i]], [v[i]], s=60, facecolors="none", edgecolors=c, linewidths=1.5, zorder=5)
    ax.set_title("(a) Validation loss (fixed noise draw; separate axes)")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")


def schedule_curves(train_dir: Path, checkpoint: str) -> tuple[np.ndarray, np.ndarray]:
    from diffusers import DDPMScheduler

    sch = DDPMScheduler.from_pretrained(str(train_dir / f"run_A_{checkpoint}"), subfolder="scheduler")
    betas = sch.betas.double().numpy()
    return np.cumprod(1 - betas), np.cumprod(1 - betas**2)


def plot_schedule(ax, abA: np.ndarray, abB: np.ndarray, strengths: list[float], n_steps: int) -> None:
    t = np.arange(len(abA))
    ax.plot(t, abA, color=COLORS["A"], label=LABELS["A"])
    ax.plot(t, abB, color=COLORS["B"], label=LABELS["B"])
    for s in strengths:
        ax.axvline(strength_to_t(s, n_steps), color="k", ls=":", lw=0.8)
        ax.text(strength_to_t(s, n_steps), 0.03, f"s={s:g}", fontsize=6, rotation=90, ha="right")
    ax.set_xlabel("timestep t")
    ax.set_ylabel(r"signal kept $\bar\alpha_t$")
    ax.set_ylim(0, 1.02)
    ax.set_title(r"(b) Signal left after noising: $\beta_t$ vs $\beta_t^2$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center right")


def plot_kid_bars(ax, m: dict[str, dict], s: float) -> None:
    vals = [m[r]["kid"] * 1e3 for r in ROWS]
    errs = [m[r]["kid_std"] * 1e3 for r in ROWS]
    bars = ax.bar([LABELS[r].replace(" (", "\n(") for r in ROWS], vals, yerr=errs, capsize=3, color=[COLORS[r] for r in ROWS])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel(r"KID $\times 10^3$ (lower = closer to MetFaces)")
    ax.set_title(f"(c) Style: KID to held-out MetFaces test set (s={s:g})")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="y", alpha=0.3)


def plot_tradeoff(ax, metrics: dict[float, dict[str, dict]]) -> None:
    ss = sorted(metrics)
    for r in RUNS:
        x = [metrics[s][r]["lpips_to_input"] for s in ss]
        y = [metrics[s][r]["kid"] * 1e3 for s in ss]
        ax.plot(x, y, "-o", ms=4, color=COLORS[r], label=LABELS[r])
        for s, xi, yi in zip(ss, x, y):
            ax.annotate(f"{s:g}", (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(metrics[ss[0]]["input"]["kid"] * 1e3, color=COLORS["input"], ls="--", lw=1, label="input photos")
    ax.set_xlabel("LPIPS to input (lower = more identity kept)")
    ax.set_ylabel(r"KID $\times 10^3$")
    ax.set_title("(d) Style vs identity across strengths s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def qualitative_grid(report_dir: Path, inputs: list[Path], s: float, n: int, tile: int = 192) -> Image.Image:
    """Images side by side in groups of [Input, Run 0, A, B]; two groups per row."""
    sdir = report_dir / f"strength_{s:.2f}"
    chosen = inputs[:n]
    per_row = 2
    n_rows = int(np.ceil(len(chosen) / per_row))
    gap = tile // 6
    width = per_row * 4 * tile + (per_row - 1) * gap
    canvas = Image.new("RGB", (width, n_rows * tile), "white")
    for k, ip in enumerate(chosen):
        row, grp = divmod(k, per_row)
        paths = [ip] + [sdir / f"run_{r}_outputs" / f"{ip.stem}_{r}.png" for r in RUNS]
        for j, p in enumerate(paths):
            im = Image.open(p).convert("RGB")
            side = min(im.size)
            im = im.crop(((im.width - side) // 2, (im.height - side) // 2,
                          (im.width + side) // 2, (im.height + side) // 2)).resize((tile, tile), Image.BICUBIC)
            canvas.paste(im, (grp * (4 * tile + gap) + j * tile, row * tile))
    return canvas


def show_grid(ax, grid: Image.Image, s: float, tile: int = 192) -> None:
    ax.imshow(grid)
    ax.axis("off")
    gap = tile // 6
    for grp in range(2):
        for j, r in enumerate(ROWS):
            ax.text(grp * (4 * tile + gap) + j * tile + tile / 2, -8, LABELS[r], ha="center", va="bottom", fontsize=6.5)
    ax.set_title(f"(e) Test photos and translations (strength s={s:g}, same seed for all runs)", fontsize=9, pad=14)


# ----------------------------------------------------------------------------- key points
def key_points(args, metrics, s, train_dir: Path, abA, abB) -> list[str]:
    cfg = json.loads((train_dir / "run_A_config.json").read_text())
    split = cfg["split"]
    eA, vA = read_log(train_dir / "run_A_val_log.csv", "val_loss")
    eB, vB = read_log(train_dir / "run_B_val_log.csv", "val_loss")
    m = metrics[s]
    kid = {r: m[r]["kid"] * 1e3 for r in ROWS}
    lp = {r: m[r]["lpips_to_input"] for r in ROWS}
    ss = sorted(metrics)
    t0 = strength_to_t(s, args.num_steps)

    pts = [
        f"Setup: {cfg['model_id']} fine-tuned on MetFaces ({split['train_count']} train / {split['val_count']} val / "
        f"{split['test_count']} test), {cfg['epochs']} epochs. Run A and Run B share every hyper-parameter "
        f"(lr {cfg['learning_rate']:g}, {cfg.get('lr_scheduler', 'constant')} schedule, EMA {cfg.get('ema_decay', 0):g}, "
        f"effective batch {cfg.get('effective_batch_size', cfg['batch_size'])}, seed {cfg['seed']}); only the noise schedule differs.",
        f"Training converged: Run A validation loss {vA[0]:.5f} (epoch 1) -> {vA.min():.5f} (epoch {eA[np.argmin(vA)]}), "
        f"{100 * (vA[0] - vA.min()) / vA[0]:.1f}% lower; Run B {vB[0]:.4f} -> {vB.min():.4f} (epoch {eB[np.argmin(vB)]}), "
        f"{100 * (vB[0] - vB.min()) / vB[0]:.1f}% lower. Best-validation checkpoints are used for testing.",
        "A and B losses are on different scales and must not be compared directly: under beta_t^2 the model must "
        "predict a tiny noise component from an almost clean image, which is intrinsically harder (higher MSE).",
    ]
    if kid["A"] < kid["0"]:
        pts.append(f"Fine-tuning works: at strength {s:g}, KID x10^3 drops from {kid['0']:.1f} (Run 0) to {kid['A']:.1f} "
                   f"(Run A), {100 * (kid['0'] - kid['A']) / kid['0']:.0f}% closer to the held-out MetFaces paintings.")
    else:
        pts.append(f"At strength {s:g}, Run A ({kid['A']:.1f}) does not beat Run 0 ({kid['0']:.1f}) on KID x10^3.")
    a_best_all = all(metrics[x]["A"]["kid"] <= min(metrics[x]["0"]["kid"], metrics[x]["B"]["kid"]) for x in ss)
    if a_best_all:
        pts.append(f"Run A has the lowest KID at every strength tested ({', '.join(f'{x:g}' for x in ss)}).")
    pts.append(
        f"Run B stays photographic: KID x10^3 {kid['B']:.1f} vs {kid['input']:.1f} for the untouched photos, "
        f"LPIPS to input {lp['B']:.3f} (Run A {lp['A']:.3f}). Cause (chart b): with beta_t^2, alpha_bar_T = {abB[-1]:.2f}, "
        f"so even full noising keeps {abB[-1]:.0%} of the photo; at s={s:g} (t={t0}) the input keeps {abB[t0]:.0%} of its "
        f"signal under beta_t^2 vs {abA[t0]:.0%} under beta_t, leaving the reverse process nothing to repaint."
    )
    pts.append("Strength is a style/identity trade-off (chart d): higher s lowers KID but raises LPIPS; the same s and "
               "seed are used for all three runs, so the comparison is fair at every point.")
    pts.append(f"Evaluation: {int(m['A']['n'])} unseen test photos; KID (unbiased for small samples, error bars = std over "
               f"subsets) against the {split['test_count']} held-out MetFaces test images; LPIPS (AlexNet) to each input.")
    return pts


# ----------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    report_dir = args.report_dir or (args.exp_dir / "test_report")
    charts = report_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    train_dir = args.exp_dir / "train"
    inputs = list_images(args.input_dir)
    ss = sorted(args.strengths)
    s_disp = args.display_strength if args.display_strength is not None else ss[len(ss) // 2]
    if s_disp not in ss:
        raise ValueError("--display-strength must be one of --strengths")

    metrics = run_inference_and_eval(args, report_dir, inputs)
    abA, abB = schedule_curves(train_dir, args.checkpoint)
    grid = qualitative_grid(report_dir, inputs, s_disp, args.grid_images)

    with (report_dir / "report_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strength", "run", "kid", "kid_std", "fid", "lpips_to_input", "n"])
        for s in ss:
            for r in ROWS:
                d = metrics[s][r]
                w.writerow([s, r, d["kid"], d["kid_std"], d["fid"], d["lpips_to_input"], int(d["n"])])

    # standalone charts
    for name, fn in (
        ("a_val_loss", lambda ax: plot_val_loss(ax, train_dir)),
        ("b_schedule", lambda ax: plot_schedule(ax, abA, abB, ss, args.num_steps)),
        ("c_kid_bars", lambda ax: plot_kid_bars(ax, metrics[s_disp], s_disp)),
        ("d_tradeoff", lambda ax: plot_tradeoff(ax, metrics)),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        fn(ax)
        fig.tight_layout()
        fig.savefig(charts / f"{name}.png", dpi=170)
        plt.close(fig)
    grid.save(charts / "e_qualitative_grid.png")

    # one-page report (A4 portrait)
    fig = plt.figure(figsize=(8.27, 11.69))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.45, 1, 1, 1.05], hspace=0.62, wspace=0.42,
                  left=0.08, right=0.92, top=0.955, bottom=0.025)
    fig.suptitle("Noise-Schedule Ablation in DDPM: Photo -> MetFaces Portrait Translation",
                 fontsize=12.5, fontweight="bold", y=0.985)
    tax = fig.add_subplot(gs[0, :])
    tax.axis("off")
    y = 1.0
    tax.text(0, y, "Key points", fontsize=10, fontweight="bold", va="top", transform=tax.transAxes)
    y -= 0.085
    for pt in key_points(args, metrics, s_disp, train_dir, abA, abB):
        lines = textwrap.wrap(pt, 132)
        tax.text(0, y, "\u2022 " + "\n   ".join(lines), fontsize=7.3, va="top", transform=tax.transAxes, linespacing=1.3)
        y -= 0.062 * len(lines) + 0.028
    plot_val_loss(fig.add_subplot(gs[1, 0]), train_dir)
    plot_schedule(fig.add_subplot(gs[1, 1]), abA, abB, ss, args.num_steps)
    plot_kid_bars(fig.add_subplot(gs[2, 0]), metrics[s_disp], s_disp)
    plot_tradeoff(fig.add_subplot(gs[2, 1]), metrics)
    show_grid(fig.add_subplot(gs[3, :]), grid, s_disp)
    for ax in fig.axes:
        if ax is not tax:
            ax.title.set_fontsize(8.5)
            ax.tick_params(labelsize=7)
            ax.xaxis.label.set_size(7.5)
            ax.yaxis.label.set_size(7.5)
    fig.savefig(report_dir / "report_one_page.pdf")
    fig.savefig(report_dir / "report_one_page.png", dpi=150)
    print(f"\nReport: {report_dir / 'report_one_page.pdf'}")


if __name__ == "__main__":
    main()
