"""ADDED: build every figure and table used in the Phase 2 report from the saved run artifacts.  # ADDED

Inputs (all under --exp-dir): train/*_log.csv, schedule/schedule_values.csv, final_eval/selection/,  # ADDED
final_eval/test_ilvr/, final_eval/test_sdedit/, final_eval/checkpoints_{A,B}/, final_eval/kid_vs_epoch.csv.  # ADDED
Output: --exp-dir/report_figures/*.png and results_table.csv.  # ADDED
"""  # ADDED
from __future__ import annotations  # ADDED

import argparse  # ADDED
from pathlib import Path  # ADDED

import matplotlib  # ADDED

matplotlib.use("Agg")  # ADDED
import matplotlib.pyplot as plt  # ADDED
import pandas as pd  # ADDED
from PIL import Image  # ADDED

EXPONENT_X = 2.00  # ADDED
RUN_LABEL = {"0": "Run 0 (pretrained)", "A": "Run A (beta_t)", "B": f"Run B (beta_t^{EXPONENT_X:.0f})"}  # ADDED
COLORS = {"0": "#7f7f7f", "A": "#1f77b4", "B": "#d62728"}  # ADDED


def img(path: Path) -> Image.Image:  # ADDED
    return Image.open(path).convert("RGB").resize((256, 256))  # ADDED


def grid(cells: list[list[Path]], row_labels: list[str], col_labels: list[str], out: Path, title: str) -> None:  # ADDED
    nr, nc = len(cells), len(cells[0])  # ADDED
    fig, axes = plt.subplots(nr, nc, figsize=(1.9 * nc + 0.6, 1.9 * nr + 0.5), squeeze=False)  # ADDED
    for i in range(nr):  # ADDED
        for j in range(nc):  # ADDED
            ax = axes[i][j]  # ADDED
            ax.imshow(img(cells[i][j]))  # ADDED
            ax.set_xticks([])  # ADDED
            ax.set_yticks([])  # ADDED
            if i == 0:  # ADDED
                ax.set_title(col_labels[j], fontsize=9)  # ADDED
            if j == 0:  # ADDED
                ax.set_ylabel(row_labels[i], fontsize=9)  # ADDED
    fig.suptitle(title, fontsize=11)  # ADDED
    fig.tight_layout()  # ADDED
    fig.savefig(out, dpi=150)  # ADDED
    plt.close(fig)  # ADDED


def main() -> None:  # ADDED
    p = argparse.ArgumentParser()  # ADDED
    p.add_argument("--exp-dir", type=Path, required=True)  # ADDED
    p.add_argument("--test-dir", type=Path, required=True)  # ADDED
    p.add_argument("--samples", nargs="+", default=["ffhq_test_00", "ffhq_test_01", "ffhq_test_02"])  # ADDED
    args = p.parse_args()  # ADDED
    exp, fe = args.exp_dir, args.exp_dir / "final_eval"  # ADDED
    out = exp / "report_figures"  # ADDED
    out.mkdir(parents=True, exist_ok=True)  # ADDED
    S = args.samples  # ADDED

    # 1. Schedule: theta and theta^x on the same axes, alpha_bar and log-SNR.  # ADDED
    sv = pd.read_csv(exp / "schedule" / "schedule_values.csv")  # ADDED
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))  # ADDED
    axes[0].plot(sv.t, sv.beta_t, color=COLORS["A"], label="theta_t = beta_t (Run A)")  # ADDED
    axes[0].plot(sv.t, sv[f"beta_t_pow_{EXPONENT_X:.2f}"], color=COLORS["B"], label="theta_t^2 (Run B)")  # ADDED
    axes[0].set(title="Noise parameter theta_t and theta_t^x", xlabel="timestep t", ylabel="theta")  # ADDED
    axes[1].plot(sv.t, sv.alpha_bar_A, color=COLORS["A"], label="Run A")  # ADDED
    axes[1].plot(sv.t, sv.alpha_bar_B, color=COLORS["B"], label="Run B")  # ADDED
    axes[1].set(title="Signal kept: alpha_bar_t", xlabel="timestep t", ylim=(0, 1.02))  # ADDED
    axes[2].plot(sv.t, sv.log_snr_A, color=COLORS["A"], label="Run A")  # ADDED
    axes[2].plot(sv.t, sv.log_snr_B, color=COLORS["B"], label="Run B")  # ADDED
    axes[2].set(title="log SNR_t", xlabel="timestep t")  # ADDED
    for ax in axes:  # ADDED
        ax.grid(alpha=0.3)  # ADDED
        ax.legend(fontsize=8)  # ADDED
    fig.tight_layout()  # ADDED
    fig.savefig(out / "fig_schedule.png", dpi=150)  # ADDED
    plt.close(fig)  # ADDED

    # 2. Training loss of A and B on a single plot (+ validation loss).  # ADDED
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))  # ADDED
    for r in ("A", "B"):  # ADDED
        tr = pd.read_csv(exp / "train" / f"run_{r}_train_log.csv")  # ADDED
        va = pd.read_csv(exp / "train" / f"run_{r}_val_log.csv")  # ADDED
        axes[0].plot(tr.epoch, tr.train_loss, "o-", ms=3, color=COLORS[r], label=f"{RUN_LABEL[r]} train")  # ADDED
        axes[0].plot(va.epoch, va.val_loss, "s--", ms=3, color=COLORS[r], alpha=0.6, label=f"{RUN_LABEL[r]} val")  # ADDED
        axes[1].plot(tr.epoch, tr.train_loss / tr.train_loss.iloc[0], "o-", ms=3, color=COLORS[r], label=RUN_LABEL[r])  # ADDED
    axes[0].set(yscale="log", title="Noise-prediction MSE per epoch (log scale)", xlabel="epoch", ylabel="MSE")  # ADDED
    axes[1].set(title="Train loss relative to epoch 1", xlabel="epoch", ylabel="loss / loss(epoch 1)")  # ADDED
    for ax in axes:  # ADDED
        ax.grid(alpha=0.3)  # ADDED
        ax.legend(fontsize=8)  # ADDED
    fig.tight_layout()  # ADDED
    fig.savefig(out / "fig_loss.png", dpi=150)  # ADDED
    plt.close(fig)  # ADDED

    # 3. Main qualitative comparison: input | Run 0 | A | B (ILVR), plus Run A SDEdit.  # ADDED
    ti, ts = fe / "test_ilvr", fe / "test_sdedit"  # ADDED
    cells = [[args.test_dir / f"{s}.png"] + [ti / f"run_{r}_outputs" / f"{s}_{r}.png" for r in "0AB"]  # ADDED
             + [ts / "run_A_outputs" / f"{s}_A.png"] for s in S]  # ADDED
    grid(cells, S, ["input photo", "Run 0 (ILVR)", "Run A (ILVR)", "Run B (ILVR)", "Run A (SDEdit)"],  # ADDED
         out / "fig_main_comparison.png", "Test photos -> MetFaces style (final models, frozen setting)")  # ADDED

    # 4. Run 0 samples.  # ADDED
    cells = [[args.test_dir / f"{s}.png" for s in S], [ti / "run_0_outputs" / f"{s}_0.png" for s in S]]  # ADDED
    grid(cells, ["input", "Run 0"], S, out / "fig_run0.png", "Run 0: pretrained google/ddpm-celebahq-256, no fine-tuning")  # ADDED

    # 5. Labelled checkpoint samples (5 checkpoints x 3 test photos) for A and B.  # ADDED
    for r in ("A", "B"):  # ADDED
        cdir = fe / f"checkpoints_{r}"  # ADDED
        names = sorted(d.name[len("run_"):-len("_outputs")] for d in cdir.glob(f"run_{r}_e*_outputs"))  # ADDED
        cells = [[cdir / f"run_{n}_outputs" / f"{s}_{n}.png" for s in S] for n in names]  # ADDED
        grid(cells, [f"epoch {int(n.split('_e')[1])}" for n in names], S, out / f"fig_checkpoints_{r}.png",  # ADDED
             f"{RUN_LABEL[r]}: samples at each saved checkpoint")  # ADDED

    # 6. Trajectories x_t at 5 equally spaced reverse steps, final A and B models.  # ADDED
    for r in ("A", "B"):  # ADDED
        tdir = ti / f"run_{r}_trajectories"  # ADDED
        rows = [sorted(tdir.glob(f"{s}_step_*.png")) for s in S]  # ADDED
        cols = [f"t = {int(pth.stem.split('_t_')[1])}" for pth in rows[0]]  # ADDED
        grid(rows, S, cols, out / f"fig_trajectory_{r}.png", f"{RUN_LABEL[r]}: reverse trajectory x_t (final model)")  # ADDED

    # 7. KID vs epoch (checkpoints), Run 0 as reference line.  # ADDED
    ke = pd.read_csv(fe / "kid_vs_epoch.csv")  # ADDED
    ev_i = pd.read_csv(ti / "eval" / "eval_metrics.csv").set_index("run")  # ADDED
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))  # ADDED
    for r in ("A", "B"):  # ADDED
        d = ke[ke.run == r].sort_values("epoch")  # ADDED
        axes[0].errorbar(d.epoch, d.kid * 1e3, yerr=d.kid_std * 1e3, fmt="o-", capsize=3, color=COLORS[r], label=RUN_LABEL[r])  # ADDED
        axes[1].plot(d.epoch, d.lpips_to_input, "o-", color=COLORS[r], label=RUN_LABEL[r])  # ADDED
    for ax, col in ((axes[0], "kid"), (axes[1], "lpips_to_input")):  # ADDED
        scale = 1e3 if col == "kid" else 1.0  # ADDED
        ax.axhline(ev_i.loc["0", col] * scale, color=COLORS["0"], ls="--", label="Run 0")  # ADDED
        ax.axhline(ev_i.loc["input", col] * scale, color="k", ls=":", label="input photos")  # ADDED
        ax.grid(alpha=0.3)  # ADDED
        ax.legend(fontsize=8)  # ADDED
    axes[0].set(title="KID x1e3 to MetFaces test split vs epoch (lower = more painterly)", xlabel="epoch")  # ADDED
    axes[1].set(title="LPIPS to input photo vs epoch (lower = more structure kept)", xlabel="epoch")  # ADDED
    fig.tight_layout()  # ADDED
    fig.savefig(out / "fig_kid_vs_epoch.png", dpi=150)  # ADDED
    plt.close(fig)  # ADDED

    # 8. Results table (test set) for ILVR (headline) and SDEdit (comparison).  # ADDED
    rows = []  # ADDED
    for method, d in (("ILVR (selected)", ti), ("SDEdit s=0.6", ts)):  # ADDED
        ev = pd.read_csv(d / "eval" / "eval_metrics.csv")  # ADDED
        for _, x in ev.iterrows():  # ADDED
            if x.run == "input" and method != "ILVR (selected)":  # ADDED
                continue  # ADDED
            rows.append({"method": "-" if x.run == "input" else method,  # ADDED
                         "run": "input photos" if x.run == "input" else RUN_LABEL[str(x.run)],  # ADDED
                         "KID x1e3": round(x.kid * 1e3, 1), "KID std x1e3": round(x.kid_std * 1e3, 1),  # ADDED
                         "FID": round(x.fid, 1), "LPIPS to input": round(x.lpips_to_input, 3), "n": int(x.n)})  # ADDED
    pd.DataFrame(rows).to_csv(out / "results_table.csv", index=False)  # ADDED

    # 9. Dev-set selection grid (appendix).  # ADDED
    sg = pd.read_csv(fe / "selection" / "selection_grid.csv")  # ADDED
    fig, ax = plt.subplots(figsize=(5.5, 3.8))  # ADDED
    for n, mk in ((8, "o"), (16, "s")):  # ADDED
        d = sg[sg.ilvr_factor == n]  # ADDED
        ax.plot(d.lpips_to_input, d.kid * 1e3, mk + "-", label=f"ILVR N={n}")  # ADDED
        for _, x in d.iterrows():  # ADDED
            ax.annotate(f"s={x.strength:g}", (x.lpips_to_input, x.kid * 1e3), fontsize=7, xytext=(3, 3), textcoords="offset points")  # ADDED
    ax.set(xlabel="LPIPS to input (dev)", ylabel="KID x1e3 (dev)", title="Setting selection on 20 dev photos (Run A)")  # ADDED
    ax.grid(alpha=0.3)  # ADDED
    ax.legend(fontsize=8)  # ADDED
    fig.tight_layout()  # ADDED
    fig.savefig(out / "fig_selection.png", dpi=150)  # ADDED
    plt.close(fig)  # ADDED
    print("figures written to", out)  # ADDED


if __name__ == "__main__":  # ADDED
    main()  # ADDED
