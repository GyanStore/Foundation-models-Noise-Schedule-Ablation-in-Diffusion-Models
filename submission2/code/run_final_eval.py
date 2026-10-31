"""ADDED: final test evaluation with the frozen inference setting (selected on dev photos).  # ADDED

Stage "test":        Run 0 / A / B on the test photos with ILVR (selected setting, trajectories saved)  # ADDED
                     and with plain SDEdit (comparison), then KID/FID/LPIPS for each.  # ADDED
Stage "checkpoints": every saved epoch checkpoint of Run A and Run B on the test photos with the same  # ADDED
                     ILVR setting -> labelled checkpoint samples and a KID-vs-epoch table.  # ADDED
"""  # ADDED
from __future__ import annotations  # ADDED

import argparse  # ADDED
import csv  # ADDED
import json  # ADDED
import subprocess  # ADDED
import sys  # ADDED
from pathlib import Path  # ADDED

CODE_DIR = Path(__file__).resolve().parent  # ADDED
PRETRAINED = "google/ddpm-celebahq-256"  # ADDED


def infer(model: str, inp: Path, out: Path, name: str, setting: dict, method: str, strength: float,  # ADDED
          trajectory: bool) -> Path:  # ADDED
    cmd = [sys.executable, str(CODE_DIR / "run_ddpm_img2img.py"), "--model-path", model,  # ADDED
           "--input-dir", str(inp), "--output-dir", str(out), "--run-name", name,  # ADDED
           "--num-steps", str(setting["num_steps"]), "--strength", str(strength), "--method", method,  # ADDED
           "--ilvr-factor", str(setting["ilvr_factor"]), "--seed", str(setting["seed"])]  # ADDED
    if trajectory:  # ADDED
        cmd.append("--save-trajectory")  # ADDED
    subprocess.run(cmd, check=True)  # ADDED
    return out / f"run_{name}_outputs"  # ADDED


def evaluate(manifest: Path, inp: Path, runs: dict[str, Path], out: Path) -> list[dict]:  # ADDED
    cmd = [sys.executable, str(CODE_DIR / "evaluate_outputs.py"), "--reference-manifest", str(manifest),  # ADDED
           "--input-dir", str(inp), "--output-dir", str(out)]  # ADDED
    for k, v in runs.items():  # ADDED
        cmd += ["--run", f"{k}={v}"]  # ADDED
    subprocess.run(cmd, check=True)  # ADDED
    with (out / "eval_metrics.csv").open() as f:  # ADDED
        return list(csv.DictReader(f))  # ADDED


def main() -> None:  # ADDED
    p = argparse.ArgumentParser()  # ADDED
    p.add_argument("--exp-dir", type=Path, required=True)  # ADDED
    p.add_argument("--test-dir", type=Path, required=True)  # ADDED
    p.add_argument("--selected", type=Path, required=True, help="selected_setting.json from select_setting.py")  # ADDED
    p.add_argument("--sdedit-strength", type=float, default=0.6)  # ADDED
    p.add_argument("--stage", choices=["test", "checkpoints", "all"], default="all")  # ADDED
    args = p.parse_args()  # ADDED

    setting = json.loads(args.selected.read_text())  # ADDED
    train = args.exp_dir / "train"  # ADDED
    manifest = train / "run_A_test_split.txt"  # ADDED
    root = args.exp_dir / "final_eval"  # ADDED
    models = {"0": PRETRAINED, "A": str(train / "run_A_final"), "B": str(train / "run_B_final")}  # ADDED

    if args.stage in ("test", "all"):  # ADDED
        for method, strength in (("ilvr", setting["strength"]), ("sdedit", args.sdedit_strength)):  # ADDED
            out = root / f"test_{method}"  # ADDED
            dirs = {k: infer(m, args.test_dir, out, k, setting, method, strength, trajectory=True)  # ADDED
                    for k, m in models.items()}  # ADDED
            rows = evaluate(manifest, args.test_dir, dirs, out / "eval")  # ADDED
            print(method, strength, rows, flush=True)  # ADDED

    if args.stage in ("checkpoints", "all"):  # ADDED
        rows_out = []  # ADDED
        for run in ("A", "B"):  # ADDED
            ckpts = sorted(d for d in train.glob(f"run_{run}_epoch_*") if d.is_dir() and not d.name.endswith("_samples"))  # ADDED
            out = root / f"checkpoints_{run}"  # ADDED
            dirs = {}  # ADDED
            for ck in ckpts:  # ADDED
                epoch = int(ck.name.split("_")[-1])  # ADDED
                name = f"{run}_e{epoch:03d}"  # ADDED
                dirs[name] = infer(str(ck), args.test_dir, out, name, setting, "ilvr", setting["strength"], trajectory=False)  # ADDED
            for r in evaluate(manifest, args.test_dir, dirs, out / "eval"):  # ADDED
                if r["run"] != "input":  # ADDED
                    rows_out.append({"run": run, "epoch": int(r["run"].split("_e")[1]), "kid": r["kid"],  # ADDED
                                     "kid_std": r["kid_std"], "fid": r["fid"], "lpips_to_input": r["lpips_to_input"]})  # ADDED
        with (root / "kid_vs_epoch.csv").open("w", newline="") as f:  # ADDED
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))  # ADDED
            w.writeheader()  # ADDED
            w.writerows(rows_out)  # ADDED
        print("checkpoints", rows_out, flush=True)  # ADDED


if __name__ == "__main__":  # ADDED
    main()  # ADDED
