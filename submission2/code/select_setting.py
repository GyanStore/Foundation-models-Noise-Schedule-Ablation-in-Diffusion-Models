"""ADDED: choose the ILVR inference setting on the DEV photos only, then freeze it.  # ADDED

Grid: strength x ILVR factor N, evaluated with the fine-tuned Run A final model on the  # ADDED
20 FFHQ dev photos. Rule: lowest KID to the MetFaces test split among settings whose  # ADDED
LPIPS-to-input <= --max-lpips (structure kept). The test photos are never used here.  # ADDED
"""  # ADDED
from __future__ import annotations  # ADDED

import argparse  # ADDED
import csv  # ADDED
import json  # ADDED
import subprocess  # ADDED
import sys  # ADDED
from pathlib import Path  # ADDED

CODE_DIR = Path(__file__).resolve().parent  # ADDED


def main() -> None:  # ADDED
    p = argparse.ArgumentParser()  # ADDED
    p.add_argument("--model-path", type=Path, required=True)  # ADDED
    p.add_argument("--dev-dir", type=Path, required=True)  # ADDED
    p.add_argument("--reference-manifest", type=Path, required=True)  # ADDED
    p.add_argument("--output-dir", type=Path, required=True)  # ADDED
    p.add_argument("--strengths", type=float, nargs="+", default=[0.6, 0.8, 1.0])  # ADDED
    p.add_argument("--factors", type=int, nargs="+", default=[8, 16])  # ADDED
    p.add_argument("--num-steps", type=int, default=50)  # ADDED
    p.add_argument("--max-lpips", type=float, default=0.45)  # ADDED
    p.add_argument("--seed", type=int, default=42)  # ADDED
    args = p.parse_args()  # ADDED
    args.output_dir.mkdir(parents=True, exist_ok=True)  # ADDED

    rows = []  # ADDED
    for s in args.strengths:  # ADDED
        for n in args.factors:  # ADDED
            tag = f"s{s:g}_N{n}"  # ADDED
            out = args.output_dir / tag  # ADDED
            subprocess.run([sys.executable, str(CODE_DIR / "run_ddpm_img2img.py"),  # ADDED
                            "--model-path", str(args.model_path), "--input-dir", str(args.dev_dir),  # ADDED
                            "--output-dir", str(out), "--run-name", "A", "--num-steps", str(args.num_steps),  # ADDED
                            "--strength", str(s), "--method", "ilvr", "--ilvr-factor", str(n),  # ADDED
                            "--seed", str(args.seed)], check=True)  # ADDED
            subprocess.run([sys.executable, str(CODE_DIR / "evaluate_outputs.py"),  # ADDED
                            "--reference-manifest", str(args.reference_manifest), "--input-dir", str(args.dev_dir),  # ADDED
                            "--run", f"A={out / 'run_A_outputs'}", "--output-dir", str(out / "eval")], check=True)  # ADDED
            with (out / "eval" / "eval_metrics.csv").open() as f:  # ADDED
                r = [x for x in csv.DictReader(f) if x["run"] == "A"][0]  # ADDED
            rows.append({"strength": s, "ilvr_factor": n, "kid": float(r["kid"]),  # ADDED
                         "kid_std": float(r["kid_std"]), "lpips_to_input": float(r["lpips_to_input"])})  # ADDED
            print(rows[-1], flush=True)  # ADDED

    with (args.output_dir / "selection_grid.csv").open("w", newline="") as f:  # ADDED
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))  # ADDED
        w.writeheader()  # ADDED
        w.writerows(rows)  # ADDED

    ok = [r for r in rows if r["lpips_to_input"] <= args.max_lpips]  # ADDED
    best = min(ok or rows, key=lambda r: r["kid"])  # ADDED
    selected = {"method": "ilvr", "strength": best["strength"], "ilvr_factor": best["ilvr_factor"],  # ADDED
                "num_steps": args.num_steps, "seed": args.seed, "max_lpips": args.max_lpips,  # ADDED
                "constraint_met": bool(ok), "dev_kid": best["kid"], "dev_lpips": best["lpips_to_input"],  # ADDED
                "selected_on": str(args.dev_dir), "model": str(args.model_path)}  # ADDED
    (args.output_dir / "selected_setting.json").write_text(json.dumps(selected, indent=2))  # ADDED
    print("SELECTED", selected)  # ADDED


if __name__ == "__main__":  # ADDED
    main()  # ADDED
