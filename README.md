# Foundation-models-Noise-Schedule-Ablation-in-Diffusion-Models

## Project goal
This project implements **noise schedule ablation in diffusion models** for image-to-image translation.

**Task:** toonification / photographic-to-artistic portrait translation.
- Input domain: real human face photographs.
- Output domain: painterly "Old Masters"-style portraits.
- Target artistic domain: MetFaces (pre-20th-century portrait artworks).

## Assignment experiment design
- **Model:** `google/ddpm-celebahq-256` (DDPM, 256x256, pretrained on CelebA-HQ photos).
- **Native noise parameter:** theta_t = beta_t, linear from 1e-4 to 0.02, T = 1000.
- **Exponent:** `EXPONENT_X = 2.00` (named constant at the top of `train_ddpm_metfaces.py` and `analyze_schedule.py`).
- **Run 0 (baseline):** pretrained model, inference only, no fine-tuning.
- **Run A:** fine-tune on MetFaces with the native schedule beta_t.
- **Run B:** fine-tune on MetFaces with beta_t^2. Every other setting is identical to Run A.
- **Seed:** 42 for every split, training run and inference run.

## Code markers
All scripts in `submission2/code/` are new files written by the group on top of the unchanged `diffusers` library
(the "original" model and scheduler code). Each file carries an `# ADDED` header. The lines that change the native
noise schedule (beta_t -> beta_t^x) are tagged `# MODIFIED`. They are in `train_ddpm_metfaces.py`
(`EXPONENT_X`, `build_scheduler_for_run`) and `analyze_schedule.py`. Lines added for the ILVR guidance and the
float32 zero-noise fix in `run_ddpm_img2img.py` are tagged `# ADDED` individually, and the `scheduler.step` call
that now receives the seeded generator is tagged `# MODIFIED`.

## Repository structure
- `submission1/Phase1_Declaration.pdf`: Phase 1 declaration (task, model, dataset, schedule).
- `submission2/code/`
  - `train_ddpm_metfaces.py`: Run A / Run B fine-tuning. Covers the 80/10/10 split manifests, train and val CSV logs,
    EMA weights, the cosine LR schedule, deterministic validation, 5 epoch checkpoints, and best and final models.
  - `run_ddpm_img2img.py`: image-to-image inference with SDEdit or ILVR (`--method`) and optional
    reverse-trajectory capture (5 equally spaced timesteps).
  - `analyze_schedule.py`: plots theta_t and theta_t^x, alpha_bar_t and log-SNR, and verifies that the saved
    Run A / Run B schedulers use beta_t and beta_t^x.
  - `run_full_experiment.py`: one command for schedule analysis, A/B training and Run 0/A/B inference.
  - `prepare_ffhq_faces.py`: downloads the held-out FFHQ test (30) and dev (20) photos.
  - `select_setting.py`: picks the inference setting on the dev photos only and freezes it in `selected_setting.json`.
  - `run_final_eval.py`: final Run 0/A/B test (ILVR with the frozen setting, plus SDEdit) and per-checkpoint evaluation.
  - `evaluate_outputs.py`: computes KID/FID to the held-out MetFaces test split and LPIPS to the input photo.
  - `make_figures.py`: builds every report figure and the results table.
  - `make_report.py`: one-page strength-sweep summary (earlier SDEdit analysis).
- `submission2/report/`: `build_report.py` and the final report (`Phase2_Report.docx` and `.pdf`).

## Environment
Measured on macOS 15.7 with an Apple M3 Max, using the PyTorch MPS backend (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
CUDA is used automatically when available.

| package | version |
|---|---|
| Python | 3.12.4 |
| torch | 2.5.1 |
| torchvision | 0.20.1 |
| diffusers | 0.30.3 |
| transformers | 4.46.3 |
| torchmetrics | 1.9.0 (with torch-fidelity) |
| lpips | 0.1.4 |
| numpy | 2.4.2 |
| Pillow | 10.4.0 |
| pandas, matplotlib, scipy, datasets, python-docx | any recent |

## Data
| set | source | size | use |
|---|---|---|---|
| MetFaces | `huggan/metfaces` on Hugging Face (mirror of https://github.com/NVlabs/metfaces-dataset) | 1,336 | split 80/10/10 with seed 42 into 1,069 train, 134 val and 133 test images. The 133 test images are the KID/FID reference. |
| FFHQ test photos | `bitmind/ffhq-256` shard 0, seed 42 | 30 | final test inputs only |
| FFHQ dev photos | same shard, disjoint from the test photos | 20 | inference-setting selection only |

```bash
# MetFaces -> data/metfaces_real_hf_images/0000.png ...
python -c "from datasets import load_dataset; import os; os.makedirs('data/metfaces_real_hf_images', exist_ok=True); [ex['image'].convert('RGB').save(f'data/metfaces_real_hf_images/{i:04d}.png') for i, ex in enumerate(load_dataset('huggan/metfaces', split='train'))]"
# FFHQ test (30) + dev (20) photos
python submission2/code/prepare_ffhq_faces.py
```

## Exact commands (the runs reported)
All commands run from the repository root. `E=outputs_real_metfaces_mps_tuned20`.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
E=outputs_real_metfaces_mps_tuned20

# 1) Schedule analysis and verification of the saved schedulers
python submission2/code/analyze_schedule.py --model-id google/ddpm-celebahq-256 --output-dir $E/schedule \
  --run-a-model $E/train/run_A_final --run-b-model $E/train/run_B_final

# 2) Run A and Run B fine-tuning (identical except --run-name)
for RUN in A B; do
python submission2/code/train_ddpm_metfaces.py --run-name $RUN --dataset-dir data/metfaces_real_hf_images \
  --output-dir $E/train --model-id google/ddpm-celebahq-256 --image-size 256 --epochs 20 --batch-size 4 \
  --learning-rate 2e-05 --weight-decay 0.0001 --num-workers 0 --seed 42 \
  --checkpoint-input-dir data/test_faces_internet_30 --checkpoint-num-images 1 --checkpoint-num-steps 30 \
  --checkpoint-strength 0.5 --lr-scheduler cosine --lr-warmup-steps 100 --grad-accum-steps 2 --grad-clip 1.0 \
  --ema-decay 0.999 --val-repeats 3 --hflip
done

# 3) Choose the inference setting on the 20 dev photos (Run A final), saved to selected_setting.json
python submission2/code/select_setting.py --model-path $E/train/run_A_final --dev-dir data/dev_faces_ffhq_20 \
  --reference-manifest $E/train/run_A_test_split.txt --output-dir $E/final_eval/selection

# 4) Final test: Run 0 / A / B with the frozen ILVR setting plus SDEdit, and every saved checkpoint
python submission2/code/run_final_eval.py --exp-dir $E --test-dir data/test_faces_ffhq_30 \
  --selected $E/final_eval/selection/selected_setting.json --stage all

# 5) Report figures and results table
python submission2/code/make_figures.py --exp-dir $E --test-dir data/test_faces_ffhq_30

# 6) Report (.docx, numbers read from the result files) and PDF export
python submission2/report/build_report.py --exp-dir $E
soffice --headless --convert-to pdf --outdir submission2/report submission2/report/Phase2_Report.docx
```

The frozen inference setting is ILVR with strength 0.8 (start timestep t = 780), low-pass factor N = 16,
50 reverse steps and seed 42. Guidance stops below t = 200. All sampling noise (the initial noise, the
per-step DDPM posterior noise and the ILVR reference noise) comes from one generator seeded with `--seed`,
so re-running a command reproduces the same images.

Single-run inference (the same command for each run, changing only `--model-path` and `--run-name`):
```bash
# Run 0
python submission2/code/run_ddpm_img2img.py --model-path google/ddpm-celebahq-256 --input-dir data/test_faces_ffhq_30 \
  --output-dir $E/final_eval/test_ilvr --run-name 0 --num-steps 50 --strength 0.8 --method ilvr --ilvr-factor 16 \
  --seed 42 --save-trajectory
# Run A
python submission2/code/run_ddpm_img2img.py --model-path $E/train/run_A_final --input-dir data/test_faces_ffhq_30 \
  --output-dir $E/final_eval/test_ilvr --run-name A --num-steps 50 --strength 0.8 --method ilvr --ilvr-factor 16 \
  --seed 42 --save-trajectory
# Run B
python submission2/code/run_ddpm_img2img.py --model-path $E/train/run_B_final --input-dir data/test_faces_ffhq_30 \
  --output-dir $E/final_eval/test_ilvr --run-name B --num-steps 50 --strength 0.8 --method ilvr --ilvr-factor 16 \
  --seed 42 --save-trajectory
```

## Generated artifacts (under `$E`)
- `schedule/`: `schedule_values.csv`, `schedule_plot.png`, `schedule_alphabar_snr.png`, `schedule_verification.txt`
- `train/`
  - `run_{A,B}_train_log.csv` (epoch, train_loss, lr) and `run_{A,B}_val_log.csv`
  - split manifests
  - checkpoints `run_{A,B}_epoch_{001,006,010,015,020}`
  - `run_{A,B}_final` and `run_{A,B}_best`
- `final_eval/`
  - `selection/` (dev grid and `selected_setting.json`)
  - `test_ilvr/` and `test_sdedit/` (outputs, trajectories, `eval/eval_metrics.csv`)
  - `checkpoints_{A,B}/`
  - `kid_vs_epoch.csv`
- `report_figures/`: all report figures and `results_table.csv`
