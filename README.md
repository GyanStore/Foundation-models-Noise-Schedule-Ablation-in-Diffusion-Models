# Foundation-models-Noise-Schedule-Ablation-in-Diffusion-Models

## Project goal
This project implements **noise schedule ablation in diffusion models** for image-to-image translation.

**Task:** toonification / photographic-to-artistic portrait translation.
- Input domain: real human face photographs.
- Output domain: painterly "Old Masters"-style portraits.
- Target artistic domain: MetFaces (pre-20th-century portrait artworks).

## Assignment experiment design
The assignment compares three runs:

- **Run 0 (baseline):** pretrained model inference only, no fine-tuning.
- **Run A:** fine-tune with native scheduler parameterization \(\beta_t\).
- **Run B:** fine-tune with power-transformed schedule \(\beta_t^x\), with `x = 2.00`.

For Run A and Run B, all hyperparameters are kept identical except the schedule.

## Repository structure

- `submission1/`
  - `Phase1_Declaration.pdf` — phase-1 declaration submitted for model/task/dataset/schedule choice.

- `submission2/code/`
  - `train_ddpm_metfaces.py` — split-aware A/B fine-tuning pipeline with:
    - train/val/test split manifests,
    - train/validation CSV logs,
    - epoch checkpoints,
    - automatic checkpoint sample generation.
  - `run_ddpm_img2img.py` — image-to-image inference (SDEdit-style) with optional reverse trajectory capture.
  - `analyze_schedule.py` — exports schedule values and plots `beta_t` vs `beta_t^x`.
  - `run_full_experiment.py` — one-command orchestration for schedule analysis + A/B training + Run 0/A/B inference.

## Environment
Python 3.12 with PyTorch + Diffusers stack.

Core libraries:
- `torch`
- `diffusers`
- `transformers`
- `accelerate`
- `torchvision`
- `Pillow`
- `numpy`, `pandas`, `matplotlib`, `tqdm`

## How to run

### 1) Run everything end-to-end (recommended)
```bash
python submission2/code/run_full_experiment.py \
  --dataset-dir <path_to_metfaces_train_images> \
  --input-dir <path_to_real_face_inputs> \
  --output-dir <output_root> \
  --epochs 20 \
  --batch-size 8 \
  --num-workers 4 \
  --num-steps 100 \
  --strength 0.60
```

### 2) Or run components manually

#### (a) Schedule analysis
```bash
python submission2/code/analyze_schedule.py \
  --model-id google/ddpm-celebahq-256 \
  --output-dir <output_root>/schedule
```

#### (b) Fine-tune Run A / Run B
```bash
python submission2/code/train_ddpm_metfaces.py \
  --run-name A \
  --dataset-dir <path_to_metfaces_train_images> \
  --output-dir <output_root>/train \
  --epochs 20 --batch-size 8 --learning-rate 1e-5 --weight-decay 1e-4 \
  --checkpoint-input-dir <path_to_real_face_inputs>

python submission2/code/train_ddpm_metfaces.py \
  --run-name B \
  --dataset-dir <path_to_metfaces_train_images> \
  --output-dir <output_root>/train \
  --epochs 20 --batch-size 8 --learning-rate 1e-5 --weight-decay 1e-4 \
  --checkpoint-input-dir <path_to_real_face_inputs>
```

#### (c) Inference for Run 0 / A / B
```bash
# Run 0
python submission2/code/run_ddpm_img2img.py \
  --model-path google/ddpm-celebahq-256 \
  --input-dir <path_to_real_face_inputs> \
  --output-dir <output_root>/inference \
  --run-name 0 --num-steps 100 --strength 0.60 --save-trajectory

# Run A final
python submission2/code/run_ddpm_img2img.py \
  --model-path <output_root>/train/run_A_final \
  --input-dir <path_to_real_face_inputs> \
  --output-dir <output_root>/inference \
  --run-name A --num-steps 100 --strength 0.60 --save-trajectory

# Run B final
python submission2/code/run_ddpm_img2img.py \
  --model-path <output_root>/train/run_B_final \
  --input-dir <path_to_real_face_inputs> \
  --output-dir <output_root>/inference \
  --run-name B --num-steps 100 --strength 0.60 --save-trajectory
```

## Generated artifacts
The pipeline produces assignment-ready artifacts:

- `schedule/schedule_values.csv`, `schedule/schedule_plot.png`
- `train/run_A_train_log.csv`, `train/run_B_train_log.csv`
- `train/run_A_val_log.csv`, `train/run_B_val_log.csv`
- split manifests: train/val/test file lists per run
- checkpoint directories and checkpoint-sample images
- final fine-tuned models: `run_A_final`, `run_B_final`
- inference outputs and reverse-process trajectory images for Run 0/A/B

## Notes
- The implementation uses the declared exponent `x = 2.00`.
- For strict Run A vs Run B comparison, keep all other settings identical.
