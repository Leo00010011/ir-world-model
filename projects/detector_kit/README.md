# Detector kit: train and run our DINO-DETR detector with two commands

> 🇪🇸 Versión en español: [README.es.md](README.es.md)

This folder lets you use the object detector of the `ir-world-model` project without learning
detrex, detectron2, Hydra or the rest of the repo. Two scripts do everything:

```bash
bash projects/detector_kit/train.sh     --modality thermal            # train a detector on FLIR
bash projects/detector_kit/inference.sh --head det_ir --images DIR    # detect objects in your images
```

**What the detector is.** A DINO-DETR head ([detrex](https://github.com/IDEA-Research/detrex)'s
`dino_vitdet`) that does not look at pixels directly. Each image is first turned into features by
a **frozen** DINOv2-Reg ViT-B/14 encoder (+ a PCA to 1152 channels), and only the head is trained.
It predicts 6 classes: `person, bike, car, motor, light, sign`. Two trained detectors exist:

| name | trained on | FLIR val AP | FLIR val AP50 |
|---|---|---|---|
| `det_rgb` | FLIR ADAS v2, RGB camera | 0.302 | 0.573 |
| `det_ir` | FLIR ADAS v2, thermal camera | 0.330 | 0.626 |

**You do not need the compiled CUDA kernels.** detrex normally needs a CUDA op compiled with
`nvcc` (multi-scale deformable attention). If it is missing, both scripts detect that and use
detrex's pure-PyTorch version of the same op. The results are the same; it is just slower and
uses more GPU memory (see [Speed and memory](#speed-and-memory)). The log tells you which one runs:
`deformable attention: cuda` (compiled) or `deformable attention: pytorch` (fallback).

---

## 1. One-time setup

### What you need
- **Linux x86_64** with `git`, `bash` and a C++ compiler (`gcc`). The first run compiles
  detectron2/detrex's C++ code with `gcc`. `nvcc` and the CUDA toolkit are **not** required.
- **Internet** on the first run (Python packages, ~350 MB of encoder weights). On a cluster, do
  the first run on the login node.
- **For training:** an NVIDIA GPU (driver ≥ 525) and the FLIR ADAS v2 dataset.
- **For inference:** a GPU is optional (CPU works, slowly). The FLIR dataset is only needed for `--split`.
- About 10 GB of disk for the Python env (PyTorch is most of it).

### Steps
1. **Fork and clone, with submodules** (detrex and DINO-Foresight are git submodules):
   ```bash
   git clone --recurse-submodules <your fork URL> ir-world-model
   cd ir-world-model
   ```
   If you already cloned without submodules: `git submodule update --init --recursive`.

2. **Create `.env`** at the repo root. It tells the code where things live:
   ```bash
   cp .env.example .env
   ```
   Then edit the four paths (each on its own line; no comments after a value):
   ```bash
   # folder that CONTAINS teledyne_flir_adas_v2/ (read-only)
   IRWM_DATA_ROOT=/path/to/datasets
   # caches (can be anywhere)
   IRWM_CACHE_ROOT=/path/to/irwm/cache
   # model weights: encoder weights are downloaded here, your trained checkpoints go here
   IRWM_CKPT_ROOT=/path/to/irwm/checkpoints
   # logs, metrics and inference results
   IRWM_OUTPUT_ROOT=/path/to/irwm/outputs
   ```
   On a cluster, put these on scratch/work storage, not in your home directory.

3. **Get the dataset (training and `--split` only).** Download FLIR ADAS v2 (Teledyne FLIR,
   free registration), unzip it, and name the folder `teledyne_flir_adas_v2`, so that this path
   exists: `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/FLIR_ADAS_v2/images_thermal_train/coco.json`.

4. **Install the trained detectors (needed for inference with `--head`).** The weights are not
   in git; you get them as a zip, `irwm_frozen_detectors.zip` (≈ 170 MB), from Leonardo. Inside,
   the files are already in the folder layout the code expects:
   ```
   detection/frozen/det_rgb/head.pt     # RGB detector weights
   detection/frozen/det_rgb/meta.json   # its settings, val metrics and SHA-256 checksum
   detection/frozen/det_ir/head.pt      # thermal detector weights
   detection/frozen/det_ir/meta.json
   ```
   Unzip it **into the folder `IRWM_CKPT_ROOT` points to** (the one you set in `.env`, step 2):
   ```bash
   source slurm/env.sh                      # loads IRWM_CKPT_ROOT from .env into this shell
   mkdir -p "$IRWM_CKPT_ROOT"
   unzip irwm_frozen_detectors.zip -d "$IRWM_CKPT_ROOT"
   ls "$IRWM_CKPT_ROOT"/detection/frozen/*  # should list det_ir/ and det_rgb/, each with head.pt and meta.json
   ```
   Do not rename the files or folders and do not edit `meta.json`. On every load, the code checks
   that `head.pt` matches the checksum in `meta.json` and stops with a `SHA-256` error if it doesn't
   (for example after a broken download; just unzip again). Do not commit them either: `*.pt`
   files are git-ignored on purpose. Training with `train.sh` does not need these files.

5. **Run any of the two scripts.** The first run does the rest by itself:
   - installs `uv` (the Python package manager this repo uses) in `~/.local/bin` if needed,
   - builds the Python 3.11 env in `.venv/` (5–15 min; detectron2 and detrex are compiled here),
   - downloads the frozen encoder (DINOv2-Reg weights + PCA) into `$IRWM_CKPT_ROOT`.

   A quick way to do this and check that everything works:
   ```bash
   bash projects/detector_kit/train.sh --modality thermal --quick --batch-size 2
   ```

---

## 2. Inference: `inference.sh`

```bash
# your own images (a folder, searched recursively), with the thermal detector
bash projects/detector_kit/inference.sh --head det_ir --images /path/to/frames

# RGB images, CPU only
bash projects/detector_kit/inference.sh --head det_rgb --images /path/to/rgb --device cpu

# score a detector on a FLIR split (COCO mAP against the labels)
bash projects/detector_kit/inference.sh --head det_ir --split val

# a detector you trained yourself with train.sh
bash projects/detector_kit/inference.sh --ckpt /path/to/best-e034.ckpt --images /path/to/frames

# cross-modal: the RGB detector on thermal images
bash projects/detector_kit/inference.sh --head det_rgb --images /path/to/thermal --modality thermal
```

| option | meaning | default |
|---|---|---|
| `--head det_rgb\|det_ir` | a trained detector from `$IRWM_CKPT_ROOT/detection/frozen/` | one of `--head`/`--ckpt` is required |
| `--ckpt PATH` | a checkpoint written by `train.sh` | |
| `--images DIR` | every `.jpg .jpeg .png .bmp .tif .tiff` under `DIR` | one of `--images`/`--split` is required |
| `--split val\|test\|train` | a FLIR split; also computes mAP | |
| `--modality rgb\|thermal` | how images are read (thermal → grayscale, copied to 3 channels) | the detector's own modality |
| `--device auto\|cuda\|cpu` | where to run | `auto` (GPU if available) |
| `--batch-size N` | images per forward pass; lower it if you run out of GPU memory | 8 |
| `--workers N` | data-loading processes (`--split` only) | min(8, CPU cores) |
| `--score-thr X` | keep and draw boxes with score ≥ X (images mode) | 0.3 |
| `--max-draw N` / `--no-draw` | how many images to save with boxes drawn | 100 |
| `--msda auto\|cuda\|pytorch` | force the compiled op or the PyTorch fallback | `auto` |
| `--out DIR` | output folder | `$IRWM_OUTPUT_ROOT/detector_kit/<date>/<time>` |

**Outputs** (the last log line prints the folder):
- `detections.json`
  - `--images`: one entry per image, with boxes of score ≥ `--score-thr`:
    ```json
    {"file": "seq1/frame_0001.jpg",
     "detections": [{"bbox": [315.5, 231.0, 38.3, 32.8], "score": 0.967, "label": "car", "category_id": 3}]}
    ```
    `bbox` is `[x, y, width, height]` in pixels of the original image, `(x, y)` = top-left corner.
    `category_id` uses FLIR's ids (person 1, bike 2, car 3, motor 4, light 10, sign 12).
  - `--split`: the standard COCO results list, all 300 boxes per image, no threshold (what mAP needs).
- `metrics.json` (`--split` only): AP, AP50, AP75, AP per size and per class.
- `vis/*.jpg`: the images with boxes drawn.
- `inference.log`: the full log.

How images are prepared: each image is center-cropped to a 5:4 aspect ratio and resized to
448×560, exactly as during training. Boxes are mapped back to your original image. A thermal
detector expects 8-bit thermal images like FLIR's (bright = warm). Images that look very
different from FLIR (other cameras, colour palettes, viewpoints) will give worse results.

---

## 3. Training: `train.sh`

```bash
# check the setup first: 1 short epoch, takes a few minutes
bash projects/detector_kit/train.sh --modality thermal --quick --batch-size 2

# the full recipe (50 epochs) on thermal; give the run a name so you can resume it
bash projects/detector_kit/train.sh --modality thermal --batch-size 8 --name ir_bs8

# the run stopped (crash, time limit)? the same command with the same --name resumes it
bash projects/detector_kit/train.sh --modality thermal --batch-size 8 --name ir_bs8

# shorter experiment, logged to Weights & Biases
bash projects/detector_kit/train.sh --modality rgb --epochs 20 --val-every 2 --wandb
```

| option | meaning | default |
|---|---|---|
| `--modality rgb\|thermal` | which FLIR camera to train on | required |
| `--epochs N` | number of epochs; the learning rate drops ×0.1 at 80 % of them | 50 |
| `--batch-size N` | images per step per GPU (see the memory table below) | 16 |
| `--workers N` | data-loading processes | min(8, CPU cores) |
| `--gpus N` | GPUs on this machine (more than 1 uses DDP) | 1 |
| `--lr X` | learning rate | 1e-4 |
| `--val-every N` | compute mAP on FLIR val every N epochs | 5 |
| `--name NAME` | run name = checkpoint folder; the same name resumes | date and time |
| `--msda auto\|cuda\|pytorch` | force the compiled op or the PyTorch fallback | `auto` |
| `--wandb` | also log to W&B (run `uv run wandb login` once) | off |
| `--quick` | setup check: 1 epoch, 20 training and 10 validation batches | off |
| `--out DIR` | folder for logs and metrics | `$IRWM_OUTPUT_ROOT/detector_kit/<date>/<time>` |

Anything after `--` is passed to the Python script as a Hydra override (keys in
[`conf/train.yaml`](conf/train.yaml)), for example `-- data.flip_p=0.0 optim.weight_decay=5e-4`.

**Outputs:**
- Checkpoints in `$IRWM_CKPT_ROOT/detector_kit/<name>/`: `best-eNNN.ckpt` (the epoch with the best
  val AP) and `last.ckpt` (for resuming). Use `best-eNNN.ckpt` with `inference.sh --ckpt`.
- In the output folder: `val_metrics/epoch_NNN.json` (mAP of every validation),
  `csv/version_0/metrics.csv` (losses and metrics, ready to plot), `train_summary.json`
  (best AP, best checkpoint path), `train.log`.

**What the recipe is.** The same as our `det_rgb`/`det_ir`: detrex's DINO 50-epoch schedule,
AdamW (lr 1e-4, weight decay 1e-4), batch 16, mixed precision (fp16), gradient clipping 0.1,
random horizontal flips, frozen encoder. With a batch size other than 16 the learning rate is not
rescaled, so expect results close to, but not exactly, the table at the top.

### Speed and memory

Measured on an NVIDIA L4 (24 GB). The compiled op is "cuda"; the fallback is "pytorch".

| | compiled op | PyTorch fallback |
|---|---|---|
| training speed (head only, fp16) | ≈ 13 img/s | ≈ 7 img/s (1.7–1.9× slower) |
| inference speed (head only, fp16) | ≈ 50 img/s | ≈ 20 img/s |
| one full 50-epoch training run | ≈ 14 h (≈ 16 min per epoch) | roughly 1.5–2× longer |

GPU memory for training is mostly the head. Approximate allocator peaks (head, fp16):

| batch size | compiled op | PyTorch fallback | fits on |
|---|---|---|---|
| 2 | 2.5 GiB | 4.1 GiB | 6–8 GB GPUs |
| 4 | 4.7 GiB | 7.8 GiB | 10–12 GB GPUs |
| 8 | 9.1 GiB | 15.2 GiB | 16–24 GB GPUs |
| 16 | ≈ 18 GiB | ≈ 30 GiB | 24 GB with the compiled op; 40 GB+ with the fallback |

Add about 1–2 GB for the encoder and CUDA itself. Inference needs much less: batch 1 runs on a
4 GB GPU with the fallback.

### Optional: building the compiled kernels

Only worth it to train faster. You need a CUDA 12.x toolkit (`nvcc` on `PATH`, e.g.
`module load cuda/12.4` on a cluster) and `gcc` ≤ 12. Then rebuild:
```bash
bash slurm/setup_env.sh
```
It detects `nvcc` and compiles the CUDA ops for V100, T4, A100, RTX 30xx/40xx, L4 and newer
(`TORCH_CUDA_ARCH_LIST` in `slurm/setup_env.sh`; add your GPU's compute capability if it is older).
The next run should print `deformable attention: cuda`.

---

## 4. Running on a SLURM cluster

The scripts run on whatever machine they start on, so inside a job they just work:
```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=24:00:00
#SBATCH --output=train-%j.out
cd /path/to/ir-world-model
bash projects/detector_kit/train.sh --modality thermal --batch-size 8 --name ir_bs8
```
Build the env and download the weights once on the login node first (step 5 of the setup), since
compute nodes are often offline. Because the run has a fixed `--name`, submitting the same job
again after a time-out resumes it.

---

## 5. Troubleshooting

| message | what to do |
|---|---|
| `set IRWM_... in .env` | step 2 of the setup |
| `frozen detector not found` | step 4 of the setup (unzip into `$IRWM_CKPT_ROOT`) |
| `SHA-256 ... != meta.json head_sha256` | `head.pt` is damaged or from another version: unzip the zip again |
| `CUDA out of memory` | lower `--batch-size` (see the memory table) |
| `no CUDA GPU visible` (training) | training needs an NVIDIA GPU; check `nvidia-smi` |
| `encoder download failed` | the first run needs internet; run it once on a machine/login node that has it |
| `FileNotFoundError ... coco.json` | `IRWM_DATA_ROOT` must be the folder that contains `teledyne_flir_adas_v2/` |
| the env build fails in detectron2/detrex | you need `gcc`/`g++` (`sudo apt install build-essential`, or `module load gcc` on a cluster) |
| W&B asks for a login | don't pass `--wandb`, or run `uv run wandb login` once |
| anything else | the full log is in the output folder (`train.log` / `inference.log`); add `HYDRA_FULL_ERROR=1` before the command for the full Python traceback |

---

## 6. Where the code is (if you want to change things)

| file | what it does |
|---|---|
| `projects/detector_kit/train.sh`, `inference.sh` | flag parsing; they call the Python scripts below |
| `projects/detector_kit/_common.sh` | first-run setup: `.env`, Python env, encoder download |
| `projects/detector_kit/train.py`, `conf/train.yaml` | training entry point and its settings |
| `projects/detector_kit/inference.py`, `conf/inference.yaml` | inference entry point and its settings |
| `lib/irwm/detection/detector.py` | the detector (`DINODetector`): detrex's DINO with a feature input stem |
| `lib/irwm/detection/lightning.py` | training loop (`DetectorModule`, PyTorch Lightning) |
| `lib/irwm/detection/detrex_adapter.py` | detrex loading and the compiled-op / PyTorch-fallback switch (`select_msda`) |
| `lib/irwm/features/encoder.py`, `preprocess.py` | the frozen DINOv2-Reg + PCA encoder and image preprocessing |
| `lib/irwm/data/detection.py`, `flir.py` | FLIR reading and the training dataset |

To train on a dataset other than FLIR, the place to start is `lib/irwm/data/detection.py`: the
model only needs batches of `images [B, 3, 448, 560]` (ImageNet-normalised) and per-image
`boxes [N, 4]` (x1, y1, x2, y2 in those 448×560 pixels) with `labels [N]` in `0..5`.
