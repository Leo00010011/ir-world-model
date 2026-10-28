"""Score both frozen heads on both modalities' video test splits: the 2×2 detector × test-modality matrix.

Each test split is encoded once (the encoder in the heads' ``meta.json``: ``native``, SDPA, fp16) and
both heads run on the same features. Needs CUDA and detrex's CUDA ops:

    bash slurm/submit.sh ctcomp3 projects/detection/transfer.py \\
        cluster.timeout_min=60 cluster.cpus_per_task=8 cluster.num_workers=8 cluster.mem_gb=32

Writes the 4 detection files (``irwm.detection.inference.detections_path``, write-once) and
``<run_dir>/transfer.json`` (cells, transfer ratios, diagonal vs F8 val, timing, provenance).
Spec: spec/2026-10-28-cross-modal-detector-transfer (FR6).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from irwm.detection import cuda_ops_available
from irwm.utils.paths import load_env
from irwm.utils.tracking import git_info, make_wandb_logger

# Must run before @hydra.main resolves hydra.run.dir, which reads ${oc.env:IRWM_OUTPUT_ROOT}.
load_env()

MATRIX_KEYS = ("AP", "AP50", "AP_person_car", "AP50_person_car")  # transfer ratios and the W&B table


def _ratio(a: float, b: float) -> float:
    return a / b if b and not math.isnan(b) else math.nan


@hydra.main(version_base="1.3", config_path="conf", config_name="transfer")
def main(cfg: DictConfig) -> None:
    if not cuda_ops_available():
        sys.exit("projects/detection/transfer.py: cuda_ops_available() is False. The frozen heads run with "
                 "detrex's compiled CUDA ops (build them with slurm/setup_env.sh on a node with nvcc).")
    # Heavy imports after the check, so a misconfigured machine fails in seconds.
    import wandb

    from irwm.data.flir import FlirSplit, flir_root
    from irwm.detection.coco_val import coco_map
    from irwm.detection.frozen import FROZEN_MODALITY, frozen_meta, load_frozen_detector
    from irwm.detection.inference import (check_encoder_config, check_outputs_free, detections_path,
                                          predict_split, save_detections, shared_encoder_meta, with_person_car)
    from irwm.features.encoder import DinoV2RegEncoder, set_attention
    from irwm.utils.paths import output_root

    t_start = time.perf_counter()
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    heads, modalities = list(cfg.heads), list(cfg.test_modalities)

    # 1. One encoder setup for all heads, and the config agrees with it.
    shared = shared_encoder_meta(heads)
    check_encoder_config(cfg.encoder, shared)
    enc = shared["encoder"]
    # 2. Write-once: fail before anything is loaded or encoded.
    paths = {h: {m: detections_path(h, m, cfg.split) for m in modalities} for h in heads}
    check_outputs_free([p for per in paths.values() for p in per.values()], overwrite=cfg.overwrite)
    # 3. Frozen heads + the encoder, built once.
    detectors = {h: load_frozen_detector(h, device="cuda") for h in heads}
    encoder = DinoV2RegEncoder.from_assets(precision=enc["precision"], hub_version=enc["hub_version"], device="cuda")
    set_attention(encoder.backbone, enc["attention"])
    logger = make_wandb_logger(cfg, save_dir=run_dir)

    # 4. Per test modality: encode once, run every head, save and score.
    cells: dict[str, dict[str, dict]] = {h: {} for h in heads}
    det_files: dict[str, dict[str, dict]] = {h: {} for h in heads}
    n_images: dict[str, int] = {}
    encode_s: dict[str, float] = {}
    head_s: dict[str, dict[str, float]] = {h: {} for h in heads}
    for m in modalities:
        split = FlirSplit.load(m, cfg.split)
        pred = predict_split(detectors, split, encoder, policy=enc["policy"], image_size=shared["image_size"],
                             batch_size=cfg.batch_size, num_workers=cfg.num_workers, device="cuda")
        n_images[m] = len(pred.image_ids)
        encode_s[m] = pred.encode_s
        gt = split.to_coco_gt("normalised")
        for h in heads:
            dets = pred.detections[h]
            sha = save_detections(paths[h][m], dets, head=h, split=split, overwrite=cfg.overwrite)
            det_files[h][m] = {"path": paths[h][m].relative_to(output_root()).as_posix(), "sha256": sha,
                               "n_images": len(split.records), "n_detections": len(dets)}
            cells[h][m] = with_person_car(coco_map(gt, dets))
            head_s[h][m] = pred.head_s[h]
            print(f"[transfer] {h} on {m}: AP {cells[h][m]['AP']:.4f} AP50 {cells[h][m]['AP50']:.4f}", flush=True)

    # Cross-modal ÷ in-domain on the same test modality (same GT and frames: the ratio isolates the head).
    in_domain = {FROZEN_MODALITY[h]: h for h in heads}
    transfer_ratio: dict[str, dict[str, dict]] = {}
    for h in heads:
        for m in modalities:
            if m == FROZEN_MODALITY[h] or m not in in_domain:
                continue
            ref = cells[in_domain[m]][m]
            transfer_ratio.setdefault(h, {})[m] = {k: _ratio(cells[h][m][k], ref[k]) for k in MATRIX_KEYS}

    metas = {h: frozen_meta(h) for h in heads}
    diag_vs_val: dict[str, dict] = {}
    for h in heads:
        m = FROZEN_MODALITY[h]
        if m not in cells[h]:
            continue
        val = metas[h]["val_metrics"]["metrics"]
        test = cells[h][m]
        diag_vs_val[h] = {"test_AP": test["AP"], "val_AP": val["AP"], "ratio": _ratio(test["AP"], val["AP"]),
                          "test_AP50": test["AP50"], "val_AP50": val["AP50"],
                          "ratio50": _ratio(test["AP50"], val["AP50"])}

    result = {
        "cells": cells,
        "transfer_ratio": transfer_ratio,
        "diag_vs_val": diag_vs_val,
        "heads": {h: {"head_sha256": metas[h]["head_sha256"], "source_ckpt": metas[h]["source_ckpt"],
                      "epoch": metas[h]["epoch"]} for h in heads},
        "encoder": enc,
        "image_size": list(shared["image_size"]),
        "detections": det_files,
        "n_images": n_images,
        "timing": {"encode_s": encode_s, "head_s": head_s, "wall_s": time.perf_counter() - t_start},
        "device": {"name": torch.cuda.get_device_name(), "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_node": os.environ.get("SLURMD_NODENAME")},
        "data_root": str(flir_root()),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git": git_info(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out = run_dir / "transfer.json"
    out.write_text(json.dumps(result, indent=2))

    scalars = {f"test/{h}/{m}/{k}": v for h in heads for m, c in cells[h].items() for k, v in c.items()}
    scalars.update({f"transfer/{h}/{m}/{k}": v for h, per in transfer_ratio.items() for m, r in per.items()
                    for k, v in r.items()})
    scalars.update({f"diag/{h}/{k}": v for h, d in diag_vs_val.items() for k, v in d.items()})
    logger.log_metrics(scalars, step=0)
    rows = [[h, m, *(cells[h][m][k] for k in MATRIX_KEYS)] for h in heads for m in cells[h]]
    logger.experiment.log({"transfer/matrix": wandb.Table(columns=["head", "modality", *MATRIX_KEYS], data=rows)})

    print(f"[transfer] AP / AP50 ({result['device']['name']}, {result['timing']['wall_s']:.0f} s)")
    print(" " * 10 + "".join(f"{m:>16}" for m in modalities))
    for h in heads:
        print(f"{h:<10}" + "".join(f"{cells[h][m]['AP']:>8.4f}{cells[h][m]['AP50']:>8.4f}" for m in modalities))
    print(f"[transfer] -> {out}", flush=True)


if __name__ == "__main__":
    main()
