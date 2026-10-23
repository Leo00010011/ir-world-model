"""F7 DINO-DETR head cost on the GPU: train-step time and peak memory on synthetic features.

For every (grid, batch size, precision), 3 warm-up + 10 timed train steps of the default head
(forward, ``sum(losses).backward()``, grad clip 0.1, AdamW lr 1e-4 wd 1e-4). Needs CUDA and
detrex's CUDA ops:

    bash slurm/submit.sh ctcomp3 projects/detection/head_bench.py cluster.timeout_min=60

Writes <run_dir>/head_bench.json and logs it to W&B, so F8 can budget its step.
Spec: spec/2026-10-23-dino-detr-head-port (FR10).
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from irwm.detection import DINODetector, cuda_ops_available
from irwm.detection.detector import PATCH
from irwm.utils.paths import load_env
from irwm.utils.tracking import git_info, make_wandb_logger

load_env()

BOXES_PER_IMAGE = 5


def synthetic_targets(b: int, height: int, width: int, gen: torch.Generator) -> list[dict[str, torch.Tensor]]:
    """``BOXES_PER_IMAGE`` random xyxy boxes (32-200 px sides) per image, on CUDA."""
    targets = []
    for _ in range(b):
        wh = torch.empty(BOXES_PER_IMAGE, 2).uniform_(32, 200, generator=gen)
        c = torch.rand(BOXES_PER_IMAGE, 2, generator=gen) * torch.tensor([width, height])
        boxes = torch.cat([c - wh / 2, c + wh / 2], 1).clamp(min=0)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(max=width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(max=height)
        targets.append({"boxes": boxes.cuda(), "labels": torch.randint(0, 6, (BOXES_PER_IMAGE,), generator=gen).cuda()})
    return targets


def bench(model: DINODetector, cfg: DictConfig) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(0)
    rows = []
    for h, w in cfg.grids:
        for b in cfg.batch_sizes:
            for precision in cfg.precisions:
                row: dict[str, Any] = {"grid": [h, w], "batch_size": b, "precision": precision}
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    feats = torch.randn(b, cfg.in_channels, h, w, device="cuda")
                    targets = synthetic_targets(b, PATCH * h, PATCH * w, gen)
                    times = []
                    for step in range(cfg.warmup + cfg.timed):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
                            losses = model(feats, targets)
                        sum(losses.values()).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        if step >= cfg.warmup:
                            times.append(time.perf_counter() - t0)
                    step_s = statistics.median(times)
                    row.update(oom=False, median_step_s=step_s, images_per_s=b / step_s,
                               peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
                    del feats, targets, losses
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    row.update(oom=True)
                print(f"[head_bench] {row}", flush=True)
                rows.append(row)
    return rows


@hydra.main(version_base="1.3", config_path="conf", config_name="head_bench")
def main(cfg: DictConfig) -> None:
    ops = cuda_ops_available()
    if not ops:
        raise SystemExit("[head_bench] needs a CUDA GPU and detrex built with CUDA ops "
                         "(FORCE_CUDA=1, see slurm/setup_env.sh); cuda_ops_available() is False")
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    torch.manual_seed(cfg.seed)
    model = DINODetector(**OmegaConf.to_container(cfg.detector, resolve=True)).cuda().train()
    results: dict[str, Any] = {
        "git": git_info(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_ops_available": ops,
        "torch": torch.__version__,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    out_path = run_dir / "head_bench.json"
    results["runs"] = bench(model, cfg)
    out_path.write_text(json.dumps(results, indent=2))

    try:
        logger = make_wandb_logger(cfg, save_dir=run_dir)
        flat = {}
        for r in results["runs"]:
            key = f"head/{r['grid'][0]}x{r['grid'][1]}/{r['precision']}/b{r['batch_size']}"
            flat.update({f"{key}/{k}": v for k, v in r.items() if isinstance(v, float)})
        logger.experiment.log(flat)
        logger.experiment.save(str(out_path), base_path=str(run_dir))
        logger.experiment.finish()
    except Exception as e:  # W&B problems must not lose the numbers
        print(f"[head_bench] W&B logging failed: {e!r}")
    print(f"[head_bench] results: {out_path}")


if __name__ == "__main__":
    main()
