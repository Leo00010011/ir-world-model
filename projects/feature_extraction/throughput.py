"""F4 encoder throughput on the GPU: synthetic batches, then FLIR train end-to-end.

GPU-only: frames/s and peak memory for every (attention, size, precision, batch size), 5 warm-up +
20 timed batches. Fidelity: max |Δ| and token cosine of ``sdpa`` vs ``plain`` attention.
End-to-end: ``FlirImageDataset`` → ``Preprocess`` in DataLoader workers → ``encode_pca`` (fp16),
10 warm-up + 100 timed batches of 32 per attention, with the share of wall time spent in
transfer + encoding and the projected seconds per epoch. Loader-only: the same DataLoader with no
GPU work, i.e. the read + decode + resize rate of ``$IRWM_DATA_ROOT``. Each end-to-end / loader
run reads its own disjoint slice of the split, so no run is helped by files an earlier one cached.
Needs CUDA:

    bash slurm/submit.sh ctcomp3 projects/feature_extraction/throughput.py cluster.timeout_min=60
    IRWM_STAGE_DATA=1 bash slurm/submit.sh ...   # same, reading FLIR from node-local scratch

Writes <run_dir>/throughput.json and logs it to W&B, so F8 can budget epochs.
Spec: spec/2026-10-16-dinov2-reg-encoder-wrapper (FR12).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from itertools import islice  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from irwm.data.flir import FlirImageDataset, FlirSplit  # noqa: E402
from irwm.features.encoder import DinoV2RegEncoder, encoder_tag, set_attention  # noqa: E402
from irwm.features.preprocess import Preprocess  # noqa: E402
from irwm.utils.paths import load_env  # noqa: E402
from irwm.utils.tracking import git_info, make_wandb_logger  # noqa: E402

load_env()

WARMUP, TIMED = 5, 20  # FR12, GPU-only part


def collate(batch: list[dict]) -> torch.Tensor:
    """Stack the ``Preprocess`` images; geometry and ids are not needed for timing."""
    return torch.stack([item["image"][0] for item in batch])


def gpu_only(encoders: dict[str, DinoV2RegEncoder], cfg: DictConfig) -> list[dict]:
    rows = []
    for attention in cfg.attentions:
        set_attention(encoders["fp16"].backbone, attention)  # the fp32 encoder shares this backbone
        for h, w in cfg.sizes:
            for precision in cfg.precisions:
                for b in cfg.batch_sizes:
                    row: dict[str, Any] = {"attention": attention, "size": [h, w], "precision": precision,
                                           "batch_size": b}
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        x = torch.randn(b, 3, h, w, device="cuda")
                        for _ in range(WARMUP):
                            encoders[precision].encode_pca(x)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(TIMED):
                            encoders[precision].encode_pca(x)
                        torch.cuda.synchronize()
                        dt = time.perf_counter() - t0
                        row.update(frames_per_s=b * TIMED / dt,
                                   peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
                        del x
                    except torch.cuda.OutOfMemoryError:
                        row.update(frames_per_s="oom", peak_mem_gb="oom")
                    print(f"[throughput] {row}", flush=True)
                    rows.append(row)
    set_attention(encoders["fp16"].backbone, "plain")
    return rows


def fidelity(encoders: dict[str, DinoV2RegEncoder], cfg: DictConfig) -> list[dict]:
    """``sdpa`` vs ``plain`` ``encode_pca`` on one random batch per (size, precision)."""
    rows = []
    gen = torch.Generator().manual_seed(0)
    backbone = encoders["fp16"].backbone
    for h, w in cfg.sizes:
        x = torch.randn(4, 3, h, w, generator=gen).cuda()
        for precision in cfg.precisions:
            set_attention(backbone, "plain")
            ref = encoders[precision].encode_pca(x)
            set_attention(backbone, "sdpa")
            out = encoders[precision].encode_pca(x)
            d = (out - ref).abs()
            cos = torch.nn.functional.cosine_similarity(out, ref, dim=-1)
            row = {"size": [h, w], "precision": precision, "max_abs_diff": d.max().item(),
                   "mean_abs_diff": d.mean().item(), "mean_token_cosine": cos.mean().item(),
                   "min_token_cosine": cos.min().item()}
            print(f"[throughput] fidelity {row}", flush=True)
            rows.append(row)
    set_attention(backbone, "plain")
    return rows


def _loader(split: FlirSplit, modality: str, start: int, n_batches: int, cfg: DictConfig) -> DataLoader:
    ds = FlirImageDataset(split, transform=Preprocess(cfg.policy, modality))
    stop = start + n_batches * cfg.e2e_batch_size
    if stop > len(ds):
        raise ValueError(f"{modality}: slice [{start}, {stop}) exceeds the {len(ds)}-frame split; "
                         "lower e2e_batches / loader_batches or the number of attentions")
    return DataLoader(Subset(ds, range(start, stop)), batch_size=cfg.e2e_batch_size,
                      num_workers=int(cfg.num_workers), pin_memory=True, collate_fn=collate)


def end_to_end(encoder: DinoV2RegEncoder, split: FlirSplit, modality: str, start: int, cfg: DictConfig) -> dict:
    it = iter(_loader(split, modality, start, cfg.e2e_warmup + cfg.e2e_batches, cfg))
    for x in islice(it, cfg.e2e_warmup):
        encoder.encode_pca(x.cuda(non_blocking=True))
    torch.cuda.synchronize()
    n_frames, encode_s = 0, 0.0
    t_start = time.perf_counter()
    for x in it:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoder.encode_pca(x.cuda(non_blocking=True))
        torch.cuda.synchronize()
        encode_s += time.perf_counter() - t0
        n_frames += x.shape[0]
    wall_s = time.perf_counter() - t_start
    fps = n_frames / wall_s
    row = {"modality": modality, "policy": cfg.policy, "split_frames": len(split), "first_frame": start,
           "timed_frames": n_frames, "wall_s": wall_s, "encode_s": encode_s, "frames_per_s": fps,
           "encode_share": encode_s / wall_s, "projected_epoch_s": len(split) / fps}
    print(f"[throughput] e2e {row}", flush=True)
    return row


def loader_only(split: FlirSplit, modality: str, start: int, cfg: DictConfig) -> dict:
    """Read + decode + resize rate of the DataLoader alone (no GPU): the data path's ceiling."""
    it = iter(_loader(split, modality, start, cfg.e2e_warmup + cfg.loader_batches, cfg))
    for _ in islice(it, cfg.e2e_warmup):
        pass
    n_frames = 0
    t_start = time.perf_counter()
    for x in it:
        n_frames += x.shape[0]
    wall_s = time.perf_counter() - t_start
    row = {"modality": modality, "policy": cfg.policy, "first_frame": start, "timed_frames": n_frames,
           "wall_s": wall_s, "frames_per_s": n_frames / wall_s}
    print(f"[throughput] loader {row}", flush=True)
    return row


@hydra.main(version_base="1.3", config_path="conf", config_name="throughput")
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("[throughput] needs a CUDA GPU (FR12 measures GPU throughput)")
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    enc16 = DinoV2RegEncoder.from_assets(precision="fp16", hub_version=cfg.encoder.hub_version, device="cuda")
    encoders = {"fp16": enc16, "fp32": DinoV2RegEncoder(enc16.backbone, enc16.pca, precision="fp32")}
    results: dict[str, Any] = {
        "git": git_info(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "encoder_tag": encoder_tag(cfg.policy),
        "gpu": torch.cuda.get_device_name(0),
        "host": socket.gethostname(),
        "data_root": os.environ.get("IRWM_DATA_ROOT"),
        "staged": os.environ.get("IRWM_STAGE_DATA") == "1",
        "cpu_count": os.cpu_count(),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "torch": torch.__version__,
    }
    out_path = run_dir / "throughput.json"
    if cfg.run_gpu_only:
        results["gpu_only"] = gpu_only(encoders, cfg)
        out_path.write_text(json.dumps(results, indent=2))
    if "sdpa" in cfg.attentions:
        results["fidelity"] = fidelity(encoders, cfg)
        out_path.write_text(json.dumps(results, indent=2))
    results["end_to_end"], results["loader_only"] = [], []
    window = (cfg.e2e_warmup + cfg.e2e_batches) * cfg.e2e_batch_size
    for modality in ("rgb", "thermal"):
        split = FlirSplit.load(modality, "train")
        for k, attention in enumerate(cfg.attentions):
            set_attention(enc16.backbone, attention)
            row = end_to_end(enc16, split, modality, k * window, cfg)
            results["end_to_end"].append({"attention": attention, **row})
            out_path.write_text(json.dumps(results, indent=2))
        results["loader_only"].append(loader_only(split, modality, len(cfg.attentions) * window, cfg))
        out_path.write_text(json.dumps(results, indent=2))

    try:
        logger = make_wandb_logger(cfg, save_dir=run_dir)
        flat = {}
        for r in results.get("gpu_only", []):
            key = f"gpu_only/{r['attention']}/{r['size'][0]}x{r['size'][1]}/{r['precision']}/b{r['batch_size']}"
            flat.update({f"{key}/{k}": v for k, v in r.items() if isinstance(v, float)})
        for r in results["end_to_end"]:
            flat.update({f"e2e/{r['attention']}/{r['modality']}/{k}": v for k, v in r.items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)})
        for r in results["loader_only"]:
            flat.update({f"loader/{r['modality']}/{k}": v for k, v in r.items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)})
        logger.experiment.log(flat)
        logger.experiment.save(str(out_path), base_path=str(run_dir))
        logger.experiment.finish()
    except Exception as e:  # W&B problems must not lose the numbers
        print(f"[throughput] W&B logging failed: {e!r}")
    print(f"[throughput] results: {out_path}")


if __name__ == "__main__":
    main()
