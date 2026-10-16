"""F4 R5 probe on FLIR: 1-step DINO-Foresight forecast vs copy-last under each resolution policy.

For every clip of the committed clip set (evaluable or not), each modality and each policy:
encode the 4 context frames + the first future frame with our DINOv2-Reg encoder (fp16),
predict the last frame with the reference model (fp16 autocast), and compare prediction and
copy-last against the real last frame in PCA space. Offline (weights from download.py), GPU:

    bash slurm/submit.sh ctcomp3 projects/feature_extraction/r5_probe.py cluster.timeout_min=120

Writes <run_dir>/r5_probe.json and logs the per-(modality, policy) summary to W&B. It makes
no R5 decision. Spec: spec/2026-10-16-dinov2-reg-encoder-wrapper (FR11).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import contextlib  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from itertools import permutations  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from irwm.data.clips import ClipIndex, clip_set_path, file_sha256  # noqa: E402
from irwm.data.flir import FlirSplit  # noqa: E402
from irwm.features.encoder import DinoV2RegEncoder, encoder_tag  # noqa: E402
from irwm.features.preprocess import POLICY_SIZE, Preprocess  # noqa: E402
from irwm.utils.paths import load_env  # noqa: E402
from irwm.utils.tracking import git_info, make_wandb_logger  # noqa: E402
from irwm.worldmodel import assets as A  # noqa: E402
from irwm.worldmodel import reference as R  # noqa: E402

load_env()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_ckpt() -> dict:
    ckpt = torch.load(A.dinof_dir() / A.DINOF_HIGHRES, map_location="cpu", weights_only=False)
    for heavy in ("optimizer_states", "lr_schedulers", "callbacks", "loops"):
        ckpt.pop(heavy, None)  # only state_dict + hyper_parameters are needed
    return ckpt


def _img_size(policy: str) -> tuple[int, int] | None:
    """Reference model grid: the trained 448×896 for stretch/crop, interpolated 448×560 for native."""
    size = POLICY_SIZE[policy]
    return None if size == (448, 896) else size


def _frames(split: FlirSplit, ids: list[int], pre: Preprocess) -> torch.Tensor:
    out = []
    for image_id in ids:
        with Image.open(split.by_image_id(image_id).path) as im:
            out.append(pre(np.array(im, dtype=np.uint8))[0])
    return torch.stack(out)  # [5, 3, H, W]


def _summary(rows: list[dict]) -> dict:
    ratios = [r["ratio"] for r in rows]
    per_seq: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        per_seq[r["rgb_video_id"]].append(r["ratio"])
    return {
        "n_clips": len(rows),
        "grid": rows[0]["grid"] if rows else None,
        "mean_ratio": statistics.fmean(ratios),
        "median_ratio": statistics.median(ratios),
        "summed_ratio": sum(r["model"]["smoothl1"] for r in rows) / sum(r["copy_last"]["smoothl1"] for r in rows),
        "mean_model_cosine": statistics.fmean(r["model"]["cosine"] for r in rows),
        "mean_copy_last_cosine": statistics.fmean(r["copy_last"]["cosine"] for r in rows),
        "per_sequence_mean_ratio": {k: statistics.fmean(v) for k, v in sorted(per_seq.items())},
    }


def _wins(summaries: dict, modality: str, policies: list[str]) -> dict:
    """``"p<q"`` → number of sequences where policy p has a lower mean ratio than q."""
    out = {}
    for p, q in permutations(policies, 2):
        sp = summaries[f"{modality}/{p}"]["per_sequence_mean_ratio"]
        sq = summaries[f"{modality}/{q}"]["per_sequence_mean_ratio"]
        out[f"{p}<{q}"] = sum(sp[k] < sq[k] for k in sp if k in sq)
    out["n_sequences"] = len(summaries[f"{modality}/{policies[0]}"]["per_sequence_mean_ratio"])
    return out


def _finite(rows: list[dict]) -> bool:
    return all(math.isfinite(v) for r in rows for m in ("model", "copy_last") for v in r[m].values())


@hydra.main(version_base="1.3", config_path="conf", config_name="r5_probe")
def main(cfg: DictConfig) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    device = _device(cfg.device)
    clips = ClipIndex.load(cfg.clip_set)
    results: dict[str, Any] = {
        "git": git_info(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "clip_set": cfg.clip_set,
        "clip_set_sha256": file_sha256(clip_set_path(cfg.clip_set)),
        "encoder_tags": {p: encoder_tag(p) for p in cfg.policies},
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "rows": [],
        "summary": {},
        "sequence_wins": {},
    }
    out_path = run_dir / "r5_probe.json"

    encoder = DinoV2RegEncoder.from_assets(precision="fp16", hub_version=cfg.encoder.hub_version, device=device)
    results["pca_sha256"] = encoder.pca.sha256
    splits = {m: FlirSplit.load(m, "test") for m in cfg.modalities}
    ckpt = _load_ckpt()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" else contextlib.nullcontext

    model, model_size = None, "unset"
    # stretch/crop share one reference model; native needs the 448×560 one. Load each once.
    for policy in sorted(cfg.policies, key=lambda p: _img_size(p) is not None):
        if _img_size(policy) != model_size:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model, info = R.load_reference(ckpt, pca_ckpt=A.dinof_dir() / A.DINOF_PCA, img_size=_img_size(policy),
                                           hub_version=cfg.encoder.hub_version, device=device)
            model_size = _img_size(policy)
            results.setdefault("reference_load", {})[str(model_size)] = {"args_diff": info["args_diff"]}
        beta = model.args.beta_smoothl1
        for modality in cfg.modalities:
            pre = Preprocess(policy, modality)
            rows = []
            t0 = time.perf_counter()
            for clip in clips:
                ids = [int(i) for i in clip.context_ids(modality)] + [int(clip.future_ids(modality)[0])]
                z = encoder.encode_pca(_frames(splits[modality], ids, pre).to(device))  # [5, h, w, 1152]
                x = z.unsqueeze(0)  # [1, 5, h, w, 1152]
                with torch.no_grad(), autocast():
                    pred, _ = R.one_step(model, x.clone())
                gt, last = x[:, -1], x[:, -2]
                m = R.pca_metrics(pred, gt, beta)
                c = R.pca_metrics(last, gt, beta)
                rows.append({
                    "clip_id": clip.clip_id, "rgb_video_id": clip.rgb_video_id, "segment_index": clip.segment_index,
                    "modality": modality, "policy": policy, "grid": list(x.shape[2:4]),
                    "model": m, "copy_last": c, "ratio": m["smoothl1"] / c["smoothl1"],
                })
            key = f"{modality}/{policy}"
            results["rows"].extend(rows)
            results["summary"][key] = {**_summary(rows), "all_finite": _finite(rows),
                                       "seconds": round(time.perf_counter() - t0, 1)}
            s = results["summary"][key]
            print(f"[r5_probe] {key}: n={s['n_clips']} grid={s['grid']} mean={s['mean_ratio']:.4f} "
                  f"median={s['median_ratio']:.4f} summed={s['summed_ratio']:.4f} ({s['seconds']} s)", flush=True)
            out_path.write_text(json.dumps(results, indent=2))  # incremental, survives a crash

    for modality in cfg.modalities:
        results["sequence_wins"][modality] = _wins(results["summary"], modality, list(cfg.policies))
    out_path.write_text(json.dumps(results, indent=2))

    try:
        logger = make_wandb_logger(cfg, save_dir=run_dir)
        flat = {f"{k}/{stat}": v for k, s in results["summary"].items()
                for stat, v in s.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        flat.update({f"wins/{m}/{k}": v for m, w in results["sequence_wins"].items() for k, v in w.items()})
        logger.experiment.log(flat)
        logger.experiment.save(str(out_path), base_path=str(run_dir))
        logger.experiment.finish()
    except Exception as e:  # W&B problems must not lose the probe results
        print(f"[r5_probe] W&B logging failed: {e!r}")
    print(f"[r5_probe] results: {out_path}")


if __name__ == "__main__":
    main()
