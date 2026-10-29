"""Run a DINO-DETR head on a folder of images, or on a FLIR split with COCO mAP. GPU optional.

The model is a frozen head (``head=det_rgb|det_ir``, ``irwm.detection.frozen``) or a checkpoint
written by ``train.py`` (``ckpt=...``). Images go through the same path as F8 validation and F15:
``Preprocess`` (``native``, 448×560) → frozen DINOv2-Reg + PCA (encoder settings stored with the
model) → head (fp16 autocast on CUDA) → boxes in original-image pixels. Without detrex's compiled
CUDA ops, ``msda=auto`` uses its PyTorch fallback; on CPU that fallback is always used.
Normally launched through ``inference.sh``::

    bash projects/detector_kit/inference.sh --head det_ir --images /path/to/frames
    bash projects/detector_kit/inference.sh --head det_rgb --split test

Writes to the run dir: ``detections.json`` (images: per-image boxes ≥ ``score_thr``; split: the
COCO results list, all ≤ 300 boxes per image), ``metrics.json`` (split only), ``vis/*.jpg``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

from irwm.utils.paths import load_env
from irwm.utils.tracking import git_info

# Must run before @hydra.main resolves hydra.run.dir, which reads ${oc.env:IRWM_OUTPUT_ROOT}.
load_env()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLORS = ("#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4")  # one per class label


def load_model(cfg: DictConfig, device: torch.device):
    """``(detector, encoder settings, image_size (H, W), trained modality, description)``."""
    from irwm.detection.detector import DINODetector
    from irwm.detection.frozen import frozen_meta, load_frozen_detector
    from irwm.features.preprocess import POLICY_SIZE

    if cfg.head:
        meta = frozen_meta(cfg.head)
        det = load_frozen_detector(cfg.head, device=device)
        return det, meta["encoder"], tuple(meta["image_size"]), meta["modality"], f"frozen head {cfg.head}"
    ckpt_path = Path(cfg.ckpt).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hyper_parameters"]
    state = {k.removeprefix("detector."): v for k, v in ckpt["state_dict"].items() if k.startswith("detector.")}
    if not state:
        sys.exit(f"[inference] {ckpt_path} has no 'detector.*' weights; is it a checkpoint from train.py?")
    det = DINODetector(**hp["detector"])
    det.load_state_dict(state, strict=True)
    det.eval().requires_grad_(False)
    encoder = dict(hp["encoder"])
    return (det.to(device), encoder, POLICY_SIZE[encoder["policy"]], hp.get("modality"),
            f"checkpoint {ckpt_path} (epoch {ckpt.get('epoch')})")


def read_image(path: Path, modality: str) -> np.ndarray:
    """``uint8 [H, W]`` (thermal, any input converted to grayscale) or ``[H, W, 3]`` (rgb)."""
    with Image.open(path) as im:
        return np.array(im.convert("L" if modality == "thermal" else "RGB"), dtype=np.uint8)


def draw(src: Path, dets: list[dict], out: Path, score_thr: float) -> None:
    """Copy of ``src`` with every detection ≥ ``score_thr`` drawn (``bbox`` xywh in image px)."""
    from irwm.data.classes import CATEGORY_ID_TO_LABEL, CLASS_NAMES

    with Image.open(src) as im:
        canvas = im.convert("RGB")
    pen = ImageDraw.Draw(canvas)
    for d in dets:
        if d["score"] < score_thr:
            continue
        label = CATEGORY_ID_TO_LABEL[d["category_id"]]
        x, y, w, h = d["bbox"]
        color = COLORS[label % len(COLORS)]
        pen.rectangle([x, y, x + w, y + h], outline=color, width=2)
        pen.text((x + 2, y + 1), f"{CLASS_NAMES[label]} {d['score']:.2f}", fill=color)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=90)


def predict_folder(det, encoder, files: list[Path], modality: str, policy: str,
                   image_size: tuple[int, int], batch_size: int, device: torch.device) -> list[list[dict]]:
    """Per file, the COCO-style detections (``bbox`` xywh in original pixels, all ≤ 300 boxes)."""
    from irwm.detection.coco_val import to_flir_detections
    from irwm.features.preprocess import Preprocess

    pre = Preprocess(policy, modality)
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda")
    out: list[list[dict]] = []
    with torch.no_grad():
        for start in range(0, len(files), batch_size):
            chunk = files[start:start + batch_size]
            images, geoms = zip(*(pre(read_image(f, modality)) for f in chunk))
            feats = encoder.encode_pca(torch.stack(images).to(device)).permute(0, 3, 1, 2).contiguous()
            with autocast:
                preds = det(feats, image_size=image_size)
            dets = to_flir_detections(preds, list(range(len(chunk))), list(geoms))
            out.extend([d for d in dets if d["image_id"] == i] for i in range(len(chunk)))
            print(f"[inference] {min(start + batch_size, len(files))}/{len(files)} images", flush=True)
    return out


@hydra.main(version_base="1.3", config_path="conf", config_name="inference")
def main(cfg: DictConfig) -> None:
    if bool(cfg.head) == bool(cfg.ckpt):
        sys.exit("[inference] give exactly one model: head=det_rgb|det_ir or ckpt=<path to a .ckpt>")
    if bool(cfg.images) == bool(cfg.split):
        sys.exit("[inference] give exactly one input: images=<folder> or split=val|test|train")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    # Heavy imports after the checks, so a misconfigured call fails in seconds.
    from irwm.data.classes import CATEGORY_ID_TO_LABEL, CLASS_NAMES
    from irwm.detection.detrex_adapter import select_msda
    from irwm.features.encoder import DinoV2RegEncoder, set_attention

    msda = select_msda(cfg.msda) if device.type == "cuda" else "pytorch"  # CPU always runs the PyTorch path
    det, enc, image_size, trained_on, described = load_model(cfg, device)
    modality = cfg.modality or trained_on
    if modality not in ("rgb", "thermal"):
        sys.exit("[inference] set modality=rgb|thermal (the checkpoint does not record it)")
    if trained_on and modality != trained_on:
        print(f"[inference] note: model trained on {trained_on}, running on {modality} (cross-modal)", flush=True)
    encoder = DinoV2RegEncoder.from_assets(precision=enc["precision"], hub_version=enc["hub_version"], device=device)
    set_attention(encoder.backbone, enc.get("attention", "sdpa"))
    print(f"[inference] {described} | {modality} | device {device} | deformable attention {msda}", flush=True)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    vis_dir = run_dir / "vis"
    n_draw = None if cfg.max_draw is None else int(cfg.max_draw)
    t0 = time.perf_counter()
    result: dict = {"model": described, "modality": modality, "device": str(device), "msda": msda,
                    "encoder": enc, "config": OmegaConf.to_container(cfg, resolve=True), "git": git_info()}

    if cfg.images:
        root = Path(cfg.images).expanduser().resolve()
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            sys.exit(f"[inference] no images ({', '.join(sorted(IMAGE_SUFFIXES))}) under {root}")
        per_file = predict_folder(det, encoder, files, modality, enc["policy"], image_size, cfg.batch_size, device)
        records = []
        for i, (f, dets) in enumerate(zip(files, per_file)):
            if cfg.draw and (n_draw is None or i < n_draw):
                draw(f, dets, vis_dir / f.relative_to(root).with_suffix(".jpg"), cfg.score_thr)
            kept = [{"bbox": [round(v, 2) for v in d["bbox"]], "score": round(d["score"], 4),
                     "label": CLASS_NAMES[CATEGORY_ID_TO_LABEL[d["category_id"]]],
                     "category_id": d["category_id"]} for d in dets if d["score"] >= cfg.score_thr]
            records.append({"file": str(f.relative_to(root)), "detections": kept})
        result.update(images_dir=str(root), n_images=len(files), bbox_format="xywh (x, y = top-left), pixels",
                      score_thr=cfg.score_thr, images=records)
        (run_dir / "detections.json").write_text(json.dumps(result, indent=1))
        n_boxes = sum(len(r["detections"]) for r in records)
        print(f"[inference] {len(files)} images, {n_boxes} boxes with score >= {cfg.score_thr}", flush=True)
    else:
        from irwm.data.flir import FlirSplit
        from irwm.detection.coco_val import coco_map
        from irwm.detection.inference import predict_split, with_person_car

        split = FlirSplit.load(modality, cfg.split)
        pred = predict_split({"model": det}, split, encoder, policy=enc["policy"], image_size=image_size,
                             batch_size=cfg.batch_size, num_workers=cfg.num_workers, device=device,
                             log_every=max(1, 50 // cfg.batch_size))
        dets = pred.detections["model"]
        metrics = with_person_car(coco_map(split.to_coco_gt("normalised"), dets))
        if cfg.draw:
            by_image: dict[int, list[dict]] = {}
            for d in dets:
                by_image.setdefault(d["image_id"], []).append(d)
            for r in split.records[:n_draw]:
                draw(r.path, by_image.get(r.image_id, []), vis_dir / Path(r.file_name).with_suffix(".jpg").name,
                     cfg.score_thr)
        (run_dir / "detections.json").write_text(json.dumps(dets))
        result.update(split=cfg.split, n_images=len(split.records), metrics=metrics)
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        print(f"[inference] FLIR {modality}/{cfg.split}: AP {metrics['AP']:.4f} | AP50 {metrics['AP50']:.4f} | "
              + " ".join(f"{c} {metrics[f'AP_{c}']:.3f}" for c in CLASS_NAMES), flush=True)

    print(f"[inference] done in {time.perf_counter() - t0:.0f} s -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
