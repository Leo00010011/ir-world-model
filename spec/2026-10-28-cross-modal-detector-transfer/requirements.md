# F15 · Cross-modal detector transfer: Requirements

## Goal

Score the two frozen F8 heads, `det_rgb` and `det_ir`, on both modalities' `video_*_test` splits. The result is a 2×2 detector × test-modality matrix that measures domain shift at the detector level: how much of what a head learned on one modality's DINOv2-Reg features carries over to the other's (Roadmap F15). F15 delivers three things. The first is an importable inference path (`irwm.detection.inference`), which encodes a split once and runs several frozen heads on the same features. F9 and F12 reuse it rather than writing a second one (Roadmap F9). The second is the per-image detections of all four cells, saved write-once at a stable path with enough metadata to recover each frame's video and tags, so F6/F9 can re-score them and later analyses can slice them without re-running inference. The third is a results note with the 2×2 tables. The diagonal cells are preliminary numbers for F9's Result Table 1.

## Scope

In scope:
- `lib/irwm/detection/inference.py`: the shared-encoder check, batched multi-head inference on a `FlirSplit`, the detections file format (save/load/path), and the person + car metric.
- `projects/detection/transfer.py` and `projects/detection/conf/transfer.yaml`: one job that produces the four cells, their metrics JSON, a W&B run and the four detection files.
- `tests/test_detector_transfer.py`: CPU tests, one `gpu` test, one `data` + `gpu` test.
- One run on ctcomp3 (user step) and `spec/2026-10-28-cross-modal-detector-transfer/results.md`.

Out of scope:
- Bootstrap CIs, per-horizon metrics, and the rigorous evaluator (F6). Cell differences are reported as preliminary.
- Any breakdown by video, by day/night or by other tags (user decision, 2026-10-28: "out of scope; the output must carry enough metadata to recover the tags per prediction; the analysis is decided once the outputs exist"). FR4 stores that metadata.
- The `val` split, in any cell (user decision: test 2×2 only).
- A pass/fail gate on the diagonal cells (user decision: report only, see FR7).
- Re-scoring from saved files inside `transfer.py`. A rerun with existing files fails unless `overwrite=true` (user decision). Re-scoring is done with `load_detections` + `coco_map` (FR4, F8 FR4), not by the entry point.
- Full 3072-d features, the PCA-loss check (F9, R7), and any world-model features (F12).
- Any change to the frozen heads, to `irwm.detection.coco_val`, `frozen`, `lightning`, or to `irwm.data.detection`.

## Functional Requirements

**FR1 — Shared encoder settings (`lib/irwm/detection/inference.py`).**
- `shared_encoder_meta(names: Sequence[str]) -> dict` reads `frozen_meta(name)` for each name and returns `{"encoder": <meta["encoder"]>, "image_size": tuple(meta["image_size"]), "precision": meta["precision"]}`.
- The `encoder` dicts (all keys: `policy`, `precision`, `hub_version`, `attention`, `encoder_tag`), `image_size` and `precision` must be equal across all names. Otherwise it raises `ValueError` and names the first differing key and the two values.
- An empty `names` → `ValueError`.
- `check_encoder_config(cfg_encoder: Mapping[str, Any], shared: Mapping[str, Any]) -> None` compares the keys `policy`, `precision`, `hub_version`, `attention` of a Hydra `encoder` node with `shared["encoder"]`. A mismatch raises `ValueError` naming the key and both values. FR5/FR6 use it.

**FR2 — Multi-head split inference (`lib/irwm/detection/inference.py`).**
- `predict_split(detectors: Mapping[str, DINODetector], split: FlirSplit, encoder: nn.Module, *, policy: Policy, image_size: tuple[int, int], batch_size: int = 16, num_workers: int = 0, device: str | torch.device = "cpu") -> SplitPrediction`.
- Frames: `FlirDetectionDataset(split, policy=policy, flip_p=0.0, filter_empty=False)` (F8 FR2), so every frame of the split is used in ascending `image_id`, preprocessed with `Preprocess(policy, split.modality)`. Loader: `shuffle=False`, `drop_last=False`, `collate_fn=collate_detection`, `pin_memory = device is CUDA`.
- Per batch: `feats = encoder.encode_pca(images).permute(0, 3, 1, 2).contiguous()` is computed **once**, then each detector runs on the same `feats` with `image_size=image_size`, and its output goes through `to_flir_detections` (F8 FR4). On CUDA the detector forward runs under `torch.autocast("cuda", dtype=torch.float16)` (`meta.json` `precision: 16-mixed`); on CPU there is no autocast (fp32).
- A detector in training mode → `ValueError` before any batch is read. An empty `detectors` → `ValueError`.
- `SplitPrediction` is a frozen dataclass with `detections: dict[str, list[dict]]` (COCO result dicts in FLIR px, image order ascending, all ≤ 300 boxes per image, no score threshold), `image_ids: list[int]`, `encode_s: float`, `head_s: dict[str, float]` (wall seconds with `torch.cuda.synchronize()` around each timed section on CUDA).

**FR3 — Person + car metric (`lib/irwm/detection/inference.py`).**
- `with_person_car(metrics: Mapping[str, float]) -> dict[str, float]` returns a copy of `coco_map`'s output plus `AP_person_car = mean(AP_person, AP_car)` and `AP50_person_car = mean(AP50_person, AP50_car)`. It is `nan` if either input is `nan`.

**FR4 — Detections file (`lib/irwm/detection/inference.py`).**
- `detections_path(name: str, modality: Modality, split: Split = "test") -> Path` = `output_root() / "detection" / "detections" / <frozen_meta(name)["encoder"]["encoder_tag"]> / name / f"{modality}_{split}.json.gz"`.
- `save_detections(path: Path, detections: list[dict], *, head: str, split: FlirSplit, overwrite: bool = False) -> str` writes gzip-compressed JSON (`gzip.GzipFile(..., mtime=0)`, `json.dumps(..., sort_keys=True)`, so identical content → identical bytes) to `path.with_name(path.name + ".tmp")`, then renames it to `path`. It returns the SHA-256 of the written file. The object holds:
  - `format: "irwm.detections/1"`, `head`, `head_sha256`, `head_modality`, `encoder` and `image_size` (from `frozen_meta(head)`), `precision: "16-mixed"`;
  - `modality`, `split`, `n_images`, `n_detections`;
  - `images`: one entry per frame of `split`, in ascending `image_id`: `image_id`, `file_name`, `video_id`, `frame_index`, `width`, `height`;
  - `video_tags`: `split.video_tags()` when `split.split == "test"`, else `{}`;
  - `detections`: the COCO result list unchanged;
  - `git` (`git_info()`), `created_at` (ISO 8601 UTC), `device` (`torch.cuda.get_device_name()` or `"cpu"`).
- `path` exists and `overwrite=False` → `FileExistsError` (user decision). The parent directory is created.
- A detection whose `image_id` is not in `split` → `ValueError`.
- `load_detections(path: Path) -> dict` returns the object above.
- `check_outputs_free(paths: Iterable[Path], *, overwrite: bool) -> None` raises `FileExistsError` listing every existing path when `overwrite` is false. The entry point calls it before anything is encoded.

**FR5 — Hydra config (`projects/detection/conf/transfer.yaml`).**
- `defaults: [irwm_base, _self_]`, `hydra.searchpath: ["pkg://irwm.conf"]`, `project: detection`.
- `heads: [det_rgb, det_ir]`, `test_modalities: [rgb, thermal]`, `split: test`, `batch_size: 16`, `num_workers: ${cluster.num_workers}`, `overwrite: false`.
- `encoder: {attention: sdpa}` is merged into the shared `encoder` group. This makes the logged config truthful. The entry point calls `check_encoder_config(cfg.encoder, shared)` (FR1), so a config that disagrees with the frozen heads raises `ValueError`.
- `wandb: {tags: [transfer]}`.

**FR6 — Entry point (`projects/detection/transfer.py`).**
- `@hydra.main(config_name="transfer")`, `load_env()` at import. If `cuda_ops_available()` is false, it exits non-zero with a clear message (same as `train.py`; the heads were validated with the compiled MSDA).
- Order of work:
  1. `shared = shared_encoder_meta(cfg.heads)`, then `check_encoder_config(cfg.encoder, shared)`;
  2. `check_outputs_free` on the 4 `detections_path(head, modality, cfg.split)`;
  3. load each head with `load_frozen_detector(name, device="cuda")`, and build the encoder once: `DinoV2RegEncoder.from_assets(precision=..., hub_version=..., device="cuda")`, then `set_attention(encoder.backbone, attention)`, all values taken from the shared meta;
  4. for each test modality: `FlirSplit.load(modality, cfg.split)`, `predict_split` with all heads, `save_detections` per head, then `with_person_car(coco_map(split.to_coco_gt("normalised"), dets))` per head.
- It writes `<hydra run dir>/transfer.json` (TechStack §7), with these keys:
  - `cells[head][modality]`: the FR3 metrics;
  - `transfer_ratio[head][modality]` for the two off-diagonal cells: `{"AP", "AP50", "AP_person_car", "AP50_person_car"}`, each the cross-modal value divided by the in-domain head's value on the **same** test modality;
  - `diag_vs_val[head]`: `test_AP`, `val_AP`, `ratio`, `test_AP50`, `val_AP50`, `ratio50` (val from `frozen_meta(head)["val_metrics"]["metrics"]`);
  - `heads[head]`: `head_sha256`, `source_ckpt`, `epoch`;
  - `encoder`, `image_size`;
  - `detections[head][modality]`: `path` (relative to `output_root()`), `sha256`, `n_images`, `n_detections`;
  - `n_images[modality]`;
  - `timing`: `encode_s[modality]`, `head_s[head][modality]`, `wall_s`;
  - `device`: GPU name, `torch.cuda.max_memory_allocated()` in GiB, `SLURM_JOB_ID` and `SLURMD_NODENAME` (or `null`);
  - `data_root` (`flir_root()`), `config` (resolved), `git`, `created_at`.
- W&B through `make_wandb_logger` (one run). Every metric of every cell is logged as `test/<head>/<modality>/<metric>`, every ratio as `transfer/<head>/<modality>/<key>`, and `diag_vs_val` as `diag/<head>/<key>`. A `wandb.Table` `transfer/matrix` holds one row per cell: `head, modality, AP, AP50, AP_person_car, AP50_person_car`.
- On stdout it prints a 2×2 AP / AP50 grid and the path of `transfer.json`.

**FR7 — Run and results note (`spec/2026-10-28-cross-modal-detector-transfer/results.md`).**
- One run of `transfer.py` on ctcomp3, on whatever GPU the scheduler assigns (the L4 is a lower bound, not a target). The GPU is recorded.
- The note holds:
  - the 2×2 tables of AP, AP50, AP_person_car, AP50_person_car;
  - per cell: APs/APm/APl and per-class AP / AP50;
  - the transfer ratios;
  - `diag_vs_val` for both heads. This is **reported, not gated** (user decision, 2026-10-28). If a ratio is below 0.5 or above 1.5, the note says so in one sentence, without investigation;
  - cost (encode and head seconds, wall time, GPU, peak memory), SLURM job id, W&B run id, git commit, the 4 detection file paths with SHA-256;
  - caveats: R10 for the cross-modal cells, preliminary status until F6 CIs exist, and the `native` RGB edge crop (F8 plan.md "GT is the full FLIR frame").

**FR8 — Tests (`tests/test_detector_transfer.py`).**
- CPU tests (default suite, no data, no weights, < 60 s) cover FR1–FR6 (validation.md Groups 1–6) on a tiny synthetic `video_*_test` tree, with a stub encoder and F7's reduced detector.
- One `@pytest.mark.gpu` test covers FR2 on CUDA under fp16 autocast (Group 7).
- One `@pytest.mark.data` + `gpu` test runs the real frozen heads and encoder on 4 real thermal test frames (Group 8).

## Public API Summary

```python
# lib/irwm/detection/inference.py   (imported explicitly, not re-exported from irwm.detection)
@dataclass(frozen=True)
class SplitPrediction:
    detections: dict[str, list[dict]]   # head name -> COCO results in FLIR px, image_id ascending
    image_ids: list[int]
    encode_s: float
    head_s: dict[str, float]

def shared_encoder_meta(names: Sequence[str]) -> dict: ...   # {"encoder", "image_size", "precision"}
def check_encoder_config(cfg_encoder: Mapping[str, Any], shared: Mapping[str, Any]) -> None: ...
def predict_split(detectors: Mapping[str, DINODetector], split: FlirSplit, encoder: nn.Module, *,
                  policy: Policy, image_size: tuple[int, int], batch_size: int = 16,
                  num_workers: int = 0, device: str | torch.device = "cpu") -> SplitPrediction: ...
def with_person_car(metrics: Mapping[str, float]) -> dict[str, float]: ...
def detections_path(name: str, modality: Modality, split: Split = "test") -> Path: ...
def save_detections(path: Path, detections: list[dict], *, head: str, split: FlirSplit,
                    overwrite: bool = False) -> str: ...   # returns SHA-256
def load_detections(path: Path) -> dict: ...
def check_outputs_free(paths: Iterable[Path], *, overwrite: bool) -> None: ...
```

## Dependencies

| Item | Direction | Notes |
|---|---|---|
| `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/FLIR_ADAS_v2/video_{rgb,thermal}_test/` | read | `coco.json`, `index.json` (tags), 3,749 JPEGs per modality (F2) |
| `$IRWM_CKPT_ROOT/detection/frozen/{det_rgb,det_ir}/` | read | `head.pt` + `meta.json`, SHA-checked by `load_frozen_detector` (F8); never written |
| `$IRWM_CKPT_ROOT/{dino_foresight,torch_hub}/` | read | DINOv2-Reg weights, pinned hub code, PCA (F1, F4) |
| `irwm.detection.{coco_val,frozen,detector}`, `irwm.data.detection`, `irwm.features.{encoder,preprocess}` | read (import) | F4, F7, F8, unchanged |
| `$IRWM_OUTPUT_ROOT/detection/detections/dinov2_vitb14reg_448x560_l3-6-9-12_pca1152/{det_rgb,det_ir}/{rgb,thermal}_test.json.gz` | write (once) | 4 files, ≤ 1,124,700 detections each, ≈ 30–40 MB gzipped each (estimate) |
| `$IRWM_OUTPUT_ROOT/detection/<date>/<time>/` | write | Hydra run dir: `transfer.json`, W&B files |
| GPU time on ctcomp3 | — | 7,498 frames encoded once + 4 × 3,749 head passes ≈ 10 min GPU, plus a few minutes of startup (R11) |
| New deps | — | none |
