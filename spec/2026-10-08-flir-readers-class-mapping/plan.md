# F2 · FLIR ADAS v2 readers and class mapping: Plan

## Context and Design Decisions

- **The label space is 6 FLIR classes: person, bike, car, motor, light, sign** (user decision, 2026-10-08). The user chose these after an imbalance study, run during spec writing, of the as-downloaded `coco.json` files. They are the 6 most frequent classes overall. Each has ≥ 1116 train boxes and ≥ 113 test boxes in both modalities.
  - The rejected alternatives: Mission's 3 coarse classes (vehicle, pedestrian, cyclist), all 16 non-empty classes, and 9 "evaluable" classes.
  - Labels `0..5` follow ascending FLIR category id, which gives a deterministic order taken straight from `coco.json`.
  - **Known weakness, kept on purpose:** bike has only 113 thermal / 446 RGB test boxes, so its AP will have wide CIs (R2, F6).
- **There is no coarse mapping** (user decision, 2026-10-08). The primary metric becomes AP/AP50 over the 6 classes. This **deviates from Mission "Evaluation protocol" and TechStack §3.4**, which still describe 3 coarse classes. The constitution must be amended (flagged to the user). Until then, this spec governs.
- **Boxes of every other class are ignore regions** (user decision, 2026-10-08). This covers truck, bus, other vehicle, hydrant, skateboard, stroller, scooter, dog, train and deer.
  - They are neither targets nor background.
  - Dropping them was rejected: the test split has 2634 thermal and 1499 RGB truck boxes (plus other vehicle), so car detections on trucks would become modality-dependent false positives.
  - Merging them into car was rejected because "car" would then stop meaning car.
  - In the COCO export they become `iscrowd: 1`, replicated once per kept category, because `COCOeval` matches crowd GT only within the same `category_id` (FR7).
- **Size buckets use areas normalised to a 640×512 frame** (user decision, 2026-10-08). RGB frames are 1224×1024 to 1800×1600, while thermal is 640×512. In native pixels the same object is 4–8× larger in RGB, so APS/M/L would compare different populations.
  - The normalisation is `w·h·(640·512)/(W·H)`, with the COCO thresholds 32² and 96².
  - Thermal is unchanged by it.
  - Native buckets are still reported in the stats for reference.
  - Bucketing in model-input pixels was rejected, because it depends on R5, which is still open.
- **The data is read from the as-downloaded layout under one configurable root** (user decision, 2026-10-08). `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/` contains `FLIR_ADAS_v2/<split_dir>/`, and the map file sits next to `FLIR_ADAS_v2/`, not inside it. The folder name is the one on disk. *(Amended 2026-10-09: the user renamed the as-downloaded `teledyine_…` folder to the correct spelling `teledyne_flir_adas_v2`; see Amendments.)*
  - This **deviates from TechStack §4.1** (`$IRWM_DATA_ROOT/flir_adas_v2/`) and from the comment in `.env.example` (line 3). Both must be corrected (flagged).
  - The name exists in exactly two places: `FLIR_DIRNAME` and `conf/flir/default.yaml`. A test keeps them equal.
  - **Reminder (TechStack §6):** no absolute paths in code or configs. `/home/leonardo.ulloa/...` goes only in the git-ignored `.env`.
- **The annotation source is `coco.json`, not `index.json`.** The README recommends `coco.json`, which is the filtered COCO version. Facts found in the data (`coco.json` of every split):
  - **80 COCO-style categories are declared**, and 16 have boxes. **The README's id list is wrong for `scooter`:** README says 77, `coco.json` says 75. Ids and names are always read from `coco.json`.
  - **`annotations[].area == bbox w·h` for 100 % of boxes**, so computing `w·h` from the bbox is equivalent.
  - **No `iscrowd: true` and no degenerate boxes exist.** The FR4 error checks are guards, not filters.
  - **Only the test splits have `track_id`.** Image-split annotations get `-1`.
- **All `coco.json` images are kept, including those without annotations.**
  - The counts: 265 / 16 / 0 (RGB train / val / test) and 264 / 16 / 256 (thermal) have no annotation at all.
  - Video continuity (F3) needs every frame, and empty frames are valid DETR training negatives.
  - The image set is defined by `coco.json`. **One orphan JPEG**, `images_rgb_train/data/video-GNhgScttHWdvYkPke-frame-000000-JQkmaAp6qjNJu4w5H.jpg`, has no `coco.json` entry. It is ignored by the readers and listed by the stats.
- **`image_id` is per-split.** Ids are unique and sorted within each `coco.json`, but ranges overlap across splits: RGB train 0–10726, RGB val 2246–5629. This matches the per-split cache files of TechStack §4.2. Lookups always go through a `FlirSplit`.
- **`video_id` and `frame_index` are parsed from `file_name`.**
  - Every file name has the form `data/video-<video_id>-frame-<6 digits>-<frame uid>.jpg`.
  - `extra_info.video_id` agrees with it in every split.
  - `index.json` `frames[].videoMetadata.frameIndex` holds the same index, so `index.json` is not parsed for it.
- **The test split has no per-frame timestamps.** Neither `coco.json` nor `index.json` has any, only `frameIndex`. The only time information is the README's "30 FPS" and a capture-start epoch in `videos[].filename`.
  - So Roadmap F3's "verify from the timestamps that the video really is 30 FPS" cannot be done (flagged).
  - F2 reports the `frame_index` gaps instead. There are 8 sequences per modality. RGB `RMxN6a4CcCeLGu4tA` and thermal `ZAtDSNuZZjkZFvMAo` have gaps of 271 and 178. RGB `msNEBxJE5PPDqenBM` and thermal `SCiKdG3MqZfiE292B` have one gap of 1253. All other steps are 1.
- **Test sequences pair one-to-one across modalities, with equal frame indices.** This was checked during spec writing on all 3749 map entries. F2 only loads the map; the consistency check stays in F3 (Roadmap F3).
- **Pixels are returned at native resolution as `uint8`**: thermal in PIL mode `L` → `[H, W]`, RGB → `[H, W, 3]`.
  - Resizing (R5) and 1→3 replication belong to F4 (Roadmap F4, TechStack §3.1).
  - A sample of 776 JPEGs decoded as `L` (thermal) and `RGB` (RGB test). An unexpected mode raises instead of being converted silently, and the stats check every file's header (FR8.6).
  - RGB test frames have two sizes: 1224×1024 (2151 frames) and 1024×768 (1598). This matters for R5.
- **The API is a split index plus a thin torch Dataset** (user decision, 2026-10-08). `FlirSplit` holds parsed metadata only, so F3 and the stats never decode pixels. `FlirImageDataset` decodes lazily.
  - The rejected alternatives: one Dataset class doing everything, and plain functions over dicts.
  - `__getitem__` returns only `image` and `image_id`. Consumers look the targets up with `by_image_id`, because the default `collate` cannot batch dataclasses or variable-size arrays.
- **The stats entry point is a new project, `projects/data_stats/`** (user decision, 2026-10-08). The computation lives in `lib/irwm/data/stats.py`, so it can be tested on synthetic splits without importing `projects/` (TechStack §2). The project mirrors `projects/smoke/` (`pyproject.toml`, `conf/`, `load_env()` before `@hydra.main`, `projects/smoke/train.py:31`).
- **Stats and data tests run on ctdesk236, where the data is** (user decision, 2026-10-08). They run with no SLURM, since the job is CPU-only. Copying the dataset to ctcomp3 is deferred to before F5.
- **The stats include the imbalance study the user asked for** (user decision, 2026-10-08):
  - per-modality differences per class;
  - under-represented classes;
  - classes absent from val or test;
  - classes absent from COCO-80.

  It is reported for all 16 raw classes, so the 6-class choice stays auditable.
  - **The COCO-80 table (FR9) is hand-written.** FLIR reuses COCO-80 slot numbers but renames some. For example, slot 12 is "stop sign" in COCO and "sign" (any street sign) in FLIR, and slot 18 is "horse" in COCO and "deer" in FLIR. So membership is judged by meaning, not by id.
  - **The "under-represented" rule:** < 1 % of the split's boxes, or < 100 train boxes. It is a reporting flag only.
- **The 16-bit `analyticsData` TIFFs are not read.** TechStack §3.1 fixes the 8-bit JPEG + replication path. R4 may revisit this later.
- **Pillow becomes an explicit dependency of `irwm`.** It is currently only transitive, through torchvision. `uv.lock` already has 12.3.0, so nothing new is downloaded.

## Step 1 — Shared config group and dataset root
- Create `lib/irwm/conf/flir/default.yaml` with one key: `root: ${paths.data_root}/teledyne_flir_adas_v2`, plus a comment pointing to this spec.
- Add `- flir: default` to the defaults of `lib/irwm/conf/irwm_base.yaml`, after `paths`.
- In `lib/irwm/data/flir.py`, define:
  - `FLIR_DIRNAME`;
  - `_SPLIT_DIRS: dict[tuple[str, str], str]`, the FR1 table;
  - `flir_root()` and `split_dir()`, with the FR1 errors.
- Update the comment in `.env.example` line 3 to say `expects teledyne_flir_adas_v2/ inside`.

Covers: FR1.

## Step 2 — Label space and size buckets
Create `lib/irwm/data/classes.py` with the FR2 and FR3 constants and functions. It is pure Python with no numpy, so any module can import it. The module docstring states the decision, the ignore rule and the normalised-area rule, and cites this spec folder.

Covers: FR2, FR3.

## Step 3 — `FlirRecord` and `FlirSplit`
In `lib/irwm/data/flir.py`:
1. Load `coco.json` with `json.load`. Build `cat_ids = {c["id"] for c in categories}`. Group annotations by `image_id` in one pass, keeping file order.
2. For each image, sorted by `id`:
   - apply the FR4 regex and the `extra_info.video_id` check;
   - split its annotations through `label_of` into kept and ignored;
   - build the numpy arrays with explicit dtypes. Use empty `[0, 4]` `float32` arrays when there are none, never shape `[0]`.
3. Make `by_image_id` a dict lookup built once at load. `image_ids()` returns `np.array([...], dtype=np.int64)`.
4. Make `video_tags()` read `<split_dir>/index.json` lazily, only when called, and return `{v["id"]: list(v["tags"]) for v in d["videos"]}`.
5. In `to_coco_gt()`, convert every numpy value with `int()` or `float()`. Ignore-box ids continue from `max(ann_ids over all records) + 1`, incrementing once per copy.

`load_rgb_thermal_map()` is a plain `json.load` of the map file.

Covers: FR4, FR6, FR7.

## Step 4 — `FlirImageDataset`
- Decode with `PIL.Image.open(path)` inside a `with` block. Check `im.mode` and `im.size` before `np.asarray(im)`.
- `np.asarray` of an `L` image is already `[H, W]` `uint8`. Return a contiguous copy, so later in-place transforms cannot touch PIL buffers.
- Apply `transform` last.

Covers: FR5.

## Step 5 — Statistics module
Create `lib/irwm/data/stats.py` with `COCO80_NAME` (FR9), `compute_stats` and `render_markdown` (FR8).
- Per-class counts come from the raw `coco.json` categories, so the 10 ignored classes are counted too. Use `record.ignore_category_ids` plus the kept `labels` mapped back through `LABEL_TO_CATEGORY_ID`. This needs no second JSON parse.
- **File integrity** uses `Image.open(path)` without `.load()`, so only the header is read. It compares `im.size` and `im.mode`.
- **Orphans** are `set(os.listdir(split_dir / "data")) - {Path(r.file_name).name for r in split}`.
- **Gaps** are computed per `video_id` on the sorted `frame_index` values.
- **`render_markdown` output format:**
  - one `##` section per FR8 item;
  - GitHub tables;
  - counts as integers and percentages with 2 decimals;
  - absent classes shown as `0`, never omitted.

Covers: FR8, FR9.

## Step 6 — `projects/data_stats/` entry point
- **`pyproject.toml`:** a copy of `projects/smoke/pyproject.toml` with name `irwm-data-stats`.
- **Workspace registration:** add `irwm-data-stats` to the root `pyproject.toml` `dependencies` and to `[tool.uv.sources]` (`{ workspace = true }`). Then run `uv lock`.
- **`conf/stats.yaml`:** `defaults: [irwm_base, _self_]`, the searchpath `pkg://irwm.conf`, `project: data_stats`, and the FR10 keys.
- **`stats.py`:**
  1. call `load_env()` at import time;
  2. load the splits from `Path(cfg.flir.root)`;
  3. call `compute_stats`;
  4. add `config` and `git`;
  5. write `flir_stats.json` (`indent=2`) and `flir_stats.md` to `HydraConfig.get().runtime.output_dir`;
  6. log two `wandb.Table`s through `make_wandb_logger(cfg, run_dir).experiment`;
  7. print the run dir.
- **Docstring:** the run command.
- Declare `pillow>=10` in `lib/pyproject.toml`.

Covers: FR10.

## Step 7 — Tests
Create `tests/test_flir_data.py`. A fixture `fake_flir(tmp_path)` writes a tiny tree with the as-downloaded layout:
- 6 split dirs, each with a `coco.json`;
- 2–3 images per split, as real tiny JPEGs written with PIL: thermal 64×48 in mode `L`; RGB 80×64 in mode `RGB`, with one RGB test image at 64×48 to cover mixed sizes;
- annotations covering a kept class, an ignored class (truck, id 8) and an image with no annotations;
- `track_id` on the test annotations;
- an `index.json` with `videos[].tags` in the test dirs;
- the map file;
- one orphan JPEG.

The real-data tests use `@pytest.mark.data` and read `IRWM_DATA_ROOT` from the environment, skipping if it is unset. Cover every item in `validation.md` Groups 1–7.

Covers: FR1–FR9.

## Step 8 — Run stats and real-data tests on ctdesk236 (user)
1. In `<repo>/.env`, set `IRWM_DATA_ROOT=/home/leonardo.ulloa/projects/data`. Also set `IRWM_OUTPUT_ROOT` and the other roots if they are missing.
2. `uv sync`
3. `uv run pytest tests/test_flir_data.py -m data -v`
4. `uv run python projects/data_stats/stats.py`

The user returns the pytest summary line and the printed run dir. The run dir holds `flir_stats.json` and `flir_stats.md`.

Covers: FR10, FR11.

## Step 9 — Write up results
Copy `flir_stats.md` to `spec/2026-10-08-flir-readers-class-mapping/stats.md`. Prepend a header with the git commit, run dir and date. Check its numbers against the Data Validity items in `validation.md`. Do not update `Roadmap.md` or `TechStack.md` in this feature; that is the user's call, after the flagged deviations.

Covers: FR11.

## Implementation Order
1. Step 1 — Shared config group and dataset root
2. Step 2 — Label space and size buckets
3. Step 3 — `FlirRecord` and `FlirSplit`
4. Step 4 — `FlirImageDataset`
5. Step 5 — Statistics module
6. Step 6 — `projects/data_stats/` entry point
7. Step 7 — Tests → `uv run pytest tests/test_flir_data.py`, then the full suite `uv run pytest`, which must still pass all 36 F0 tests and the F1 tests
8. Step 8 — Run stats and real-data tests on ctdesk236 (user)
9. Step 9 — Write up results

## Amendments
- **2026-10-09 — dataset folder renamed to `teledyne_flir_adas_v2`** (user decision). The as-downloaded folder was misspelt `teledyine_flir_adas_v2`; the user renamed it on ctdesk236 and changed `FLIR_DIRNAME`, `conf/flir/default.yaml`, `.env.example`, the tests and the spec text to match (commit `a4d65ef`). FR1, Step 1, validation Group 1 and the Read-only data item now use the new name. The "two places only" rule and its test are unchanged. TechStack §4.1 still needs correcting to `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/FLIR_ADAS_v2/<split_dir>/`.
