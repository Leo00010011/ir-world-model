# F2 · FLIR ADAS v2 readers and class mapping: Requirements

## Goal

Give every later feature one read-only, tested way to get FLIR ADAS v2 frames and labels. It covers both modalities (`rgb`, `thermal`) and all three splits (`train`, `val`, `test` = the 30 FPS video split). F2 also fixes the project's **single detection label space**: 6 FLIR classes. Every box of any other class becomes an **ignore region**. Finally, F2 writes a dataset-statistics report that records the class-imbalance study behind that choice.
- F3 builds clips from the per-frame `video_id` and `frame_index` exposed here.
- F4 and the R5 probe take raw native-resolution images from here.
- F5 encodes every frame through `FlirImageDataset`.
- F6 scores against the COCO GT dict that `FlirSplit.to_coco_gt` exports.
- F8 trains on the 6-class targets.

## Scope

In scope:
- `lib/irwm/data/classes.py`: the 6-class label space, its mappings to FLIR category ids, the ignore rule, and the 640×512-normalised size buckets.
- `lib/irwm/data/flir.py`:
  - the dataset-root resolution;
  - `FlirRecord`, `FlirSplit`, `FlirImageDataset`;
  - the RGB↔thermal video-map loader.
- `lib/irwm/data/stats.py`: pure functions that compute the statistics report from loaded splits.
- `lib/irwm/conf/flir/default.yaml`: a shared Hydra group, added to `irwm_base`.
- `projects/data_stats/`: a Hydra entry point that writes the report (JSON + Markdown), plus a W&B run.
- `tests/test_flir_data.py`: synthetic unit tests, plus `@pytest.mark.data` tests on the real dataset.
- `spec/2026-10-08-flir-readers-class-mapping/stats.md`: the report produced on ctdesk236, committed as the results note.

Out of scope:
- **Clip building, stride, FPS checks and the RGB↔thermal map consistency check** (F3). F2 only exposes the per-frame metadata and loads the map.
- **Resizing, the R5 resolution policy, 1→3 channel replication and normalisation** (F4). F2 returns native-resolution `uint8` pixels.
- **The 16-bit raw thermal TIFFs** (`analyticsData/`). The pipeline uses the 8-bit JPEGs (TechStack §3.1).
- **The Conservator `index.json` annotations.** Only the `videos[].tags` of the test split are read (FR6).
- **COCO evaluation, normalising detection areas and bootstrap CIs** (F6). F2 only exports the GT.
- **Copying the dataset to ctcomp3** (before F5).

## Functional Requirements

**FR1 — Dataset root (`irwm/data/flir.py`, `irwm/conf/flir/default.yaml`).**
- `FLIR_DIRNAME = "teledyne_flir_adas_v2"`: the folder name exactly as it is on disk, misspelling included.
- `flir_root() -> Path` returns `irwm.utils.paths.data_root() / FLIR_DIRNAME`.
- The layout is the zip's own:
  - split `<split_dir>` lives at `<root>/FLIR_ADAS_v2/<split_dir>/`;
  - the RGB↔thermal map is `<root>/rgb_to_thermal_vid_map.json`.
- Split directories:

  | modality | `train` | `val` | `test` |
  |---|---|---|---|
  | `rgb` | `images_rgb_train` | `images_rgb_val` | `video_rgb_test` |
  | `thermal` | `images_thermal_train` | `images_thermal_val` | `video_thermal_test` |

- The Hydra group `flir/default.yaml` contains `root: ${paths.data_root}/teledyne_flir_adas_v2`. `irwm_base.yaml` lists `flir: default` in its defaults.
- Every reader function takes `root: Path | None = None`. `None` means `flir_root()`.
- Errors:
  - unknown `modality` or `split` → `ValueError` listing the valid values;
  - missing `coco.json` → `FileNotFoundError` naming the full path.

**FR2 — Label space (`irwm/data/classes.py`).**
- `CLASS_NAMES: tuple[str, ...] = ("person", "bike", "car", "motor", "light", "sign")`. Contiguous labels are `0..5` in that order, which is ascending FLIR category id.
- `CATEGORY_ID_TO_LABEL: dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 10: 4, 12: 5}`, and its inverse `LABEL_TO_CATEGORY_ID`.
- `NUM_CLASSES = 6`.
- `label_of(category_id: int) -> int | None` returns the label, or `None` for any other FLIR category. A box whose label is `None` is an **ignore region**.
- No coarse (vehicle/pedestrian/cyclist) mapping exists anywhere in the code.

**FR3 — Size buckets (`irwm/data/classes.py`).**
- `REF_WIDTH, REF_HEIGHT = 640, 512`.
- `normalised_area(w: float, h: float, img_w: int, img_h: int) -> float` returns `w * h * (640 * 512) / (img_w * img_h)`.
- `size_bucket(area: float) -> Literal["S", "M", "L"]` returns `"S"` if `area < 32**2`, `"M"` if `area < 96**2`, and `"L"` otherwise. This uses the COCO thresholds, applied to the normalised area.

**FR4 — Records and split index (`irwm/data/flir.py`).**
- A frozen `@dataclass FlirRecord`:
  - `image_id: int`
  - `file_name: str`: as in `coco.json`, for example `"data/video-…jpg"`
  - `path: Path`: absolute
  - `width: int`, `height: int`
  - `video_id: str`, `frame_index: int`
  - `boxes: np.ndarray`: `float32 [N, 4]`, COCO `xywh`, native pixels
  - `labels: np.ndarray`: `int64 [N]` in `0..5`
  - `ann_ids: np.ndarray`: `int64 [N]`
  - `track_ids: np.ndarray`: `int64 [N]`, `-1` when the annotation has no `track_id`
  - `ignore_boxes: np.ndarray`: `float32 [M, 4]`
  - `ignore_category_ids: np.ndarray`: `int64 [M]`, the raw FLIR ids
- `FlirSplit.load(modality: Modality, split: Split, root: Path | None = None) -> FlirSplit` parses `coco.json` once and does no image I/O. Here `Modality = Literal["rgb", "thermal"]` and `Split = Literal["train", "val", "test"]`.
- Attributes: `modality`, `split`, `root`, `split_dir: Path`, `records: list[FlirRecord]`. The records are sorted by `image_id`, and there is one per `coco.json` image, including images without annotations.
- Methods:
  - `__len__`, `__iter__`
  - `by_image_id(image_id: int) -> FlirRecord`. Raises `KeyError` if the id is not in this split.
  - `image_ids() -> np.ndarray`: `int64 [N]`, ascending
- Within each record, annotations keep their `coco.json` order. `boxes` and `labels` hold only the 6 classes. Every other category goes to `ignore_boxes`.
- **Parsing rules (all raise `ValueError` naming the offending image or annotation):**
  - `video_id` and `frame_index` come from `file_name`, which must match `^data/video-(?P<video_id>[A-Za-z0-9]+)-frame-(?P<frame>\d{6})-[A-Za-z0-9]+\.jpg$`;
  - the parsed `video_id` must equal `extra_info["video_id"]`;
  - duplicate `image_id` → error;
  - an annotation `category_id` absent from the file's `categories` → error;
  - a box with `w <= 0` or `h <= 0` → error.
- **`image_id` is unique only within one (modality, split).** For example, RGB `train` ids span 0–10726 and RGB `val` ids 2246–5629. No API looks up an id across splits.

**FR5 — Image dataset (`irwm/data/flir.py`).**
- `FlirImageDataset(split: FlirSplit, transform: Callable[[np.ndarray], Any] | None = None)` is a `torch.utils.data.Dataset`.
- `__len__` equals `len(split)`. `__getitem__(i)` returns `{"image": transform(img) if transform else img, "image_id": int}`, where `img` is the decoded pixels of `split.records[i]`:
  - `rgb` → `uint8 [H, W, 3]`, PIL mode `"RGB"`;
  - `thermal` → `uint8 [H, W]`, PIL mode `"L"`.
- The decoded mode must match the expected one. Otherwise → `ValueError`; there is no silent `convert()`. The decoded `(W, H)` must equal the record's `(width, height)`. Otherwise → `ValueError`.
- No resize and no channel replication. It holds no open file handles between items, so it can be pickled for `DataLoader` workers.

**FR6 — Video metadata (`irwm/data/flir.py`).**
- `load_rgb_thermal_map(root: Path | None = None) -> dict[str, str]` maps an RGB test basename to a thermal test basename, read from `rgb_to_thermal_vid_map.json`. It does no validation; that is F3's job.
- `FlirSplit.video_tags() -> dict[str, list[str]]` maps each `video_id` to `tags`, read from `<split_dir>/index.json` → `videos[]`. It works for `split == "test"` only; other splits → `ValueError`.

**FR7 — COCO GT export (`FlirSplit.to_coco_gt`).**
- `to_coco_gt(area: Literal["normalised", "native"] = "normalised") -> dict` returns a COCO dict with:
  - `images`: `id`, `file_name`, `width`, `height`, in record order;
  - `categories`: the 6 classes with their **FLIR** ids and names;
  - `annotations`: every kept box with `id` = the original annotation id, `category_id` = the FLIR id, `bbox`, `iscrowd: 0`, and `area` per the `area` argument (FR3 for `"normalised"`, `w * h` for `"native"`).
- **Every ignore box is emitted once per kept category (6 copies)**, with `iscrowd: 1` and fresh ids starting at `max(original ann id) + 1`. `COCOeval` matches crowd regions only within one category, so 6 copies make detections of any class on an ignored truck neither TP nor FP.
- The output is JSON-serialisable: plain `int`, `float` and `str` values, no numpy scalars.

**FR8 — Statistics computation (`irwm/data/stats.py`).**
- `compute_stats(splits: Sequence[FlirSplit], *, check_files: bool = True) -> dict` returns a JSON-serialisable dict that covers all six (modality, split) pairs passed in. It contains:
  1. **Images:**
     - the count per split;
     - the `(width, height)` histogram;
     - the number of images without any kept box;
     - the number of images without any annotation.
  2. **Per raw FLIR class** (every category with ≥1 box in any passed split), per modality and split:
     - box count and image count;
     - share of the split's boxes, in %;
     - S/M/L counts, both native (`w*h`) and normalised (FR3);
     - flags: `selected` (in the 6), `coco80` (FR9 table), `underrepresented` (< 1 % of the split's boxes, or < 100 boxes in `train`), `absent_val`, `absent_test`.
  3. **Modality comparison:** `thermal/rgb` box-count ratio per class and split. For `test` the frames are time-synced pairs, so the ratio measures annotation asymmetry.
  4. **Selected 6 + ignore:** kept boxes per class, ignore boxes per raw class, totals.
  5. **Video sequences** (from `test` records): per `video_id`:
     - `n_frames`, `frame_index` min and max;
     - the number of gaps (consecutive sorted indices differing by > 1), each gap's size, and the lengths of the contiguous segments;
     - `tags` (FR6).
  6. **File integrity**, when `check_files=True`:
     - for every record, the JPEG header's `(W, H)` and mode, compared with the record and with FR5's expected mode;
     - files in `data/` that are not in `coco.json` ("orphans"), listed by name.

     With `check_files=False`, the `file_integrity` key is omitted and no image file is opened.
- `render_markdown(stats: dict) -> str` turns the dict into tables: sections 1–6, plus a 6-class summary table with the same columns as the class table in FR8.2.

**FR9 — COCO-80 membership (`irwm/data/stats.py`).**
- A fixed table `COCO80_NAME: dict[str, str | None]` maps the 16 FLIR names to COCO-80 names:
  - person → person, bike → bicycle, car → car, motor → motorcycle, bus → bus, train → train, truck → truck;
  - light → traffic light, hydrant → fire hydrant, dog → dog, skateboard → skateboard;
  - sign → `None` (COCO has only "stop sign"), deer → `None`, stroller → `None`, scooter → `None`, other vehicle → `None`.
- A FLIR name that is not in the table → `KeyError`. The report must not silently mark a class "not in COCO".

**FR10 — Stats entry point (`projects/data_stats/stats.py`, `projects/data_stats/conf/stats.yaml`).**
- A `@hydra.main` script. `project: data_stats`. It follows the Hydra wiring rules of TechStack §2.
- Config keys:
  - `modalities: [rgb, thermal]`
  - `splits: [train, val, test]`
  - `check_files: true`
- It loads the splits from `cfg.flir.root`, then writes `flir_stats.json` and `flir_stats.md` into the Hydra run dir. The JSON includes `config` (resolved) and `git` (`git_info()`).
- W&B: it creates the logger with `make_wandb_logger(cfg, run_dir)` and logs the selected-6 table and the per-class table as `wandb.Table`s.
- It is a uv workspace member `irwm-data-stats`, depending only on `irwm`.
- It runs on CPU with no network apart from W&B, and finishes in < 10 min on ctdesk236.

**FR11 — Results note (`spec/2026-10-08-flir-readers-class-mapping/stats.md`).**
- `flir_stats.md` from the ctdesk236 run, copied into the spec folder. Its header records the git commit, the run dir and the date.

## Public API Summary

```python
# lib/irwm/data/classes.py
CLASS_NAMES: tuple[str, ...]            # ("person", "bike", "car", "motor", "light", "sign")
NUM_CLASSES: int                        # 6
CATEGORY_ID_TO_LABEL: dict[int, int]    # {1: 0, 2: 1, 3: 2, 4: 3, 10: 4, 12: 5}
LABEL_TO_CATEGORY_ID: dict[int, int]
REF_WIDTH: int; REF_HEIGHT: int         # 640, 512
def label_of(category_id: int) -> int | None: ...
def normalised_area(w: float, h: float, img_w: int, img_h: int) -> float: ...
def size_bucket(area: float) -> Literal["S", "M", "L"]: ...

# lib/irwm/data/flir.py
Modality = Literal["rgb", "thermal"]
Split = Literal["train", "val", "test"]
FLIR_DIRNAME: str                       # "teledyne_flir_adas_v2"
def flir_root() -> Path: ...
def split_dir(modality: Modality, split: Split, root: Path | None = None) -> Path: ...
def load_rgb_thermal_map(root: Path | None = None) -> dict[str, str]: ...

@dataclass(frozen=True)
class FlirRecord: ...                   # fields per FR4

class FlirSplit:
    modality: Modality; split: Split; root: Path; split_dir: Path; records: list[FlirRecord]
    @classmethod
    def load(cls, modality: Modality, split: Split, root: Path | None = None) -> "FlirSplit": ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[FlirRecord]: ...
    def by_image_id(self, image_id: int) -> FlirRecord: ...
    def image_ids(self) -> np.ndarray: ...              # int64 [N]
    def video_tags(self) -> dict[str, list[str]]: ...   # test only
    def to_coco_gt(self, area: Literal["normalised", "native"] = "normalised") -> dict: ...

class FlirImageDataset(torch.utils.data.Dataset):
    def __init__(self, split: FlirSplit, transform: Callable[[np.ndarray], Any] | None = None): ...
    def __getitem__(self, i: int) -> dict: ...          # {"image": u8 [H,W] | [H,W,3], "image_id": int}

# lib/irwm/data/stats.py
COCO80_NAME: dict[str, str | None]
def compute_stats(splits: Sequence[FlirSplit], *, check_files: bool = True) -> dict: ...
def render_markdown(stats: dict) -> str: ...
```

## Dependencies

| Item | Direction | Notes |
|---|---|---|
| `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/FLIR_ADAS_v2/<split_dir>/{coco.json,data/*.jpg}` | read | ~30.8k JPEGs, 6 split dirs. On ctdesk236: `IRWM_DATA_ROOT=/home/leonardo.ulloa/projects/data` |
| `.../FLIR_ADAS_v2/video_*_test/index.json` | read | `videos[].tags` only (36 MB per modality) |
| `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/rgb_to_thermal_vid_map.json` | read | 3749 entries |
| `$IRWM_OUTPUT_ROOT/data_stats/<date>/<time>/flir_stats.{json,md}` | write | Hydra run dir |
| W&B project `ir-world-model`, group `data_stats` | write | `WANDB_MODE` honoured |
| `pillow` | new dep | Declared explicitly in `lib/pyproject.toml` (`pillow>=10`). It is already in `uv.lock` (12.3.0) through torchvision |
| `projects/data_stats/pyproject.toml` | new workspace member | `irwm-data-stats`, depends on `irwm`. Added to the root `pyproject.toml` dependencies and `[tool.uv.sources]` |
