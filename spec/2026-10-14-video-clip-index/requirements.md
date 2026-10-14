# F3 · Video clip index: Requirements

## Goal

F3 delivers one small, committed file that names every evaluation clip of the FLIR video test split. It also delivers the code that builds and reads that file. A clip is 4 context and 6 future steps at frame stride 5 (Roadmap F3). The index holds **references only**, never pixels or annotations, so the test set is never copied. Each clip has a stable integer `clip_id`: clip 15 is the same time window every time, on every machine. The index is **shared by both modalities**: one row is one time window, and it lists the RGB and the thermal `image_id`s of the same time-synced frames. So "clip 15 in thermal" and "clip 15 in RGB" are the same row. F11 rolls out these clips, F12 scores them, and F6's bootstrap resamples them.

## Scope

In scope:
- `lib/irwm/data/clips.py`: the RGB↔thermal pairing check, segment detection, clip construction, the evaluable flag, deterministic serialisation, and the `ClipIndex` / `Clip` reader.
- `projects/clip_index/`: a Hydra entry point that builds the index, writes a count report, and optionally installs the index into the repo.
- The committed clip set `lib/irwm/data/clip_sets/flir_test_fs5_c4f6_cs10.json`, plus the `.gitignore` exception that lets git track it and the `.gitattributes` rule that keeps its bytes exact.
- `tests/test_clip_index.py`.
- The results note `spec/2026-10-14-video-clip-index/clips.md` with the clip and sequence counts.

Out of scope:
- Pixels and features. No pixel `Dataset` over clips. Features are looked up by `image_id` in the F5 cache.
- Clips from the `train`/`val` image splits. Those splits are not continuous video (Mission "Dataset").
- Rollouts and the prediction cache (F11). F11 decides whether it rolls out all clips or only the evaluable ones.
- Bootstrap resampling (F6). F6 uses the `rgb_video_id` and `segment_index` columns to resample by sequence or segment.
- Edits to `Roadmap.md` and `TechStack.md`. The user makes them after this feature.

## Functional Requirements

**FR1 — RGB↔thermal pairing check (`irwm.data.clips.check_pairing`).**
- `check_pairing(rgb: FlirSplit, thermal: FlirSplit, rgb_to_thermal: dict[str, str]) -> list[tuple[str, str]]`. It returns the `(rgb_video_id, thermal_video_id)` pairs, sorted by `rgb_video_id`.
- It raises `ValueError` if `rgb.split` or `thermal.split` is not `"test"`, or if `rgb.modality != "rgb"` or `thermal.modality != "thermal"`.
- `rgb_to_thermal` maps file basenames (no `data/` prefix), as returned by `load_rgb_thermal_map`. The check collects **every** violation of these rules, then raises one `ValueError`. The message gives the number of violations and the first 20 of them:
  1. every key is the basename of exactly one RGB test record, and every value is the basename of exactly one thermal test record;
  2. the map is a bijection that covers both splits: `len(map) == len(rgb) == len(thermal)`, and no value appears twice;
  3. each mapped pair has the same `frame_index`;
  4. each RGB `video_id` maps to exactly one thermal `video_id`, and each thermal `video_id` maps to exactly one RGB `video_id`.
- No partial result is returned. Nothing is written when the check fails (user decision: hard fail).

**FR2 — Gap-free segments (`irwm.data.clips.find_segments`).**
- `find_segments(frame_indices: Sequence[int]) -> list[tuple[int, int]]`. It returns the maximal runs of consecutive `frame_index` values (step 1) as inclusive `(first, last)` pairs, in ascending order. The input may be unsorted.
- A duplicate `frame_index` raises `ValueError`. An empty input returns `[]`.
- Segments are computed on the RGB frame indices of each video pair. FR1 guarantees the thermal indices are identical.

**FR3 — Clip construction (`irwm.data.clips.build_clip_index`).**
- `build_clip_index(rgb, thermal, rgb_to_thermal, *, frame_stride: int = 5, n_context: int = 4, n_future: int = 6, clip_stride: int = 10, inputs: dict[str, str] | None = None) -> dict`. It calls `check_pairing` first. Any parameter `< 1` raises `ValueError`.
- `T = n_context + n_future` and `span = frame_stride·(T − 1) + 1` raw frames (46 by default).
- **Windows.** For each pair in `rgb_video_id` order, and each segment `(a, b)` of that pair in ascending order, the clip starts are `s = a + k·clip_stride` for `k = 0, 1, …` while `s + frame_stride·(T − 1) ≤ b`. A clip's frames are `frame_indices = [s + frame_stride·j for j in range(T)]`. The first `n_context` are context, the last `n_future` are future. A clip never spans a gap, because every frame lies in one segment. A segment shorter than `span` gives no clip.
- **Order and ids.** Clips are ordered by (`rgb_video_id` ascending as a Python `str` sort, start `frame_index` ascending). `clip_id` is the position in that order: `0 … N − 1`, contiguous, assigned **before** the evaluable flag (FR4). The thermal video order is never used, because it differs from the RGB order.
- **Key.** `key = f"{rgb_video_id}:{s:06d}"`, for example `BzZspxAweF8AnKhWK:000895`. Keys are unique.
- **Per-modality references.** For each modality, the clip stores the `image_id`s and the file basenames of its `T` frames, looked up by `(video_id, frame_index)` in that modality's `FlirSplit`.

**FR4 — Evaluable flag.**
- `n_future_boxes[m]` is the number of **kept** boxes (`len(record.boxes)`, the 6 classes of `irwm.data.classes`) summed over the `n_future` future frames of modality `m`. Ignore boxes do not count.
- `evaluable = n_future_boxes["rgb"] ≥ 1 and n_future_boxes["thermal"] ≥ 1` (user decision).
- Non-evaluable clips stay in the index with their `clip_id` (user decision: ids never depend on labels). `ClipIndex.evaluable()` gives the evaluable subset (FR7).

**FR5 — Document schema and serialisation (`irwm.data.clips.dump_clip_index`).**
- `build_clip_index` returns a JSON-serialisable `dict` with exactly these top-level keys, in this order:
  - `format_version`: `1`;
  - `clip_set`: `clip_set_name(frame_stride, n_context, n_future, clip_stride)` = `f"flir_test_fs{frame_stride}_c{n_context}f{n_future}_cs{clip_stride}"`;
  - `params`: `{"split": "test", "frame_stride", "n_context", "n_future", "clip_stride", "fps": 30, "fps_source": "FLIR README.txt: 30 FPS; not verifiable from the data, the test split has no per-frame timestamps (F2)", "evaluable_rule": "kept future boxes >= 1 in both modalities"}`;
  - `inputs`: the `inputs` argument (`{}` if `None`). The entry point passes `{"rgb_coco_sha256", "thermal_coco_sha256", "rgb_to_thermal_map_sha256"}`;
  - `pairs`: one entry per FR1 pair, in `rgb_video_id` order: `{"rgb_video_id", "thermal_video_id", "n_frames", "segments": [[first, last], …], "n_clips", "n_evaluable"}`;
  - `clips`: one entry per clip, in `clip_id` order, with keys in this order: `clip_id`, `key`, `rgb_video_id`, `thermal_video_id`, `segment_index` (0-based within the pair), `frame_indices`, `rgb_image_ids`, `thermal_image_ids`, `rgb_file_names`, `thermal_file_names`, `n_future_boxes` (`{"rgb": int, "thermal": int}`), `evaluable`.
- The document holds **no** timestamp, git commit, hostname or absolute path, so rebuilding from the same data gives the same bytes. Provenance goes to the run-dir report (FR8).
- `dump_clip_index(doc: dict) -> str` is deterministic:
  - every top-level key except `clips` is written by `json.dumps(..., indent=2)`;
  - `clips` is written as a JSON list with **one clip per line**, each line `json.dumps(clip, separators=(", ", ": "))`;
  - the text ends with one `"\n"`;
  - `json.loads(dump_clip_index(doc)) == doc`.

**FR6 — Paths and the write guard.**
- `clip_set_path(clip_set: str) -> Path` = `Path(irwm.data.__file__).parent / "clip_sets" / f"{clip_set}.json"`. It is a repo path, not a data path, so it does not use an env var.
- `write_clip_index(doc: dict, path: Path) -> bool` writes `dump_clip_index(doc)` as UTF-8 and creates parent directories. If `path` already holds exactly that text, it writes nothing and returns `False`. If `path` holds different text, it raises `FileExistsError` and leaves the file untouched. Otherwise it writes and returns `True`. A committed clip set can therefore never be renumbered silently.
- `file_sha256(path: Path) -> str` returns the hex SHA-256 of the file bytes.

**FR7 — Reader API (`irwm.data.clips.ClipIndex`, `irwm.data.clips.Clip`).**
- `DEFAULT_CLIP_SET = "flir_test_fs5_c4f6_cs10"`.
- `ClipIndex.load(clip_set: str = DEFAULT_CLIP_SET, path: Path | None = None) -> ClipIndex` reads `path`, or `clip_set_path(clip_set)` if `path` is `None`. It needs no dataset and no env var. It raises:
  - `FileNotFoundError` if the file is missing;
  - `ValueError` if `format_version != 1`, if any `clip_id` differs from its position, if a key is duplicated, or if a per-frame list does not have length `n_context + n_future`.
- `ClipIndex` attributes and methods:
  - `clip_set: str`, `params: dict`, `pairs: list[dict]`;
  - `__len__`, `__iter__` (in `clip_id` order);
  - `__getitem__(clip_id: int) -> Clip`. It raises `IndexError` for `clip_id < 0` or `≥ len`. Negative indices are refused because a `clip_id` is an identifier, not a position;
  - `by_key(key: str) -> Clip`, which raises `KeyError` for an unknown key;
  - `evaluable() -> list[Clip]`, the clips with `evaluable == True`, in `clip_id` order.
- `Clip` is a frozen dataclass:
  - fields `clip_id: int`, `key: str`, `rgb_video_id: str`, `thermal_video_id: str`, `segment_index: int`, `n_context: int`, `frame_indices: tuple[int, ...]`, `rgb_image_ids: tuple[int, ...]`, `thermal_image_ids: tuple[int, ...]`, `rgb_file_names: tuple[str, ...]`, `thermal_file_names: tuple[str, ...]`, `n_future_boxes_rgb: int`, `n_future_boxes_thermal: int`, `evaluable: bool`;
  - `video_id(modality) -> str`;
  - `image_ids(modality) -> np.ndarray`, `int64 [T]`;
  - `context_ids(modality) -> np.ndarray`, `int64 [n_context]`;
  - `future_ids(modality) -> np.ndarray`, `int64 [T − n_context]`;
  - `file_names(modality) -> tuple[str, ...]`;
  - `n_future_boxes(modality) -> int`;
  - every method raises `ValueError` for a modality other than `"rgb"` or `"thermal"`.

**FR8 — Entry point (`projects/clip_index/`).**
- `projects/clip_index/build.py` is a `@hydra.main` script with config `conf/build.yaml`: `defaults: [irwm_base, _self_]`, searchpath `pkg://irwm.conf`, `project: clip_index`, `frame_stride: 5`, `n_context: 4`, `n_future: 6`, `clip_stride: 10`, `install: false`.
- It loads both test `FlirSplit`s and the map from `Path(cfg.flir.root)`. It hashes the two test `coco.json` files and the map file with `file_sha256` and passes them as `inputs`. Then it calls `build_clip_index`.
- It writes to the Hydra run dir:
  - `<clip_set>.json`, through `write_clip_index`;
  - `clip_index_report.json`: the FR10 counts, plus `config`, `git` (`irwm.utils.tracking.git_info()`), `inputs` and `clip_set_sha256`;
  - `clip_index_report.md`: the counts as GitHub tables.
- It logs the per-pair table as a `wandb.Table` through `make_wandb_logger(cfg, save_dir=run_dir)`.
- With `install=true` it also calls `write_clip_index(doc, clip_set_path(clip_set))`. A differing existing file raises `FileExistsError`, and the job fails.
- It prints the run dir, the clip set path and its SHA-256.
- It is registered in the uv workspace as `irwm-clip-index`.

**FR9 — Committed clip set.**
- `lib/irwm/data/clip_sets/flir_test_fs5_c4f6_cs10.json` is built by FR8 with the defaults and `install=true`, then committed.
- `.gitignore` gets the exception `!lib/irwm/data/clip_sets/*.json`, placed after the existing `*.json` line.
- `.gitattributes` gets `lib/irwm/data/clip_sets/*.json -text`, so `* text=auto` never converts line ends on checkout (amendment 2026-10-14).
- `tests/test_clip_index.py` pins the file's SHA-256 in the constant `COMMITTED_SHA256`. Any edit to the file fails the CPU test suite.
- File size < 1 MB.

**FR10 — Counts and results note.**
- `summarise(doc: dict) -> dict` returns:
  - `n_pairs`, `n_segments`, `n_clips`, `n_evaluable`;
  - per pair, the FR5 `pairs` fields;
  - per modality, the frames covered by at least one clip, and the count of non-evaluable clips caused by that modality (`n_future_boxes[m] == 0`).
- Because the index is shared, the clip and sequence counts are equal for RGB and thermal. The report states this explicitly.
- `render_markdown(summary: dict) -> str` gives the report as GitHub tables.
- `spec/2026-10-14-video-clip-index/clips.md` holds the rendered report from the real run, with a provenance header.

## Public API Summary

```python
# lib/irwm/data/clips.py
DEFAULT_CLIP_SET: str = "flir_test_fs5_c4f6_cs10"

def check_pairing(rgb: FlirSplit, thermal: FlirSplit, rgb_to_thermal: dict[str, str]) -> list[tuple[str, str]]: ...
def find_segments(frame_indices: Sequence[int]) -> list[tuple[int, int]]: ...
def clip_set_name(frame_stride: int, n_context: int, n_future: int, clip_stride: int) -> str: ...
def build_clip_index(rgb: FlirSplit, thermal: FlirSplit, rgb_to_thermal: dict[str, str], *,
                     frame_stride: int = 5, n_context: int = 4, n_future: int = 6,
                     clip_stride: int = 10, inputs: dict[str, str] | None = None) -> dict: ...
def dump_clip_index(doc: dict) -> str: ...
def clip_set_path(clip_set: str) -> Path: ...
def write_clip_index(doc: dict, path: Path) -> bool: ...
def file_sha256(path: Path) -> str: ...
def summarise(doc: dict) -> dict: ...
def render_markdown(summary: dict) -> str: ...

@dataclass(frozen=True)
class Clip:
    clip_id: int; key: str; rgb_video_id: str; thermal_video_id: str; segment_index: int
    n_context: int; frame_indices: tuple[int, ...]
    rgb_image_ids: tuple[int, ...]; thermal_image_ids: tuple[int, ...]
    rgb_file_names: tuple[str, ...]; thermal_file_names: tuple[str, ...]
    n_future_boxes_rgb: int; n_future_boxes_thermal: int; evaluable: bool
    def video_id(self, modality: Modality) -> str: ...
    def image_ids(self, modality: Modality) -> np.ndarray: ...     # int64 [T]
    def context_ids(self, modality: Modality) -> np.ndarray: ...   # int64 [n_context]
    def future_ids(self, modality: Modality) -> np.ndarray: ...    # int64 [T - n_context]
    def file_names(self, modality: Modality) -> tuple[str, ...]: ...
    def n_future_boxes(self, modality: Modality) -> int: ...

class ClipIndex:
    clip_set: str; params: dict; pairs: list[dict]
    @classmethod
    def load(cls, clip_set: str = DEFAULT_CLIP_SET, path: Path | None = None) -> ClipIndex: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Clip]: ...
    def __getitem__(self, clip_id: int) -> Clip: ...
    def by_key(self, key: str) -> Clip: ...
    def evaluable(self) -> list[Clip]: ...
```

## Dependencies

| Item | Direction | Notes |
|---|---|---|
| `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/FLIR_ADAS_v2/video_{rgb,thermal}_test/coco.json` | read | Through `FlirSplit.load` (F2). Only on ctdesk236, where the data is. |
| `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/rgb_to_thermal_vid_map.json` | read | Through `load_rgb_thermal_map` (F2). |
| `irwm.data.flir` (`FlirSplit`, `FlirRecord`, `Modality`, `load_rgb_thermal_map`) | read (import) | F2, unchanged. |
| `irwm.utils.tracking` (`git_info`, `make_wandb_logger`), `irwm.utils.paths.load_env` | read (import) | F0. |
| `lib/irwm/data/clip_sets/flir_test_fs5_c4f6_cs10.json` | write (once), then read | Committed. < 1 MB. |
| `$IRWM_OUTPUT_ROOT/clip_index/<date>/<time>/` | write | Hydra run dir: the clip set copy and the reports. |
| `.gitignore`, `.gitattributes`, root `pyproject.toml`, `uv.lock` | write | `.gitignore` exception; workspace member `irwm-clip-index`. |
| New deps | — | None. JSON only; no Parquet library is in `uv.lock`. |
