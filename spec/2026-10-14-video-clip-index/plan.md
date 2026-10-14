# F3 · Video clip index: Plan

## Context and Design Decisions

- **The index stores references, never data** (user decision, 2026-10-14). The user asked explicitly not to copy the test set again. A clip row holds frame indices, `image_id`s and file basenames. The only label-derived values are two box counts per clip (FR4). Pixels come from `FlirImageDataset` and features from the F5 cache, both keyed by `image_id`.
- **One shared index for both modalities** (user decision). One row is one time window, with the RGB and the thermal references of the same time-synced frames. `clip_id` 15 is therefore the same window in both modalities, and cross-modal lookup needs no mapping. The rejected alternatives: two per-modality files with a cross-reference column (two numberings to keep in sync), and string keys only (consumers would still need an integer position).
- **`image_id` cannot identify a clip across modalities** (found on the real data, 2026-10-14). Only 338 of the 3749 mapped pairs have equal `image_id`s. Ids follow the `video_id` alphabetical order inside each `coco.json`, and paired videos have unrelated random ids. For example, RGB `RMxN6a4CcCeLGu4tA` pairs with thermal `ZAtDSNuZZjkZFvMAo`, so the sequences sort differently. That is why each row stores both modalities' ids, and why the clip order follows RGB alone (FR3).
- **Pairing facts the check relies on** (F2 plan, "Test sequences pair one-to-one"; re-checked 2026-10-14). The map has 3749 entries, is a bijection, keeps `frame_index` equal, and pairs 8 RGB videos with 8 thermal videos one-to-one. Within one video, `image_id` increases with `frame_index`. Any violation is a broken download, so the check fails hard (user decision) instead of dropping frames.
- **Clip stride 10** (user decision). With span 46 frames this gives 330 clips. The rejected options: stride 1 (3254 clips, ≈ 92 GB of F11 predictions per modality), stride 5 (655 clips, ≈ 18 GB), stride 46 (75 non-overlapping clips, wide CIs, R2). At stride 10, F11 needs ≈ 330 × 6 × 4.7 MB ≈ 9.3 GB per modality. Clips still overlap (10 < 46), so F6 must resample by sequence or segment, not by clip. Hence the `rgb_video_id` and `segment_index` columns.
- **Frame stride 5, 4 context + 6 future, 30 FPS** (Roadmap F3, F1 audit). The FPS comes only from the README (`teledyne_flir_adas_v2/README.txt`, line 20: "30 frames per second (FPS)"). It cannot be verified because the test split has no per-frame timestamps (F2 plan). FR5 records this in `params.fps_source`.
- **Clip starts are anchored at each segment start**, `s = a + k·clip_stride`. The alternative, a global grid (`s % clip_stride == 0`), was rejected: segments start at arbitrary indices (745, 368, 6393, …) and the grid would waste up to 9 frames per segment. The anchor is deterministic given the data, which is all stability needs.
- **Order: RGB `video_id` ascending (Python `str` sort), then start frame.** RGB is chosen because RGB `image_id`s follow the same order, so `clip_id` grows with the first RGB `image_id`. The order is fixed once and documented. Changing it later would renumber everything.
- **Ids are assigned before the evaluable filter, which is a flag** (user decisions). Clips whose 6 future frames have no kept box in RGB **or** in thermal are not evaluable. On the real data that is 26 of 330 clips (7 empty in RGB, 19 in thermal, 0 in both), leaving 304. Keeping them in the index means `clip_id` never depends on labels. Consumers that score detection use `ClipIndex.evaluable()`. The rejected alternatives: renumbering after filtering, and dropping only clips empty in both modalities (0 clips today).
- **Kept boxes only count towards "non-empty"**. This is `len(record.boxes)`, the 6 classes. Ignore regions (F2, TechStack §3.4) are not targets, so a frame with only a truck box counts as empty.
- **The clip set is committed to the repo** (user decision). It lives at `lib/irwm/data/clip_sets/<clip_set>.json`. Git guarantees that clip 15 never changes, and every machine (laptop, ctcomp3) reads the same file without the raw dataset. ctcomp3 does not have the data yet (F2 deferral).
  - **Gotcha:** `.gitignore` ignores `*.json` globally (last line). Without the `!lib/irwm/data/clip_sets/*.json` exception, the file is silently not committed.
  - Hatchling builds the `irwm` wheel with `packages = ["irwm"]` (`lib/pyproject.toml`) and respects `.gitignore`. The negation makes the JSON part of the package. The editable install reads it from the source tree anyway.
- **The clip set file is deterministic and self-describing.** It contains its parameters and the SHA-256 of its 3 input files. It contains no timestamp or git commit, so an `@pytest.mark.data` test can rebuild it and compare bytes. Provenance (commit, config, date) goes to the run-dir report.
- **Two guards keep the committed file stable.** `write_clip_index` refuses to overwrite a differing file (FR6). A CPU test pins the file's SHA-256 (FR9). A new stride or filter therefore produces a new `clip_set` name and a new file.
- **The clip set name encodes the window parameters.** `flir_test_fs5_c4f6_cs10` = frame stride 5, 4 context, 6 future, clip stride 10. It is the `<clip_set>` of the prediction cache path (TechStack §4.3).
- **JSON with one clip per line.** No Parquet library is in `uv.lock`, so JSON adds no dependency. One clip per line keeps git diffs readable. The estimated size is ≈ 0.5 MB, mostly the 20 file basenames per clip.
- **The API is `ClipIndex` plus a frozen `Clip` dataclass** (user decision). Metadata only: it is the clip-level counterpart of `FlirSplit`. A pixel `Dataset` over clips was rejected because F5 caches the features anyway.
- **The entry point is a new project, `projects/clip_index/`** (user decision). It mirrors `projects/data_stats/`: `pyproject.toml` with `package = false`, `conf/`, `load_env()` before `@hydra.main` (`projects/data_stats/stats.py:27`), a W&B table, and a printed run dir. The logic lives in `lib/irwm/data/clips.py`, so tests never import `projects/` (TechStack §2).
- **The builder does not read pixels.** `FlirSplit.load` parses `coco.json` only (`lib/irwm/data/flir.py:133-210`). Synthetic tests therefore need only `coco.json` files and a map, with no JPEGs.
- **Record fields used:** `FlirRecord.video_id`, `frame_index`, `image_id`, `file_name` (`"data/<basename>"`; the clip stores `Path(file_name).name`), and `boxes` (`lib/irwm/data/flir.py:85-107`). The map keys and values are basenames (`load_rgb_thermal_map`, `lib/irwm/data/flir.py:75-82`).
- **Reminder (TechStack §6):** the only path from the environment is `cfg.flir.root`. The clip set path comes from the package location (`irwm.data.__file__`), never from a hard-coded string.

## Step 1 — Library module `lib/irwm/data/clips.py`
Write the module docstring: purpose, the "references only, shared across modalities" rule, and a pointer to this spec folder. Then implement, in this order:
1. **`check_pairing`.** Build `basename → record` dicts per split with `Path(r.file_name).name`. Walk the map and collect violations into a list of strings: unknown key or value, duplicate value, size mismatch, `frame_index` mismatch, RGB video mapped to more than one thermal video and the reverse. Raise a single `ValueError` if the list is non-empty. Return `sorted(set(pairs))`.
2. **`find_segments`.** Sort, reject duplicates, then split where `b != a + 1`.
3. **`clip_set_name`** and **`build_clip_index`.** For each pair: index the records of both modalities by `frame_index`; take the segments of the RGB indices; generate windows per FR3; fill the per-modality lists. Count kept future boxes with `len(rec.boxes)` and set `evaluable`. Assign `clip_id` with a running counter over the ordered windows. Build the `pairs` entries with `n_clips` and `n_evaluable`. Cast every numpy value to `int` so the dict is JSON-native.
4. **`dump_clip_index`.** Write the head with `json.dumps({k: v for k, v in doc.items() if k != "clips"}, indent=2)`. Remove its closing `}` and append `,\n  "clips": [\n`. Join the clip lines with `",\n"`, each indented by 4 spaces and written with `json.dumps(clip, separators=(", ", ": "))`. Close with `\n  ]\n}\n`. For an empty `clips` list, write `"clips": []`. Check `json.loads(text) == doc` once inside the function and raise `AssertionError` if not, since this is a bug guard.
5. **`clip_set_path`, `write_clip_index`, `file_sha256`** per FR6. Read the existing text with `encoding="utf-8"` and compare strings.
6. **`Clip`** and **`ClipIndex`** per FR7. `ClipIndex.load` validates the document, then builds `Clip`s with tuples. `_by_key` is a dict built once. A private `_check_modality(m)` raises the `ValueError`.
7. **`summarise`** and **`render_markdown`** per FR10. Render one table of totals, one table per pair (`rgb_video_id`, `thermal_video_id`, `n_frames`, segments, `n_clips`, `n_evaluable`, `clip_id` range), and one per-modality table. Add a sentence saying that the counts are identical for both modalities because the index is shared.

Covers: FR1–FR7, FR10.

## Step 2 — `projects/clip_index/` entry point
- **`projects/clip_index/pyproject.toml`:** a copy of `projects/data_stats/pyproject.toml` with `name = "irwm-clip-index"` and the description "Video clip index over the FLIR test split (F3)."
- **Workspace registration:** add `irwm-clip-index` to the root `pyproject.toml` `dependencies` (line 6) and to `[tool.uv.sources]` as `{ workspace = true }`. Then run `uv lock`.
- **`conf/build.yaml`:** the FR8 keys, with a comment on `install` ("writes into lib/irwm/data/clip_sets/; refuses to overwrite a different file").
- **`build.py`:**
  1. call `load_env()` at import time;
  2. load `FlirSplit.load("rgb", "test", root)`, `FlirSplit.load("thermal", "test", root)` and `load_rgb_thermal_map(root)`;
  3. compute `inputs` with `file_sha256` on `split_dir(m, "test", root) / "coco.json"` and `root / "rgb_to_thermal_vid_map.json"`;
  4. build the document, then write the three run-dir files per FR8;
  5. log `wandb.Table(columns=[…pair columns…])` and call `finish()`;
  6. if `cfg.install`, call `write_clip_index` on `clip_set_path`;
  7. print the run dir, the path and the SHA-256.
- **Docstring:** the run commands from Step 5.

Covers: FR8.

## Step 3 — `.gitignore` exception
Append `!lib/irwm/data/clip_sets/*.json` on a new line right after the `*.json` line, with a one-line comment: "committed clip sets (F3): small reference indices, not data". Check it with `git check-ignore lib/irwm/data/clip_sets/x.json` (no `-v`: with `-v` git also prints the matching `!` line and exits 0), which must print nothing (exit code 1).

Also append `lib/irwm/data/clip_sets/*.json -text` to `.gitattributes`, with a comment. The repo sets `* text=auto`, so without it a Windows checkout with `core.autocrlf=true` would write CRLF and break the pinned SHA-256 (FR9). Check it with `git check-attr text <file>`, which must print `text: unset`.

Covers: FR9.

## Step 4 — Tests `tests/test_clip_index.py`
- **Fixture `fake_video_test(irwm_env)`.** It writes `coco.json` for `video_rgb_test` and `video_thermal_test` and the map under `$IRWM_DATA_ROOT/teledyne_flir_adas_v2`. No JPEGs. Use the `categories` list of `tests/test_flir_data.py` (car id 3 kept, truck id 8 ignored), copied locally, not imported. Content:
  - RGB `vidA`: frames `0–4` and `7–12` (one gap). RGB `vidB`: frames `0–9`.
  - Thermal `tvZ` pairs with `vidA` and `tvY` with `vidB`. The thermal alphabetical order is reversed relative to RGB, so ordering by RGB is tested. Thermal `image_id`s are numbered `tvY` first, so they differ from the RGB ids.
  - Every frame has one car box in both modalities, except thermal `tvY` frame 6, which has only a truck box.
  - Test parameters: `frame_stride=2, n_context=2, n_future=1, clip_stride=2`, so `T = 3` and span 5.
  - Expected result: 5 clips. Keys in order: `vidA:000000`, `vidA:000007`, `vidB:000000`, `vidB:000002`, `vidB:000004`. Only clip 3 is non-evaluable: its future frame is `vidB`/`tvY` frame 6.
- **Error cases** are made by editing the fixture's map or `coco.json` in place, as `_edit_coco` does in `tests/test_flir_data.py`.
- **`COMMITTED_SHA256`.** A module constant holding the SHA-256 of the committed clip set, filled in Step 6. The committed-file tests (Group 6) load it with `ClipIndex.load()` and need no data. They are skipped with a clear reason only while the file does not exist yet, before Step 5.
- **Real-data tests** use `@pytest.mark.data` and read `IRWM_DATA_ROOT`, skipping when it is unset, like the `real` fixture in `tests/test_flir_data.py:465`.

Cover every item of `validation.md` Groups 1–7.

Covers: FR1–FR10.

## Step 5 — Build the clip set on ctdesk236
This step needs the raw data, which is only on ctdesk236 (`IRWM_DATA_ROOT=/home/leonardo.ulloa/projects/data`, F2). If the implementing session runs on ctdesk236 with that variable set in `.env`, it runs the commands itself. Otherwise it is a **(user)** step.
1. `uv sync`
2. `uv run python projects/clip_index/build.py install=true`
3. `uv run pytest tests/test_clip_index.py -m data -v`
4. `uv run python projects/clip_index/build.py`, a second run without `install`. Then check with `cmp` that `<run_dir>/flir_test_fs5_c4f6_cs10.json` is byte-identical to the committed copy.

The output is the printed run dirs and SHA-256, plus the pytest summary line. When this is a user step, the user returns them.

Covers: FR8, FR9.

## Step 6 — Pin the hash and write up results
1. Set `COMMITTED_SHA256` in `tests/test_clip_index.py` to the SHA-256 printed in Step 5. Run the full CPU suite.
2. Copy `clip_index_report.md` from the first run dir to `spec/2026-10-14-video-clip-index/clips.md`. Prepend a provenance header: git commit, machine, run dir, date, `clip_set_sha256`, input SHA-256s, the pytest summary line.
3. Check the numbers against the Data Validity items in `validation.md`.
4. Do not edit `Roadmap.md` or `TechStack.md`. Committing the clip set file is part of the user's normal commit of this feature.

Covers: FR9, FR10.

## Implementation Order
1. Step 1 — Library module `lib/irwm/data/clips.py`
2. Step 3 — `.gitignore` exception
3. Step 4 — Tests → `uv run pytest tests/test_clip_index.py`
4. Step 2 — `projects/clip_index/` entry point
5. Step 5 — Build the clip set on ctdesk236
6. Step 6 — Pin the hash and write up results → full suite `uv run pytest`, which must still pass every F0, F1 and F2 test

## Amendments

- **2026-10-14 — `.gitattributes` rule for committed clip sets** (user decision). The repo's `.gitattributes` sets `* text=auto`. On the Windows laptop with `core.autocrlf=true`, the clip set would be checked out with CRLF line ends, its SHA-256 would change and the Group 6 hash test would fail. Added `lib/irwm/data/clip_sets/*.json -text`. Changed: FR9 (new bullet), Scope, Dependencies, Step 3, and a new Group 6 item with its test `TestCommitted.test_no_eol_conversion`.
- **2026-10-14 — Step 3 check command corrected** (user decision). `git check-ignore -v` also prints a match on the `!` negation line and exits 0, so the plan's check could never pass as written. The check is now without `-v`, as the Group 6 item already had it. No code change.
