# F3 · Video clip index: Validation

## Code Correctness (`tests/test_clip_index.py`, CPU, no data)

Groups 1–5 use the `fake_video_test` fixture and the test parameters of plan Step 4 (`frame_stride=2, n_context=2, n_future=1, clip_stride=2`), unless stated otherwise.

### Group 1 — Pairing check (FR1)
One corruption can break several rules at once. For example, remapping a key to an already-mapped value also creates a duplicate. So each item checks that the message **contains** the named violation, not that it is the only one.
- [x] Fixture as written → `check_pairing` returns `[("vidA", "tvZ"), ("vidB", "tvY")]`.
- [x] One map entry removed → `ValueError` whose message contains the size mismatch.
- [x] One map value replaced by a basename that does not exist in thermal → `ValueError` naming that basename.
- [x] Two keys mapped to the same thermal basename → `ValueError` (duplicate value).
- [x] One pair remapped to a thermal frame with a different `frame_index` → `ValueError` naming both basenames.
- [x] One `vidA` frame remapped to a `tvY` frame with the same `frame_index` → `ValueError` saying that `vidA` maps to two thermal videos.
- [x] Three corruptions at once (an unknown thermal basename, a `frame_index` mismatch and a missing entry) → one `ValueError`. Its message gives a violation count of at least 3 and names all three.
- [x] `rgb` argument loaded with split `"train"`, or the two splits swapped → `ValueError` before any map check.

### Group 2 — Segments (FR2)
- [x] `find_segments([0,1,2,3,4,7,8,9,10,11,12])` → `[(0, 4), (7, 12)]`. A shuffled input gives the same result.
- [x] `find_segments([5])` → `[(5, 5)]`. `find_segments([])` → `[]`.
- [x] `find_segments([1, 2, 2, 3])` → `ValueError`.

### Group 3 — Clip construction, order and ids (FR3, FR4)
- [x] Fixture → 5 clips with `clip_id` `0..4` and keys `vidA:000000`, `vidA:000007`, `vidB:000000`, `vidB:000002`, `vidB:000004`, in this order.
- [x] Clip 1 → `frame_indices == [7, 9, 11]`, `segment_index == 1`, `thermal_video_id == "tvZ"`. Its `rgb_image_ids` and `thermal_image_ids` equal the fixture's ids for those frames, and the two lists differ from each other.
- [x] No window crosses the `vidA` gap: no clip has a frame in `{5, 6}`, and no clip mixes frames from both `vidA` segments.
- [x] Segment `0–4` with span 5 gives exactly 1 clip, start `0`, because start `2` would end at `6 > 4`.
- [x] `n_future_boxes` → `{"rgb": 1, "thermal": 1}` for clips 0, 1, 2, 4. Clip 3 → `{"rgb": 1, "thermal": 0}` and `evaluable == False`. Every other clip has `evaluable == True`.
- [x] Clip ids do not depend on labels: removing every annotation from thermal `coco.json` → the same 5 keys with the same `clip_id`s, all `evaluable == False`.
- [x] Default parameters on the fixture → 0 clips (span 46 > every segment) and an empty `clips` list, not an error.
- [x] `frame_stride=0`, `n_future=0` or `clip_stride=-1` → `ValueError`.
- [x] `clip_set_name(5, 4, 6, 10) == "flir_test_fs5_c4f6_cs10"`.

### Group 4 — Document and serialisation (FR5, FR6)
- [x] The document's top-level keys are exactly `["format_version", "clip_set", "params", "inputs", "pairs", "clips"]` in this order, and each clip's keys are in the FR5 order.
- [x] `json.loads(dump_clip_index(doc)) == doc`. The text ends with exactly one `"\n"`. The number of lines that start with `    {"clip_id"` equals `len(doc["clips"])`.
- [x] Two builds from the same fixture → identical `dump_clip_index` text.
- [x] The serialised text contains no `tmp_path` string, no git SHA and no ISO date (regex `\d{4}-\d{2}-\d{2}T`).
- [x] The serialised text contains no annotation fields: no `"bbox"`, `"category_id"` or `"track_id"`.
- [x] `write_clip_index` on a new path → returns `True` and creates the parent directories. A second call with the same doc → returns `False` and the file mtime is unchanged. A call with a different doc (other `clip_stride`) → `FileExistsError`, and the file bytes are unchanged.
- [x] `clip_set_path("x") == Path(irwm.data.__file__).parent / "clip_sets" / "x.json"`.

### Group 5 — Reader API (FR7)
- [x] Fixture document written to `tmp_path` → `ClipIndex.load(path=…)` has `len == 5`. `idx[3].key == "vidB:000002"`. `idx.by_key("vidA:000007").clip_id == 1`. `[c.clip_id for c in idx.evaluable()] == [0, 1, 2, 4]`.
- [x] `idx[5]` and `idx[-1]` → `IndexError`. `idx.by_key("nope:000000")` → `KeyError`.
- [x] `idx[1].image_ids("thermal")` is `int64` with shape `[3]`. `context_ids` has shape `[2]` and `future_ids` shape `[1]`. Their concatenation equals `image_ids`.
- [x] `idx[1].video_id("rgb") == "vidA"` and `idx[1].video_id("thermal") == "tvZ"`. `idx[1].file_names("rgb")[0]` starts with `video-vidA-frame-000007-` and has no `data/` prefix.
- [x] `idx[1].image_ids("ir")` → `ValueError`.
- [x] `Clip` is frozen: assigning `idx[0].clip_id = 9` → `dataclasses.FrozenInstanceError`.
- [x] Corrupt files → `ValueError`, in each of these cases: `format_version: 2`, two clips with swapped `clip_id`s, a duplicated key, a 2-element `frame_indices`. A missing file → `FileNotFoundError`.
- [x] `irwm.data.clips` imports without `IRWM_DATA_ROOT` set, and `ClipIndex.load(path=…)` works with `IRWM_DATA_ROOT` unset.

### Group 6 — Committed clip set (FR9; CPU, no data; skipped only until plan Step 5 has run)
- [x] `file_sha256(clip_set_path(DEFAULT_CLIP_SET)) == COMMITTED_SHA256`.
- [x] `ClipIndex.load()` → `len == 330`, `len(evaluable()) == 304`, `len(pairs) == 8`.
- [x] `ClipIndex.load()[15]` → key `BzZspxAweF8AnKhWK:000895`, `thermal_video_id == "4FRnNpmSmwktFJKjg"`, `frame_indices == [895, 900, …, 940]` (step 5). `rgb_image_ids == [150, 155, …, 195]` and `thermal_image_ids == [150, 155, …, 195]`.
- [x] `ClipIndex.load()[296].key == "msNEBxJE5PPDqenBM:006393"` and `ClipIndex.load()[329].rgb_video_id == "msNEBxJE5PPDqenBM"`.
- [x] File size < 1,000,000 bytes.
- [x] `git check-ignore lib/irwm/data/clip_sets/flir_test_fs5_c4f6_cs10.json` exits with code 1 (not ignored).
- [x] `git check-attr text lib/irwm/data/clip_sets/flir_test_fs5_c4f6_cs10.json` → `text: unset` (no line-end conversion on checkout).

### Group 7 — Real data (`@pytest.mark.data`, run on ctdesk236)
- [x] `check_pairing` on the real test splits and map → 8 pairs, no exception.
- [x] `build_clip_index` with defaults and the real input hashes → `dump_clip_index` is byte-identical to the committed file.
- [x] For every clip and every modality, each `image_id` resolves through `FlirSplit.by_image_id`. The resolved record's `video_id`, `frame_index` and basename equal the clip's stored values.
- [x] For every clip, `n_future_boxes[m] == sum(len(split.by_image_id(i).boxes) for i in future_ids(m))` in both modalities.

## Data Validity (`clip_index_report.json` from plan Step 5, copied into `clips.md`)
- [x] **Totals:** `n_pairs == 8`, `n_segments == 11`, `n_clips == 330`, `n_evaluable == 304`. If any of these differs, the window rule or the data differs from what this spec was written against: stop and report.
- [x] **Per pair** (RGB `video_id` → thermal `video_id`: clips / evaluable / `clip_id` range):
  - `BzZspxAweF8AnKhWK` → `4FRnNpmSmwktFJKjg`: 30 / 30 / 0–29
  - `FkqCGijjAKpABetZZ` → `6tLtjdkv5K5BuhB37`: 19 / 19 / 30–48
  - `PGdt7pJChnKoJDt35` → `vbrSzr4vFTm5QwuGH`: 17 / 17 / 49–65
  - `RMxN6a4CcCeLGu4tA` → `ZAtDSNuZZjkZFvMAo`: 91 / 91 / 66–156; segments `[0, 429]`, `[700, 920]`, `[1098, 1479]`
  - `YnfPeH8i2uBWmsSd2` → `ePoikf5LyTTfqchga`: 50 / 43 / 157–206
  - `dvZBYnphN2BwdMKBc` → `t3f7QC8hZr6zYXpEZ`: 52 / 52 / 207–258
  - `hnbGXq3nNPjBbc7CL` → `5RSrbWYu9eokv5bvR`: 37 / 18 / 259–295
  - `msNEBxJE5PPDqenBM` → `SCiKdG3MqZfiE292B`: 34 / 34 / 296–329; segments `[6393, 6556]`, `[7809, 8072]`
- [x] **Non-evaluable causes:** 7 clips with `n_future_boxes["rgb"] == 0`, 19 with `n_future_boxes["thermal"] == 0`, 0 with both.
- [x] **Shared counts:** the report states that the clip and sequence counts are identical for RGB and thermal.
- [x] **FPS provenance:** `params.fps == 30`, and `params.fps_source` names the README and states that it cannot be verified from the data.

## Data Architecture Integrity
- [x] **Stable ids:** two builds (plan Step 5, items 2 and 4) → byte-identical clip set files (`cmp` exits 0), equal to the committed file and to `COMMITTED_SHA256`.
- [x] **No copy of the test set:** the clip set holds only ids, frame indices, basenames and box counts (Group 4 checks the absence of annotation fields). No JPEG or `coco.json` is copied anywhere: `find $IRWM_OUTPUT_ROOT/clip_index -name '*.jpg'` → no output.
- [x] **Provenance:** `clip_index_report.json` records `git.commit`, the resolved `config`, `inputs` (3 SHA-256s) and `clip_set_sha256`. The `inputs` of the committed file equal those in the report.
- [x] **Isolation:** the build writes only into its Hydra run dir under `$IRWM_OUTPUT_ROOT/clip_index/` and, with `install=true`, into `lib/irwm/data/clip_sets/`. Nothing is written under `$IRWM_DATA_ROOT`: its file mtimes are unchanged.
- [x] **No regressions:** `uv run pytest` passes every pre-existing test (F0, F1, F2) plus Groups 1–6.
