# F3 · Video clip index: Results

Provenance:
- **Date / machine:** 2026-10-14, ctdesk236 (CPU), `IRWM_DATA_ROOT=/home/leonardo.ulloa/projects/data`.
- **Git commit:** `4da4fa9` with a dirty tree: the F3 code was not committed yet. The clip set holds no commit, so its bytes do not depend on it.
- **Run dirs:** `$IRWM_OUTPUT_ROOT/clip_index/2026-10-14/19-32-00` (`install=true`, source of this report) and `.../19-32-18` (second build, no install). `cmp` of both run-dir copies against `lib/irwm/data/clip_sets/flir_test_fs5_c4f6_cs10.json` exits 0.
- **`clip_set_sha256`:** `f9b3735c13cb1ab91eb42ac72bd6b1617eea5e3b54cd40624df8137c729a6c65` (574,098 bytes), pinned as `COMMITTED_SHA256` in `tests/test_clip_index.py`.
- **Input SHA-256s** (`inputs` in the clip set and in `clip_index_report.json`, equal):
  - `rgb_coco_sha256`: `44d9b93f415f4ffcdc5dba54180b99a343e21cb8973933abc14fb701f1f01631`
  - `thermal_coco_sha256`: `8da913c6cf57fd148afdd2ec81716e6bc44f1869308b92d6b6280263788d487a`
  - `rgb_to_thermal_map_sha256`: `a9362f1ccb1ee0c54c841484df96ca3bb3aeddc28f634fe4ea2798bc901b9ae3`
- **Tests:** `uv run pytest tests/test_clip_index.py -m data -v` → `4 passed, 48 deselected`. Full CPU suite `uv run pytest` → `144 passed, 16 deselected`.

The rest of this note is `clip_index_report.md` from the first run dir, unchanged.

---

# Clip set `flir_test_fs5_c4f6_cs10`

Split `test`, frame stride 5, 4 context + 6 future frames, clip stride 10. FPS 30 (FLIR README.txt: 30 FPS; not verifiable from the data, the test split has no per-frame timestamps (F2)). Evaluable rule: kept future boxes >= 1 in both modalities.

The index is shared by both modalities: one row is one time window with the RGB and the thermal frames of the same time-synced frames, so the clip and sequence counts below are identical for RGB and thermal.

## Totals

| pairs | segments | clips | evaluable | not evaluable (both empty) |
|---:|---:|---:|---:|---:|
| 8 | 11 | 330 | 304 | 26 (0) |

## Per pair

| rgb_video_id | thermal_video_id | n_frames | segments | n_clips | n_evaluable | clip_id range |
|---|---|---:|---|---:|---:|---|
| BzZspxAweF8AnKhWK | 4FRnNpmSmwktFJKjg | 338 | [745, 1082] | 30 | 30 | 0–29 |
| FkqCGijjAKpABetZZ | 6tLtjdkv5K5BuhB37 | 226 | [0, 225] | 19 | 19 | 30–48 |
| PGdt7pJChnKoJDt35 | vbrSzr4vFTm5QwuGH | 208 | [368, 575] | 17 | 17 | 49–65 |
| RMxN6a4CcCeLGu4tA | ZAtDSNuZZjkZFvMAo | 1033 | [0, 429], [700, 920], [1098, 1479] | 91 | 91 | 66–156 |
| YnfPeH8i2uBWmsSd2 | ePoikf5LyTTfqchga | 540 | [300, 839] | 50 | 43 | 157–206 |
| dvZBYnphN2BwdMKBc | t3f7QC8hZr6zYXpEZ | 565 | [0, 564] | 52 | 52 | 207–258 |
| hnbGXq3nNPjBbc7CL | 5RSrbWYu9eokv5bvR | 411 | [99, 509] | 37 | 18 | 259–295 |
| msNEBxJE5PPDqenBM | SCiKdG3MqZfiE292B | 428 | [6393, 6556], [7809, 8072] | 34 | 34 | 296–329 |

## Per modality

| modality | frames covered by ≥ 1 clip | not evaluable because this modality has 0 future boxes |
|---|---:|---:|
| rgb | 748 | 7 |
| thermal | 748 | 19 |
