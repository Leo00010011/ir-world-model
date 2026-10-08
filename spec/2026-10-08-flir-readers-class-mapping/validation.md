# F2 · FLIR ADAS v2 readers and class mapping: Validation

Expected real-data values were measured from the as-downloaded `coco.json` files during spec writing (2026-10-08, ctdesk236). Any mismatch means the reader is wrong or the data copy differs. Investigate before relaxing a value.

## Code Correctness (`tests/test_flir_data.py`, CPU, no data, synthetic `fake_flir` tree)

### Group 1 — Root and config
- [ ] With `IRWM_DATA_ROOT=<tmp>`, `flir_root()` → `<tmp>/teledyine_flir_adas_v2`, and `split_dir("thermal", "test")` → `<tmp>/teledyine_flir_adas_v2/FLIR_ADAS_v2/video_thermal_test`.
- [ ] `split_dir("ir", "train")` → `ValueError` whose message lists `rgb` and `thermal`. `split_dir("rgb", "video")` → `ValueError` whose message lists `train`, `val` and `test`.
- [ ] `FlirSplit.load` on a root without `coco.json` → `FileNotFoundError`, and the message contains the full missing path.
- [ ] Composing `projects/data_stats/conf/stats.yaml` with Hydra's compose API and `IRWM_DATA_ROOT=<tmp>` → `cfg.flir.root == str(flir_root())`. This also proves `FLIR_DIRNAME` equals the yaml string.
- [ ] `grep -rn "teledyine" lib/ projects/` → matches only in `lib/irwm/data/flir.py` (the `FLIR_DIRNAME` line) and `lib/irwm/conf/flir/default.yaml`. No absolute paths anywhere.

### Group 2 — Label space and sizes
- [ ] `CLASS_NAMES == ("person", "bike", "car", "motor", "light", "sign")`, `NUM_CLASSES == 6`, `CATEGORY_ID_TO_LABEL == {1: 0, 2: 1, 3: 2, 4: 3, 10: 4, 12: 5}`, and `LABEL_TO_CATEGORY_ID` is its exact inverse.
- [ ] `label_of(8)` (truck), `label_of(6)` (bus), `label_of(79)` (other vehicle) and `label_of(999)` → `None`.
- [ ] `normalised_area(10, 10, 640, 512) == 100.0`, and `normalised_area(20, 20, 1280, 1024) == 100.0`.
- [ ] `size_bucket(1023.9) == "S"`, `size_bucket(1024) == "M"`, `size_bucket(9215.9) == "M"`, `size_bucket(9216) == "L"`.
- [ ] No coarse mapping: an `ast` walk over the module-level assignments of `classes.py` and `flir.py` finds no name containing `coarse`, `vehicle`, `pedestrian` or `cyclist`.

### Group 3 — `FlirSplit`
- [ ] Fake thermal test split → `len(split)` equals the number of fake `coco.json` images, including the image without annotations. `image_ids()` is `int64` and strictly ascending.
- [ ] A record with one car, one person and one truck → `boxes.shape == (2, 4)`, `labels.tolist() == [2, 0]` in file order, `ignore_boxes.shape == (1, 4)`, `ignore_category_ids.tolist() == [8]`. Dtypes are `float32`, `int64`, `float32`, `int64`.
- [ ] The image without annotations → `boxes.shape == (0, 4)`, `labels.shape == (0,)`, `ignore_boxes.shape == (0, 4)`.
- [ ] Test-split records → `track_ids` equal the fake `track_id`s. Train-split records → all `-1`.
- [ ] `video_id` and `frame_index` are parsed from `data/video-abc123-frame-000042-xyz.jpg` → `("abc123", 42)`.
- [ ] Each malformed input → `ValueError` naming the image or annotation:
  - a `file_name` that doesn't match the pattern;
  - `extra_info.video_id` ≠ the parsed id;
  - a duplicate `image_id`;
  - an annotation `category_id` not in `categories`;
  - a box with `w == 0`.
- [ ] `by_image_id(<existing>)` → the matching record. `by_image_id(-5)` → `KeyError`.
- [ ] `video_tags()` on the test split → the fake `{video_id: tags}`. On the train split → `ValueError`.
- [ ] `load_rgb_thermal_map(root)` → the fake map dict, unchanged.

### Group 4 — COCO GT export
- [ ] Fake split with K kept boxes and I ignore boxes → `len(gt["annotations"]) == K + 6 * I`. Exactly `6 * I` of them have `iscrowd == 1`, and for each ignore box the 6 copies cover category ids `{1, 2, 3, 4, 10, 12}`.
- [ ] Annotation ids are unique. Kept boxes keep their original ids. Ignore copies start at `max(original id) + 1`.
- [ ] `gt["categories"]` → the 6 entries `{id, name}` with FLIR ids and names.
- [ ] `area="normalised"` on an 80×64 RGB image with a 10×10 box → `area == 100 * 640 * 512 / (80 * 64) == 6400.0`. `area="native"` → `100.0`.
- [ ] `json.dumps(gt)` succeeds: no numpy types.
- [ ] `pycocotools` is not a dependency yet (F6), so the export is checked structurally only.

### Group 5 — `FlirImageDataset`
- [ ] Thermal item → `image.dtype == uint8`, `image.shape == (48, 64)`. RGB item → `(64, 80, 3)`. `image_id` equals `split.records[i].image_id`.
- [ ] `transform=lambda a: a.shape` → `item["image"]` equals the shape tuple.
- [ ] A fake thermal JPEG saved in mode `RGB` → `ValueError` mentioning the mode. A `coco.json` `width` that differs from the JPEG → `ValueError` mentioning the size.
- [ ] `pickle.loads(pickle.dumps(ds))[0]` gives the same image as `ds[0]`, which shows the dataset is DataLoader-worker safe. `DataLoader(ds, batch_size=None, num_workers=0)` iterates every item.

### Group 6 — Stats
- [ ] `compute_stats` on the fake tree → images per split, kept and ignore counts per class, the orphan file name, and per-sequence gap counts all equal hand-computed fake values. A fake sequence with frames `[0, 1, 2, 5, 6]` → 1 gap of size 3, segments `[3, 2]`.
- [ ] Fake JPEG size ≠ `coco.json` → `file_integrity` lists that image as a mismatch. `check_files=False` → there is no `file_integrity` key, and no file is opened (monkeypatch `PIL.Image.open` to raise).
- [ ] `COCO80_NAME` has exactly the 16 FLIR names. `COCO80_NAME["sign"] is None` and `COCO80_NAME["motor"] == "motorcycle"`. `compute_stats` on a fake split with a box in a raw category named `unicorn` → `KeyError` naming `unicorn`.
- [ ] `render_markdown(stats)` contains one `##` heading per FR8 section and the 6-class table. Every one of the 16 raw class names appears in the per-class table.
- [ ] `json.dumps(compute_stats(...))` succeeds.

### Group 7 — Real data (`@pytest.mark.data`, run on ctdesk236 with `IRWM_DATA_ROOT=/home/leonardo.ulloa/projects/data`)
- [ ] **Image counts:** RGB train/val/test = `10318 / 1085 / 3749`; thermal = `10742 / 1144 / 3749`.
- [ ] **Kept / ignore box counts:**

  | modality | train | val | test |
  |---|---|---|---|
  | RGB | `163856 / 5318` | `16502 / 407` | `81221 / 3565` |
  | thermal | `169422 / 5618` | `16305 / 391` | `58685 / 3632` |

- [ ] **Kept boxes per class, train/val/test:**

  | class | RGB | thermal |
  |---|---|---|
  | person | `35007 / 3223 / 11278` | `50478 / 4470 / 12323` |
  | bike | `7560 / 193 / 446` | `7237 / 170 / 113` |
  | car | `71281 / 7285 / 30893` | `73623 / 7133 / 30517` |
  | motor | `1837 / 77 / 3573` | `1116 / 55 / 3314` |
  | light | `18640 / 2143 / 17818` | `16198 / 2005 / 6758` |
  | sign | `29531 / 3581 / 17213` | `20770 / 2472 / 5660` |

- [ ] **Images with no annotation at all:** RGB `265 / 16 / 0`; thermal `264 / 16 / 256`. **Images with no kept box:** RGB `280 / 16 / 98`; thermal `280 / 19 / 256`.
- [ ] **Normalised S/M/L of kept boxes:**

  | modality | train | val | test |
  |---|---|---|---|
  | RGB | `125708 / 32608 / 5540` | `13499 / 2476 / 527` | `72480 / 7100 / 1641` |
  | thermal | `126045 / 36650 / 6727` | `12218 / 3395 / 692` | `50312 / 7162 / 1211` |

  Thermal normalised equals thermal native.
- [ ] **Test sequences:** 8 `video_id`s per modality, with frame counts `{1033, 565, 540, 428, 411, 338, 226, 208}` in both. `to_coco_gt()` on thermal test → `58685 + 6·3632 = 80477` annotations; on RGB test → `81221 + 6·3565 = 102611`.
- [ ] **Raw classes:** the union of raw classes with ≥ 1 box over all 6 splits has exactly 16 names: person, bike, car, motor, bus, train, truck, light, hydrant, sign, dog, deer, skateboard, stroller, scooter, other vehicle.
- [ ] **Decoding:** `FlirImageDataset` decodes thermal test `image_id 0` to shape `(512, 640)` and RGB test `image_id 0` to `(1024, 1224, 3)`.

## Data Validity (`flir_stats.json` / `stats.md` from the Step 8 run on ctdesk236)
- [ ] **Every Group 7 number matches the report exactly.** A mismatch means the run read a different data copy or the code diverged; Step 9 does not proceed.
- [ ] **File integrity:** 0 size mismatches and 0 mode mismatches over all 30,787 records. Exactly 1 orphan: `images_rgb_train/data/video-GNhgScttHWdvYkPke-frame-000000-JQkmaAp6qjNJu4w5H.jpg`. If any mismatch appears, F4/F5 cannot trust the `coco.json` sizes. Report it to the user.
- [ ] **Image sizes:** thermal is 640×512 for all images. RGB test is `{1224×1024: 2151, 1024×768: 1598}`. RGB train is `{1800×1600: 5015, 1224×1024: 1992, 2048×1536: 1886, 1280×1024: 1378, 720×480: 47}`.
- [ ] **Sequence gaps:** in RGB `RMxN6a4CcCeLGu4tA` and thermal `ZAtDSNuZZjkZFvMAo`, 2 gaps of sizes 271 and 178. In RGB `msNEBxJE5PPDqenBM` and thermal `SCiKdG3MqZfiE292B`, 1 gap of size 1253. All other test sequences have 0 gaps.
- [ ] **Imbalance flags:**
  - `absent_test` is true in both modalities for exactly bus, train, deer, skateboard, stroller and scooter;
  - `coco80` is false for exactly sign, deer, stroller, scooter and other vehicle;
  - none of the 6 selected classes is `absent_test`.
- [ ] **Modality comparison:** the thermal/RGB `test` ratio is `0.33 ± 0.01` for sign and `0.38 ± 0.01` for light. This records the annotation asymmetry on time-synced frames.
- [ ] **Runtime:** the stats run finishes in < 10 min on ctdesk236, as measured from the Hydra log timestamps.

## Data Architecture Integrity
- [ ] **Reproducibility:** two stats runs on the same commit → `flir_stats.json` identical except `config`/`git`/run-dir fields (compare after deleting those keys).
- [ ] **Provenance:** `flir_stats.json` contains `git.commit`, `git.dirty` and the resolved `config`, including `flir.root`. `stats.md` has a header with the commit and date.
- [ ] **Read-only data:** after Step 8, `find $IRWM_DATA_ROOT/teledyine_flir_adas_v2 -newer <repo>/spec/2026-10-08-flir-readers-class-mapping/plan.md` → no files.
- [ ] **Writes confined:** the stats run writes only under `$IRWM_OUTPUT_ROOT/data_stats/` and the W&B dir in that run dir. `git status` shows no new untracked files outside `spec/`, `lib/` and `projects/data_stats/`.
- [ ] **Isolation:** `import irwm.data.flir, irwm.data.classes, irwm.data.stats` works without `irwm[reference]` installed. The existing no-heavy-imports test in `tests/test_dinof_audit.py` still passes.
- [ ] **No regression:** `uv run pytest` (default markers) passes all previous tests plus the new non-data tests in < 1 min on CPU.
