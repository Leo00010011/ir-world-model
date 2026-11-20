# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python==4.11.0.86", "python-dotenv"]
# ///
"""Mark contiguous-video segments (and day/night) on the aligned FLIR test set.

Run (needs a display; uses its own env because the project env ships opencv-headless):

    uv run --script projects/segment_annotation/annotate_segments.py [--split test] [--out PATH]

Keys:
    Right / Left    next / previous frame (also l / h)
    s               mark the current frame as the start of a contiguous video,
                    then press d (day) or n (night); Esc cancels
    x               remove the segment start on the current frame
    q / Esc         quit (everything is already saved)

A segment runs from its start id up to the frame before the next start. The JSON is
rewritten atomically (tmp file + fsync + rename) after every change and every move, so a
crash loses nothing and the next run resumes on the last frame you were viewing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

CLASS_NAMES = ("person", "car")  # same label space as irwm.data.aligned
DROPPED = {"bicycle", "dog"}
COLORS = {0: (0, 0, 255), 1: (255, 160, 0)}  # BGR: person red, car blue
W, H = 640, 512
ENTRY_RE = re.compile(r"^FLIR_(\d+)_PreviewData$")
LIST_FILE = {"test": "align_validation.txt"}  # the aligned release calls its test list "validation"
LEFT = {65361, 2424832, 0x1000012, ord("h")}
RIGHT = {65363, 2555904, 0x1000014, ord("l")}
ESC = 27
WIN = "segment annotation"


# ---------------------------------------------------------------- data
def repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "uv.lock").is_file():
            return p
    raise FileNotFoundError("uv.lock not found above this script")


def env_path(var: str) -> Path:
    v = os.environ.get(var, "").strip()
    if not v:
        raise KeyError(f"{var} is not set (put it in .env)")
    return Path(v).expanduser()


def read_ids(root: Path, split: str) -> list[int]:
    ids = []
    for line in (root / LIST_FILE[split]).read_text().splitlines():
        if line.strip():
            m = ENTRY_RE.match(line.strip())
            if not m:
                raise ValueError(f"bad entry {line!r}")
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def parse_voc(path: Path) -> list[tuple[float, float, float, float, int]]:
    out = []
    for obj in ET.parse(path).getroot().iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name in DROPPED:
            continue
        bb = obj.find("bndbox")
        x0, y0, x1, y1 = (float(bb.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax"))
        x0, x1 = min(max(x0, 0), W), min(max(x1, 0), W)
        y0, y1 = min(max(y0, 0), H), min(max(y1, 0), H)
        out.append((x0, y0, x1, y1, CLASS_NAMES.index(name)))
    return out


# ---------------------------------------------------------------- storage
class Store:
    def __init__(self, path: Path, split: str, ids: list[int]):
        self.path, self.split, self.ids = path, split, ids
        self.segments: dict[int, str] = {}  # start_id -> "day" | "night"
        self.cursor_id = ids[0]
        if path.is_file():
            d = json.loads(path.read_text())
            if d.get("split") != split:
                raise ValueError(f"{path} is for split {d.get('split')!r}, not {split!r}")
            self.segments = {int(s["start_id"]): s["time_of_day"] for s in d["segments"]}
            if d.get("cursor_id") in ids:
                self.cursor_id = d["cursor_id"]
            if unknown := set(self.segments) - set(ids):
                raise ValueError(f"{path}: start ids not in the split: {sorted(unknown)[:5]}")

    def save(self) -> None:
        idx = {i: k for k, i in enumerate(self.ids)}
        starts = sorted(self.segments)
        segs = []
        for j, s in enumerate(starts):
            end_idx = idx[starts[j + 1]] - 1 if j + 1 < len(starts) else len(self.ids) - 1
            segs.append({"start_id": s, "end_id": self.ids[end_idx], "time_of_day": self.segments[s],
                         "n_frames": end_idx - idx[s] + 1})
        data = {"split": self.split, "source": LIST_FILE[self.split], "n_frames": len(self.ids),
                "first_id": self.ids[0], "last_id": self.ids[-1],
                "note": "a segment runs from start_id to the frame before the next start_id",
                "segments": segs, "cursor_id": self.cursor_id,
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        dfd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def segment_of(self, image_id: int) -> tuple[int, int, str] | None:
        starts = sorted(s for s in self.segments if s <= image_id)
        if not starts:
            return None
        return len(starts), starts[-1], self.segments[starts[-1]]


# ---------------------------------------------------------------- drawing
def put(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_boxes(img, boxes):
    for x0, y0, x1, y1, lab in boxes:
        p0, p1 = (round(x0), round(y0)), (round(x1), round(y1))
        cv2.rectangle(img, p0, p1, COLORS[lab], 2)
        cv2.putText(img, CLASS_NAMES[lab], (p0[0], max(p0[1] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[lab], 1, cv2.LINE_AA)


def render(root: Path, split: str, ids: list[int], k: int, store: Store, pending: bool) -> np.ndarray:
    i = ids[k]
    stem = f"FLIR_{i:05d}"
    t = cv2.imread(str(root / "JPEGImages" / f"{stem}_PreviewData.jpeg"), cv2.IMREAD_GRAYSCALE)
    t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    c = cv2.imread(str(root / "JPEGImages" / f"{stem}_RGB.jpg"), cv2.IMREAD_COLOR)
    if c.shape[:2] != (H, W):
        c = cv2.resize(c, (W, H))
    boxes = parse_voc(root / "Annotations" / f"{stem}_PreviewData.xml")
    draw_boxes(t, boxes)
    draw_boxes(c, boxes)
    npers = sum(b[4] == 0 for b in boxes)
    ncar = sum(b[4] == 1 for b in boxes)
    # same header as the review videos
    put(t, f"ID {i:05d}   {k + 1}/{len(ids)}", (10, 24))
    put(t, f"{split}   person {npers}  car {ncar}", (10, 50))
    put(c, "RGB", (10, 24))
    # segment state
    seg = store.segment_of(i)
    if seg is None:
        put(t, "no segment yet", (10, 76), color=(160, 160, 160))
    else:
        n, s, tod = seg
        col = (0, 220, 255) if tod == "day" else (255, 120, 60)
        put(t, f"segment {n}/{len(store.segments)}  from {s:05d}  {tod.upper()}", (10, 76), color=col)
    img = np.hstack([t, c])
    if i in store.segments:
        cv2.rectangle(img, (0, 0), (2 * W - 1, H - 1), (0, 255, 0), 6)
        put(img, f"START ({store.segments[i]})", (2 * W - 220, 30), 0.8, (0, 255, 0))
    if pending:
        cv2.rectangle(img, (2 * W // 2 - 300, H // 2 - 40), (2 * W // 2 + 300, H // 2 + 30), (0, 0, 0), -1)
        put(img, "segment start: press  d = day   n = night   (Esc cancels)",
            (2 * W // 2 - 285, H // 2), 0.65, (0, 255, 255))
    put(img, "<- / ->  move    s  start    x  remove start    q  quit", (W + 10, H - 12), 0.5)
    return img


# ---------------------------------------------------------------- main
def main() -> None:
    load_dotenv(repo_root() / ".env", override=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=sorted(LIST_FILE))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = env_path("IRWM_DATA_ROOT") / "aligned" / "align"
    out = a.out or env_path("IRWM_OUTPUT_ROOT") / "annotations" / f"aligned_{a.split}_segments.json"
    ids = read_ids(root, a.split)
    store = Store(out, a.split, ids)
    store.save()  # fail early if the location is not writable
    print(f"{len(ids)} frames, {len(store.segments)} segments loaded, saving to {out}")

    k = ids.index(store.cursor_id)
    pending = False
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    while True:
        cv2.imshow(WIN, render(root, a.split, ids, k, store, pending))
        key = cv2.waitKeyEx(50)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == -1:
            continue
        low = key & 0xFF if key < 256 else key
        if pending:
            if low in (ord("d"), ord("n")):
                store.segments[ids[k]] = "day" if low == ord("d") else "night"
                store.save()
                pending = False
            elif low in (ESC, ord("s")):
                pending = False
            continue  # nothing else is accepted while waiting for d / n
        if key in RIGHT and k < len(ids) - 1:
            k += 1
        elif key in LEFT and k > 0:
            k -= 1
        elif low == ord("s"):
            pending = True
            continue
        elif low == ord("x") and ids[k] in store.segments:
            del store.segments[ids[k]]
            store.save()
            continue
        elif low in (ord("q"), ESC):
            break
        else:
            continue
        store.cursor_id = ids[k]
        store.save()
    cv2.destroyAllWindows()
    print(f"saved {len(store.segments)} segments to {out}")


if __name__ == "__main__":
    main()
