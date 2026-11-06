"""Plan and write background windows that touch no annotation. Contract: `specs/label/transforms.md#ops`."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from cvsuite.common.core import VisionDataset, VisionRecord
from cvsuite.common.keys import layout_path
from cvsuite.label.core.crops import resolve_image_path

TRIES = 100
Rect = tuple[float, float, float, float]


@dataclass
class Crop:
    src_image: Path
    box_xyxy: tuple[int, int, int, int]
    split: str
    name: str
    index: int = -1


def annotation_rects(record: VisionRecord) -> list[Rect]:
    """Pixel rectangles of every box, polygon (by extent) and keypoint set (by visible points) of a record."""
    width, height = record.image.width, record.image.height
    rects: list[Rect] = [
        ((b.cx - b.w / 2) * width, (b.cy - b.h / 2) * height, (b.cx + b.w / 2) * width, (b.cy + b.h / 2) * height)
        for b in record.boxes
    ]
    for poly in record.polys:
        xs, ys = [p[0] * width for p in poly.points], [p[1] * height for p in poly.points]
        rects.append((min(xs), min(ys), max(xs), max(ys)))
    for kpts in record.kpts:
        visible = [p for p in kpts.points if p[2] > 0]
        if visible:
            xs, ys = [p[0] * width for p in visible], [p[1] * height for p in visible]
            rects.append((min(xs), min(ys), max(xs), max(ys)))
    return rects


def instance_count(record: VisionRecord) -> int:
    return len(record.polys) + sum(1 for b in record.boxes if b.kind != "seg")


def _free(window: Rect, rects: Sequence[Rect]) -> bool:
    x1, y1, x2, y2 = window
    return not any(x1 < rx2 and rx1 < x2 and y1 < ry2 and ry1 < y2 for rx1, ry1, rx2, ry2 in rects)


def _place(rng: random.Random, record: VisionRecord, blocked: list[Rect], sizes: list[Rect], imgsz: int, size_from: str, region: Rect) -> tuple[int, int, int, int] | None:
    rx1, ry1, rx2, ry2 = (math.ceil(region[0]), math.ceil(region[1]), math.floor(region[2]), math.floor(region[3]))
    for _ in range(TRIES):
        side = imgsz
        if size_from == "boxes" and sizes:
            bx1, by1, bx2, by2 = rng.choice(sizes)
            side = math.ceil(max(bx2 - bx1, by2 - by1))
        side = max(1, min(side, rx2 - rx1, ry2 - ry1))
        if rx2 - rx1 < side or ry2 - ry1 < side:
            return None
        x1, y1 = rng.randint(rx1, rx2 - side), rng.randint(ry1, ry2 - side)
        if _free((x1, y1, x1 + side, y1 + side), blocked):
            return x1, y1, x1 + side, y1 + side
    return None


def plan_crops(
    ds: VisionDataset,
    *,
    imgsz: int,
    per_image: int | None,
    ratio: float | None,
    size_from: str,
    seed: int,
    splits: set[str] | None = None,
    within: str = "image",
    avoid_placed: bool = False,
) -> tuple[list[Crop], list[tuple[str, int, int]]]:
    """Return the planned windows and the `(layout path, wanted, placed)` of every image that fell short.

    `within="annotations"` keeps windows inside the extent of the annotations (none when there are none). With `avoid_placed` a window also avoids the windows placed before it on its image.
    """
    crops: list[Crop] = []
    shortfalls: list[tuple[str, int, int]] = []
    used: set[tuple[str, str]] = set()
    for index, record in enumerate(ds.records):
        if record.image is None or (splits and record.split not in splits):
            continue
        layout = layout_path(record, ds)
        wanted = per_image if per_image is not None else round(ratio * instance_count(record))
        rng = random.Random(f"{seed}:{layout}")
        rects = annotation_rects(record)
        blocked = list(rects)
        if within == "annotations":
            region = (min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)) if rects else (0, 0, 0, 0)
        else:
            region = (0, 0, record.image.width, record.image.height)
        stem = layout.rsplit(".", 1)[0] if "." in layout.rsplit("/", 1)[-1] else layout
        base = stem.replace("/", "__")
        placed = 0
        for _ in range(wanted):
            window = _place(rng, record, blocked, rects, imgsz, size_from, region)
            if window is None:
                continue
            name = f"{base}_{placed}.jpg"
            counter = 0
            while (record.split, name) in used:
                counter += 1
                name = f"{base}_{placed}_{counter}.jpg"
            used.add((record.split, name))
            if avoid_placed:
                blocked.append(tuple(float(v) for v in window))
            crops.append(Crop(resolve_image_path(record.image.path, ds.root), window, record.split or "train", name, index))
            placed += 1
        if placed < wanted:
            shortfalls.append((layout, wanted, placed))
    return crops, shortfalls


def write_crops(crops: list[Crop], out_root: Path, preview_sheet: Path | None) -> list[Path]:
    sheet: list[np.ndarray] = []
    written: list[Path] = []
    cache: tuple[Path, np.ndarray] | None = None
    for crop in crops:
        if cache is None or cache[0] != crop.src_image:
            image = cv2.imread(str(crop.src_image))
            if image is None:
                raise FileNotFoundError(f"Cannot read image for negatives: {crop.src_image}")
            cache = (crop.src_image, image)
        x1, y1, x2, y2 = crop.box_xyxy
        dst = out_root / "negatives" / crop.split / crop.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        patch = cache[1][y1:y2, x1:x2]
        if not cv2.imwrite(str(dst), patch):
            raise RuntimeError(f"Failed to write negative: {dst}")
        written.append(dst)
        sheet.append(patch)
    if preview_sheet and sheet:
        grid = np.concatenate([cv2.resize(x, (256, 256)) for x in sheet[:24]], axis=1)
        preview_sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_sheet), grid)
    return written
