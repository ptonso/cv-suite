"""profile: what the data looks like. Figures `01_contact_sheet.jpg`, `02_class_examples.jpg`, `03_extremes.jpg`."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from cvsuite.common.io.common import resolve_record_image_path
from cvsuite.common.sampling import take
from cvsuite.common.transforms.split import record_labels
from cvsuite.eda.core.sections import Context, sampling_keys as record_keys
from cvsuite.eda.core.style import labeled_rows, tile, write_image
from cvsuite.label.core.annotate import detection_crops, render_overlay

CONTACT = 24
PER_CLASS = 6
EXTREMES = 6
CELL = 96


def _plain(ctx: Context, index: int, size: int) -> np.ndarray | None:
    image = cv2.imread(str(resolve_record_image_path(ctx.dataset, ctx.dataset.records[index])))
    return None if image is None else cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _overlay(ctx: Context, index: int, size: int) -> np.ndarray | None:
    record = ctx.dataset.records[index]
    if not (record.boxes or record.polys or record.kpts):
        return _plain(ctx, index, size)
    return render_overlay(record, ctx.dataset.classes, imgsz=size, root=ctx.dataset.root)


def _readable(ctx: Context) -> list[int]:
    return ctx.table.readable()


def needs(ctx: Context) -> str | None:
    return None if _readable(ctx) else "no readable images"


def _contact_sheet(ctx: Context, base: Path) -> str | None:
    annotated = [i for i in _readable(ctx) if ctx.dataset.records[i].boxes or ctx.dataset.records[i].polys or ctx.dataset.records[i].kpts]
    pool = annotated or _readable(ctx)
    keys = record_keys(ctx.dataset)
    chosen = [pool[j] for j in take([keys[i] for i in pool], ctx.seed, CONTACT)]
    images, labels = [], []
    for i in chosen:
        image = _overlay(ctx, i, 256)
        if image is None:
            ctx.omit(ctx.table.rows[i]["path"], "contact sheet: image could not be drawn")
            continue
        images.append(image)
        labels.append(ctx.table.rows[i]["path"].rsplit("/", 1)[-1])
    if not images:
        ctx.omit("figures/01_contact_sheet.jpg", "no drawable image")
        return None
    ctx.samples.append({"figure": "01_contact_sheet", "seed": ctx.seed, "records": len(images)})
    write_image(base / "figures" / "01_contact_sheet.jpg", tile(images, 6, 256, labels))
    return "figures/01_contact_sheet.jpg"


def _class_records(ctx: Context) -> dict[str, list[int]]:
    by_class: dict[str, list[int]] = defaultdict(list)
    for i in _readable(ctx):
        for name in sorted(set(record_labels(ctx.dataset.records[i], ctx.dataset.classes))):
            by_class[name].append(i)
    return by_class


def _crop_of(ctx: Context, index: int, name: str) -> np.ndarray | None:
    record = ctx.dataset.records[index]
    if not record.boxes:
        return _plain(ctx, index, CELL)
    for label, crop in detection_crops(record, ctx.dataset.classes, root=ctx.dataset.root):
        if label == name:
            return cv2.resize(crop, (CELL, CELL), interpolation=cv2.INTER_AREA)
    return None


def _class_examples(ctx: Context, base: Path) -> str | None:
    by_class = _class_records(ctx)
    if not by_class:
        ctx.omit("figures/02_class_examples.jpg", "no labeled records")
        return None
    ordered = sorted(by_class, key=lambda c: (-len(by_class[c]), c))
    names = list(dict.fromkeys([*ordered[: ctx.top], *ordered[::-1][: ctx.top]]))
    keys = record_keys(ctx.dataset)
    rows = []
    for name in names:
        pool = by_class[name]
        picks = [pool[j] for j in take([keys[i] for i in pool], ctx.seed, min(len(pool), PER_CLASS * 2))]
        crops = [c for c in (_crop_of(ctx, i, name) for i in picks) if c is not None][:PER_CLASS]
        rows.append((f"{name} ({len(by_class[name])})", crops))
    write_image(base / "figures" / "02_class_examples.jpg", labeled_rows(rows, PER_CLASS, CELL))
    ctx.samples.append({"figure": "02_class_examples", "seed": ctx.seed, "classes": len(rows)})
    return "figures/02_class_examples.jpg"


def _box_size(ctx: Context, index: int) -> float | None:
    record = ctx.dataset.records[index]
    width, height = ctx.table.rows[index]["width"], ctx.table.rows[index]["height"]
    sizes = [math.sqrt(b.w * width * b.h * height) for b in record.boxes if b.kind != "seg"]
    for poly in record.polys:
        xs, ys = [p[0] for p in poly.points], [p[1] for p in poly.points]
        sizes.append(math.sqrt((max(xs) - min(xs)) * width * (max(ys) - min(ys)) * height))
    return max(sizes) if sizes else None


def _extremes(ctx: Context, base: Path) -> str | None:
    if not ctx.has_geometry():
        ctx.omit("figures/03_extremes.jpg", "the dataset has no boxes or polygons")
        return None
    readable = _readable(ctx)
    sized = sorted(((s, i) for i in readable if (s := _box_size(ctx, i)) is not None), key=lambda t: (t[0], ctx.table.rows[t[1]]["path"]))
    bare = [i for i in readable if _box_size(ctx, i) is None][:EXTREMES]
    rows = []
    for label, indices in (("largest boxes", [i for _, i in sized[::-1][:EXTREMES]]), ("smallest boxes", [i for _, i in sized[:EXTREMES]]), ("no annotations", bare)):
        cells = [c for c in (_overlay(ctx, i, CELL * 2) for i in indices) if c is not None]
        rows.append((label, cells))
    write_image(base / "figures" / "03_extremes.jpg", labeled_rows(rows, EXTREMES, CELL * 2))
    return "figures/03_extremes.jpg"


def write(ctx: Context, base: Path) -> list[str]:
    written = [f for f in (_contact_sheet(ctx, base), _class_examples(ctx, base), _extremes(ctx, base)) if f]
    return written
