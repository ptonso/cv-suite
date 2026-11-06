"""health: is anything broken. Table `health.csv`, one row per check."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from cvsuite.eda.core.sections import Context

EXAMPLES = 10
IOU_DUPLICATE = 0.99


def needs(ctx: Context) -> str | None:
    return None if ctx.dataset.records else "no records"


def _rect(box, width: int, height: int) -> tuple[float, float, float, float]:
    return (box.cx - box.w / 2) * width, (box.cy - box.h / 2) * height, (box.cx + box.w / 2) * width, (box.cy + box.h / 2) * height


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _unreadable(ctx: Context) -> list[str]:
    return [path for path, _ in ctx.table.unreadable]


def _outside(ctx: Context) -> list[str]:
    hits = []
    for record, row in zip(ctx.dataset.records, ctx.table.rows):
        if row["width"] == "":
            continue
        width, height = row["width"], row["height"]
        points = []
        for box in record.boxes:
            x1, y1, x2, y2 = _rect(box, width, height)
            points += [(x1, y1), (x2, y2)]
        for poly in record.polys:
            points += [(x * width, y * height) for x, y in poly.points]
        if any(x < -1 or y < -1 or x > width + 1 or y > height + 1 for x, y in points):
            hits.append(row["path"])
    return hits


def _zero_area(ctx: Context) -> list[str]:
    return [
        row["path"]
        for record, row in zip(ctx.dataset.records, ctx.table.rows)
        if any(b.w <= 0 or b.h <= 0 for b in record.boxes)
    ]


def _duplicate_boxes(ctx: Context) -> list[str]:
    hits = []
    for record, row in zip(ctx.dataset.records, ctx.table.rows):
        if row["width"] == "":
            continue
        boxes = [(b.cls, _rect(b, row["width"], row["height"])) for b in record.boxes if b.kind != "seg"]
        if any(boxes[i][0] == boxes[j][0] and _iou(boxes[i][1], boxes[j][1]) >= IOU_DUPLICATE for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
            hits.append(row["path"])
    return hits


def _class_out_of_range(ctx: Context) -> list[str]:
    n = len(ctx.dataset.classes)
    return [
        row["path"]
        for record, row in zip(ctx.dataset.records, ctx.table.rows)
        if any(not 0 <= item.cls < n for item in [*record.boxes, *record.polys, *record.kpts])
    ]


def _unannotated(ctx: Context) -> list[str]:
    return [row["path"] for row in ctx.table.rows if row["class"] == "(empty)"]


def _shared_names(ctx: Context) -> list[str]:
    folders: dict[str, set[str]] = defaultdict(set)
    for row in ctx.table.rows:
        folder, _, name = row["path"].rpartition("/")
        folders[name].add(folder)
    shared = {name for name, f in folders.items() if len(f) > 1}
    return [row["path"] for row in ctx.table.rows if row["path"].rpartition("/")[2] in shared]


def _duplicate_files(ctx: Context) -> list[str]:
    by_size: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(ctx.table.rows):
        if row["file_bytes"] != "":
            by_size[row["file_bytes"]].append(i)
    hits = []
    for indices in by_size.values():
        if len(indices) < 2:
            continue
        by_digest: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            from cvsuite.common.io.common import resolve_record_image_path

            digest = hashlib.sha1(resolve_record_image_path(ctx.dataset, ctx.dataset.records[i]).read_bytes()).hexdigest()
            by_digest[digest].append(i)
        for same in by_digest.values():
            if len(same) > 1:
                hits += [ctx.table.rows[i]["path"] for i in same]
    return hits


CHECKS = (
    ("unreadable_or_zero_size_images", _unreadable),
    ("boxes_outside_image", _outside),
    ("zero_area_boxes", _zero_area),
    ("duplicate_boxes", _duplicate_boxes),
    ("class_id_out_of_range", _class_out_of_range),
    ("images_without_annotations", _unannotated),
    ("file_names_in_several_folders", _shared_names),
    ("exact_duplicate_files", _duplicate_files),
)


def run_checks(ctx: Context) -> list[tuple[str, int, list[str]]]:
    return [(name, len(hits := sorted(set(check(ctx)))), hits[:EXAMPLES]) for name, check in CHECKS]


def write(ctx: Context, base: Path) -> list[str]:
    path = base / "tables" / "health.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "count", "examples"])
        for name, count, examples in run_checks(ctx):
            writer.writerow([name, count, ";".join(examples)])
    return ["tables/health.csv"]
