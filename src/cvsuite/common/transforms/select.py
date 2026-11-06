"""Remove annotations, classes and records by their properties. Contract: `specs/common/select_remap.md#select`."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import replace

from cvsuite.common.core import VisionDataset, VisionRecord
from cvsuite.common.keys import layout_path
from cvsuite.common.transforms import vocab

SIZE_RULES = ("min_box_px", "min_side_px", "min_area_frac")
SHAPE_TOLERANCE = 1e-3


def attach(parser: argparse.ArgumentParser, *, pixel_flags: bool = True) -> None:
    if pixel_flags:
        parser.add_argument("--min-box-px", type=float, default=None, help="Remove annotations whose sqrt(width * height) is below N pixels.")
        parser.add_argument("--min-side-px", type=float, default=None, help="Remove annotations whose longer side is below N pixels.")
        parser.add_argument("--min-area-frac", type=float, default=None, help="Remove annotations covering less than this share of the image.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--keep-class", default=None, metavar="A,B", help="Keep only these classes.")
    group.add_argument("--drop-class", default=None, metavar="A,B", help="Remove these classes.")
    parser.add_argument("--exclude-name", action="append", default=[], metavar="REGEX", help="Remove records whose path matches (repeatable).")
    if pixel_flags:
        parser.add_argument("--empty", choices=vocab.EMPTY_CHOICES, default=None, help="Records left without annotations: drop them or keep them as images. Required when an annotation filter or class flag runs.")


def _names(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _small(w: float, h: float, width: int, height: int, flags: dict[str, float | None]) -> str | None:
    wpx, hpx = w * width, h * height
    if flags["min_box_px"] is not None and math.sqrt(wpx * hpx) < flags["min_box_px"]:
        return "min_box_px"
    if flags["min_side_px"] is not None and max(wpx, hpx) < flags["min_side_px"]:
        return "min_side_px"
    if flags["min_area_frac"] is not None and w * h < flags["min_area_frac"]:
        return "min_area_frac"
    return None


def _poly_extent(poly) -> tuple[float, float]:
    xs, ys = [p[0] for p in poly.points], [p[1] for p in poly.points]
    return max(xs) - min(xs), max(ys) - min(ys)


def _near(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    return len(a) == len(b) and all(abs(p[0] - q[0]) <= SHAPE_TOLERANCE and abs(p[1] - q[1]) <= SHAPE_TOLERANCE for p, q in zip(a, b))


def _drop_shapes(shapes: list, removed: list[tuple[str, list[tuple[float, float]]]]) -> list:
    def points(shape):
        raw = shape.get("points", []) if isinstance(shape, dict) else shape.points
        return [(float(p[0]), float(p[1])) for p in raw]

    def kind(shape):
        return shape.get("shape_type") if isinstance(shape, dict) else shape.shape_type

    return [s for s in shapes if not any(kind(s) == k and _near(points(s), pts) for k, pts in removed)]


def _filter_sizes(record: VisionRecord, flags: dict[str, float | None], counts: dict[str, int]) -> VisionRecord:
    width, height = record.image.width, record.image.height
    removed_shapes: list[tuple[str, list[tuple[float, float]]]] = []

    boxes, dropped_box = [], []
    for i, box in enumerate(record.boxes):
        rule = _small(box.w, box.h, width, height, flags)
        if rule is None:
            boxes.append(box)
            continue
        counts[rule] += 1
        dropped_box.append(i)
        x1, y1 = (box.cx - box.w / 2) * width, (box.cy - box.h / 2) * height
        removed_shapes.append(("rectangle", [(x1, y1), (x1 + box.w * width, y1 + box.h * height)]))
    polys = []
    for poly in record.polys:
        rule = _small(*_poly_extent(poly), width, height, flags)
        if rule is None:
            polys.append(poly)
            continue
        counts[rule] += 1
        removed_shapes.append(("polygon", [(x * width, y * height) for x, y in poly.points]))

    kpts = list(record.kpts)
    if record.kpts and dropped_box:
        pose = [i for i, b in enumerate(record.boxes) if b.kind == "pose"]
        if len(pose) != len(record.kpts):
            raise ValueError(f"Cannot pair keypoints with pose boxes in {layout_path(record, VisionDataset(records=[record]))}: {len(pose)} pose boxes, {len(record.kpts)} keypoint sets.")
        gone = {pose.index(i) for i in dropped_box if i in pose}
        kpts = [k for j, k in enumerate(record.kpts) if j not in gone]

    if not removed_shapes:
        return record
    round_trip = record.round_trip
    if "labelme" in round_trip:
        labelme = {**round_trip["labelme"], "shapes": _drop_shapes(list(round_trip["labelme"].get("shapes", [])), removed_shapes)}
        round_trip = {**round_trip, "labelme": labelme}
    return replace(record, boxes=boxes, polys=polys, kpts=kpts, round_trip=round_trip, labelme_shapes=_drop_shapes(list(record.labelme_shapes), removed_shapes))


def select(
    dataset: VisionDataset,
    *,
    min_box_px: float | None = None,
    min_side_px: float | None = None,
    min_area_frac: float | None = None,
    keep_class: list[str] | None = None,
    drop_class: list[str] | None = None,
    exclude_name: list[str] | None = None,
    empty: str | None = None,
) -> tuple[VisionDataset, dict]:
    flags = {"min_box_px": min_box_px, "min_side_px": min_side_px, "min_area_frac": min_area_frac}
    sizing = any(v is not None for v in flags.values())
    classing = bool(keep_class or drop_class)
    if keep_class and drop_class:
        raise ValueError("--keep-class and --drop-class are exclusive.")
    if not (sizing or classing or exclude_name):
        raise SystemExit("select needs at least one of the size flags, --keep-class, --drop-class or --exclude-name.")
    if sizing and not vocab.has_geometry(dataset):
        raise ValueError("Pixel and area flags need a dataset with boxes or polygons.")
    if classing:
        vocab.reject_masks(dataset, "The class flags of select")
    if sizing or classing:
        vocab.require_empty(dataset, empty, "select")
    for value in (min_box_px, min_side_px, min_area_frac):
        if value is not None and value <= 0:
            raise ValueError("Size thresholds must be positive.")

    info: dict = {"records_in": len(dataset.records)}
    records = dataset.records
    if exclude_name:
        patterns = [re.compile(p) for p in exclude_name]
        kept = [r for r in records if not any(p.search(layout_path(r, dataset)) for p in patterns)]
        info["records_excluded_by_name"] = len(records) - len(kept)
        records = kept
    dataset = replace(dataset, records=records)

    if sizing:
        counts = {rule: 0 for rule in SIZE_RULES}
        filtered, emptied = [], 0
        for record in dataset.records:
            new = _filter_sizes(record, flags, counts)
            if (record.boxes or record.polys or record.kpts) and not (new.boxes or new.polys or new.kpts):
                emptied += 1
                if empty == "drop":
                    continue
            filtered.append(new)
        info["annotations_removed"] = {k: v for k, v in counts.items() if flags[k] is not None}
        info["records_emptied_by_size"] = emptied
        info["records_dropped_by_size"] = emptied if empty == "drop" else 0
        dataset = replace(dataset, records=filtered)

    if classing:
        vocab.check_names(dataset, keep_class or drop_class or [], "--keep-class" if keep_class else "--drop-class")
        dropped = set(drop_class) if drop_class else {c for c in dataset.classes if c not in set(keep_class)}
        remaining = [c for c in dataset.classes if c not in dropped]
        dataset, stats = vocab.reindex_classes(dataset, {c: None for c in dropped}, remaining, empty=empty)
        info["classes_removed"] = sorted(dropped)
        info["class_annotations_removed"] = stats["annotations_removed"]
        info["records_dropped_by_class"] = stats["records_dropped"]

    if not dataset.records:
        raise SystemExit("select left no records.")
    info["records_out"] = len(dataset.records)
    return dataset, info


def run(dataset: VisionDataset, args: argparse.Namespace) -> VisionDataset:
    out, info = select(
        dataset,
        min_box_px=getattr(args, "min_box_px", None),
        min_side_px=getattr(args, "min_side_px", None),
        min_area_frac=getattr(args, "min_area_frac", None),
        keep_class=_names(args.keep_class) or None,
        drop_class=_names(args.drop_class) or None,
        exclude_name=list(args.exclude_name),
        empty=getattr(args, "empty", None),
    )
    return replace(out, meta={**out.meta, "step_info": info})
