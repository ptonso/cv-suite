"""Compare two dataset folders record by record. Dev tool of the project migrations (`specs/README.md`, delta 6).

    python scripts/compare_datasets.py A B [--pixel-tol N] [--ignore-splits] [--key layout|name|content]

Records are matched by layout path, by file name, or by image bytes (`content`, for outputs that rename colliding files). A file read twice (a yolo `val` that points at the train folder) counts once.
Class lists are compared as sets. Splits, boxes, polygons, keypoints and classification labels are compared within 1e-3, images by size and mean absolute pixel difference. Exit status 1 on any difference.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from cvsuite.common.core import VisionDataset, VisionRecord
from cvsuite.common.io import read_dataset
from cvsuite.common.io.common import resolve_record_image_path
from cvsuite.common.keys import layout_path

TOL = 1e-3
SHOWN = 20


def _name(dataset: VisionDataset, item) -> str:
    return dataset.classes[item.cls] if 0 <= item.cls < len(dataset.classes) else str(item.label or item.cls)


def _geometry(dataset: VisionDataset, record: VisionRecord) -> dict[str, dict[str, list[tuple[float, ...]]]]:
    """Per annotation kind and class name, the sorted coordinate tuples."""
    out: dict[str, dict[str, list[tuple[float, ...]]]] = {"box": defaultdict(list), "poly": defaultdict(list), "kpt": defaultdict(list)}
    for box in record.boxes:
        out["box"][_name(dataset, box)].append((box.cx, box.cy, box.w, box.h))
    for poly in record.polys:
        out["poly"][_name(dataset, poly)].append(tuple(v for point in poly.points for v in point))
    for kpt in record.kpts:
        out["kpt"][_name(dataset, kpt)].append(tuple(v for point in kpt.points for v in point))
    for groups in out.values():
        for values in groups.values():
            values.sort(key=lambda item: tuple(round(v, 3) for v in item))
    return out


def _matched(left: list[tuple[float, ...]], right: list[tuple[float, ...]]) -> bool:
    """Every item of `left` pairs with its own nearest item of `right` within the tolerance."""
    if len(left) != len(right):
        return False
    if not left:
        return True
    if len({len(item) for item in left + right}) != 1:
        return False
    free = np.ones(len(right), dtype=bool)
    target = np.asarray(right, dtype=float)
    for item in left:
        distance = np.where(free, np.abs(target - np.asarray(item)).max(axis=1), np.inf)
        best = int(distance.argmin())
        if distance[best] > TOL:
            return False
        free[best] = False
    return True


def _image_diff(a: VisionDataset, ra: VisionRecord, b: VisionDataset, rb: VisionRecord) -> tuple[bool, float]:
    with Image.open(resolve_record_image_path(a, ra)) as left, Image.open(resolve_record_image_path(b, rb)) as right:
        if left.size != right.size:
            return False, float("nan")
        x, y = np.asarray(left.convert("RGB"), dtype=np.int16), np.asarray(right.convert("RGB"), dtype=np.int16)
        return True, float(np.abs(x - y).mean())


def compare(a: VisionDataset, b: VisionDataset, *, pixel_tol: float = 1.0, ignore_splits: bool = False, key: str = "layout") -> list[str]:
    """The list of differences between two datasets; empty when they match."""
    diffs: list[str] = []
    if sorted(a.classes) != sorted(b.classes):
        diffs.append(f"classes: {sorted(set(a.classes) ^ set(b.classes))} are in one list only")

    def index(dataset: VisionDataset) -> dict[str, VisionRecord]:
        """Layout path to record. Records sharing a path (several sources, or one name in two splits) are told apart by their annotations."""
        groups: dict[str, list[VisionRecord]] = defaultdict(list)
        seen: set[Path] = set()
        for r in dataset.records:
            if (file := resolve_record_image_path(dataset, r)) in seen:
                continue
            seen.add(file)
            path = layout_path(r, dataset)
            name = {"layout": path, "name": Path(path).name}.get(key) or hashlib.sha1(resolve_record_image_path(dataset, r).read_bytes()).hexdigest()
            groups[name].append(r)
        out = {}
        for name, records in groups.items():
            records.sort(key=lambda r: repr(sorted((kind, n, [tuple(round(v, 3) for v in item) for item in vals]) for kind, by_name in _geometry(dataset, r).items() for n, vals in by_name.items())))
            out.update({name if len(records) == 1 else f"{name}#{i}": r for i, r in enumerate(records)})
        return out

    left, right = index(a), index(b)
    diffs += [f"only in A: {k}" for k in sorted(left.keys() - right.keys())]
    diffs += [f"only in B: {k}" for k in sorted(right.keys() - left.keys())]
    for k in sorted(left.keys() & right.keys()):
        ra, rb = left[k], right[k]
        if not ignore_splits and (ra.split or "train") != (rb.split or "train"):
            diffs.append(f"{k}: split {ra.split} != {rb.split}")
        la, lb = (ra.classification.label if ra.classification else None), (rb.classification.label if rb.classification else None)
        if la != lb:
            diffs.append(f"{k}: classification {la} != {lb}")
        ga, gb = _geometry(a, ra), _geometry(b, rb)
        for kind in ga:
            for name in sorted(ga[kind].keys() | gb[kind].keys()):
                va, vb = ga[kind][name], gb[kind][name]
                if not _matched(va, vb):
                    diffs.append(f"{k}: {kind} {name}: {len(va)} in A, {len(vb)} in B or coordinates differ")
        same_size, error = _image_diff(a, ra, b, rb)
        if not same_size:
            diffs.append(f"{k}: image size differs")
        elif error > pixel_tol:
            diffs.append(f"{k}: image mean abs diff {error:.2f} > {pixel_tol}")
    return diffs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--pixel-tol", type=float, default=1.0)
    parser.add_argument("--ignore-splits", action="store_true")
    parser.add_argument("--key", choices=("layout", "name", "content"), default="layout")
    parser.add_argument("--from", dest="fmt", default="auto", help="reader format for both folders")
    parser.add_argument("--task", default="auto", help="task to read (a yolo folder with labels and labels_seg reads as inst-seg unless told det)")
    parser.add_argument("--splits", help="read only these splits, comma-separated (a yolo data.yaml whose val aliases train reads those images twice)")
    args = parser.parse_args(argv)
    splits = args.splits.split(",") if args.splits else None
    a, b = (read_dataset(p, format=args.fmt, task=args.task, splits=splits) for p in (args.a, args.b))
    diffs = compare(a, b, pixel_tol=args.pixel_tol, ignore_splits=args.ignore_splits, key=args.key)
    print(f"A: {len(a.records)} records, B: {len(b.records)} records, {len(diffs)} difference(s)")
    for line in diffs[:SHOWN]:
        print(f"  {line}")
    if len(diffs) > SHOWN:
        print(f"  ... {len(diffs) - SHOWN} more")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
