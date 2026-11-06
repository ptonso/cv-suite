"""Assign records to train, val and test, or to cross-validation folds, keeping groups together."""

from __future__ import annotations

import argparse
import bisect
import math
from dataclasses import replace

import numpy as np

from cvsuite.common import splitting
from cvsuite.common.core import VisionDataset, VisionRecord
from cvsuite.common.io.common import CLASS_DIR_MULTI_LABELS_ATTR
from cvsuite.common.io.folds import FOLD_SPLITS_ATTR, FOLDS_META
from cvsuite.common.keys import group_records, parse_key

EMPTY = "(empty)"
MULTI_LABEL_ATTRS = (CLASS_DIR_MULTI_LABELS_ATTR, "classify_multi_class_labels")


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--group-by", action="append", default=[], metavar="KEY", help="Keep records with the same key together: regex:PATTERN, dir:N, attr:NAME, file:PATH.csv or similar:BITS. Repeat to join groups.")
    parser.add_argument("--stratify-by", default="class", metavar="{class,class-size:PX[,PX...],none,KEY}", help="What each split should share in proportion: class counts (default), class counts by box size (class-size:20,64), nothing, or the values of a key.")
    parser.add_argument("--val-frac", type=float, default=0.0, help="Validation share (of each fold's non-test groups with --folds).")
    parser.add_argument("--test-frac", type=float, default=0.0, help="Test share. Not valid with --folds.")
    parser.add_argument("--folds", type=int, default=None, help="Write K cross-validation folds whose test sets partition the records.")
    parser.add_argument("--only-fold", default=None, metavar="I[,I...]", help="Keep only these folds; the partition stays the K-fold one.")
    parser.add_argument("--keep", default=None, metavar="SPLIT[,SPLIT...]", help="Pin the records now in these splits; the fractions apply to the other records.")
    parser.add_argument("--on-siblings", choices=["join", "drop"], default=None, help="With --keep: records that share a group with a kept record either join the kept split (join, default) or are removed (drop).")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Largest accepted gap between a split's target and achieved share of a class (default 0.05).")
    parser.add_argument("--restarts", type=int, default=200, help="Random restarts of the packing (default 200).")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the packing.")


def _names(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _validate(args: argparse.Namespace) -> tuple[list[int], set[str]]:
    for flag in ("val_frac", "test_frac"):
        if not 0.0 <= getattr(args, flag) < 1.0:
            raise SystemExit(f"--{flag.replace('_', '-')} must be in [0, 1).")
    if args.val_frac + args.test_frac >= 1.0:
        raise SystemExit("--val-frac plus --test-frac must be below 1.")
    if args.tolerance < 0 or args.restarts < 1:
        raise SystemExit("--tolerance must be non-negative and --restarts at least 1.")
    keep = set(_names(args.keep))
    if getattr(args, "on_siblings", None) and not keep:
        raise SystemExit("--on-siblings needs --keep.")
    if keep - set(splitting.SPLITS):
        raise SystemExit(f"--keep takes splits among {', '.join(splitting.SPLITS)}, got {args.keep!r}.")
    if args.folds is None:
        if args.only_fold is not None:
            raise SystemExit("--only-fold needs --folds.")
        if args.val_frac <= 0 and args.test_frac <= 0:
            raise SystemExit("split needs a positive --val-frac or --test-frac, or --folds.")
        for name in ("val", "test"):
            if name in keep and getattr(args, f"{name}_frac") > 0:
                raise SystemExit(f"--keep {name} pins that split, so --{name}-frac cannot ask for it too.")
        return [], keep
    if args.folds < 2:
        raise SystemExit("--folds needs at least 2.")
    if args.test_frac > 0:
        raise SystemExit("--test-frac cannot be used with --folds: each fold tests on 1/K of the groups.")
    if keep:
        raise SystemExit("--keep cannot be used with --folds.")
    present = [int(i) for i in _names(args.only_fold)] if args.only_fold else list(range(args.folds))
    if any(not 0 <= i < args.folds for i in present) or len(set(present)) != len(present):
        raise SystemExit(f"--only-fold takes distinct indices in [0, {args.folds}).")
    return sorted(present), keep


def _label_name(item, classes: list[str]) -> str:
    if item.label:
        return str(item.label)
    return classes[item.cls] if 0 <= item.cls < len(classes) else str(item.cls)


def record_labels(record: VisionRecord, classes: list[str]) -> list[str]:
    """One entry per annotation (a polygon and the box derived from it count once), else the classification labels."""
    annotated = [_label_name(p, classes) for p in record.polys] + [_label_name(b, classes) for b in record.boxes if b.kind != "seg"]
    if annotated:
        return annotated
    labels: list[str] = []
    if record.classification is not None and record.classification.label:
        labels.append(str(record.classification.label))
    for attr in MULTI_LABEL_ATTRS:
        labels.extend(str(x) for x in record.attributes.get(attr) or [] if str(x) not in labels)
    return labels


def _size_buckets(spec: str) -> list[float]:
    try:
        edges = [float(x) for x in spec.split(",")]
    except ValueError as exc:
        raise SystemExit(f"--stratify-by class-size: needs numbers, got `{spec}`.") from exc
    if not edges or edges[0] <= 0 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise SystemExit("--stratify-by class-size: needs positive, strictly increasing thresholds.")
    return edges


def _bucket_names(edges: list[float]) -> list[str]:
    fmt = lambda v: f"{v:g}"  # noqa: E731
    return [f"<{fmt(edges[0])}", *[f"{fmt(a)}-{fmt(b)}" for a, b in zip(edges, edges[1:])], f">={fmt(edges[-1])}"]


def _annotation_sizes(record: VisionRecord, classes: list[str]) -> list[tuple[str, float]]:
    """`(class, sqrt(w * h) in source pixels)` for the annotations `record_labels` counts."""
    width, height = record.image.width, record.image.height
    out = [(_label_name(b, classes), math.sqrt(b.w * width * b.h * height)) for b in record.boxes if b.kind != "seg"]
    for poly in record.polys:
        xs, ys = [p[0] for p in poly.points], [p[1] for p in poly.points]
        out.append((_label_name(poly, classes), math.sqrt((max(xs) - min(xs)) * width * (max(ys) - min(ys)) * height)))
    return out


def _class_size_weights(dataset: VisionDataset, spec: str) -> tuple[np.ndarray, tuple[str, ...]]:
    edges = _size_buckets(spec)
    if not any(r.boxes or r.polys for r in dataset.records):
        raise SystemExit("class-size needs boxes or polygons.")
    buckets = _bucket_names(edges)
    sized = [_annotation_sizes(r, dataset.classes) for r in dataset.records]
    present = {name for items in sized for name, _ in items}
    classes = [c for c in dataset.classes if c in present] + sorted(present - set(dataset.classes))
    columns = [f"{c}|{b}" for c in classes for b in buckets]
    if any(not items for items in sized):
        columns.append(EMPTY)
    index = {name: i for i, name in enumerate(columns)}
    out = np.zeros((len(sized), len(columns)))
    for row, items in enumerate(sized):
        for name, size in items:
            out[row, index[f"{name}|{buckets[bisect.bisect_right(edges, size)]}"]] += 1
        if not items:
            out[row, index[EMPTY]] = 1
    return out, tuple(columns)


def weights(dataset: VisionDataset, stratify_by: str) -> tuple[np.ndarray, tuple[str, ...]]:
    n = len(dataset.records)
    if stratify_by == "none":
        return np.ones((n, 1)), ("(images)",)
    if stratify_by.startswith("class-size:"):
        return _class_size_weights(dataset, stratify_by.partition(":")[2])
    if stratify_by == "class":
        per_record = [record_labels(r, dataset.classes) for r in dataset.records]
        present = {name for names in per_record for name in names}
        columns = [c for c in dataset.classes if c in present] + sorted(present - set(dataset.classes))
        if any(not names for names in per_record):
            columns.append(EMPTY)
        index = {name: i for i, name in enumerate(columns)}
        out = np.zeros((n, len(columns)))
        for row, names in enumerate(per_record):
            for name in names:
                out[row, index[name]] += 1
            if not names:
                out[row, index[EMPTY]] = 1
        return out, tuple(columns)
    values = parse_key(stratify_by).values(dataset)
    columns = sorted(set(values))
    index = {name: i for i, name in enumerate(columns)}
    out = np.zeros((n, len(columns)))
    for row, value in enumerate(values):
        out[row, index[value]] = 1
    return out, tuple(columns)


def _pinned(dataset: VisionDataset, ids: list[str], group_of: np.ndarray, keep: set[str]) -> dict[int, int]:
    pinned: dict[int, int] = {}
    for record, group in zip(dataset.records, group_of):
        if record.split not in keep:
            continue
        split = splitting.SPLITS.index(record.split)
        if pinned.setdefault(int(group), split) != split:
            raise SystemExit(f"Group `{ids[int(group)]}` holds records pinned to different splits ({splitting.SPLITS[pinned[int(group)]]} and {record.split}).")
    return pinned


def run(dataset: VisionDataset, args: argparse.Namespace) -> VisionDataset:
    present, keep = _validate(args)
    if not dataset.records:
        raise SystemExit("split needs at least one record, but the dataset is empty at this point of the chain.")
    dropped = 0
    if getattr(args, "on_siblings", None) == "drop":
        before = group_records(dataset, list(args.group_by))
        kept_groups = {g for g, r in zip(before, dataset.records) if r.split in keep}
        survivors = [r for g, r in zip(before, dataset.records) if r.split in keep or g not in kept_groups]
        dropped, dataset = len(dataset.records) - len(survivors), replace(dataset, records=survivors)
        print(f"--on-siblings drop removed {dropped} record(s) that share a group with a kept record.")
    ids = group_records(dataset, list(args.group_by))
    matrix, columns = weights(dataset, args.stratify_by)
    group_ids, group_of, counts = splitting.group_counts(ids, matrix)
    if args.folds is not None and len(group_ids) < args.folds:
        raise SystemExit(f"--folds {args.folds} needs at least {args.folds} groups, found {len(group_ids)}.")

    if args.folds is None:
        pinned = _pinned(dataset, group_ids, group_of, keep)
        shares = [1.0 - args.val_frac - args.test_frac, args.val_frac, args.test_frac]
        packing = splitting.pack(counts, shares, pinned=pinned, seed=args.seed, restarts=args.restarts)
        reports = [splitting.make_report(packing, splitting.SPLITS, columns)]
        per_record = [[splitting.SPLITS[packing.assignment[g]]] for g in group_of]
        info = {"mode": "ratio", **reports[0].info()}
    else:
        folds, packings = splitting.pack_folds(counts, args.folds, args.val_frac, seed=args.seed, restarts=args.restarts)
        reports = [splitting.make_report(packings[i], splitting.SPLITS, columns, label=f"fold {i}") for i in present]
        per_record = [[splitting.SPLITS[folds[i][g]] for i in range(args.folds)] for g in group_of]
        info = {"mode": "folds", "k": args.folds, "present": present, "folds": [r.info() for r in reports]}

    for report in reports:
        print(splitting.format_report(report))
    splitting.check_tolerance(reports, args.tolerance)

    records = []
    for record, splits in zip(dataset.records, per_record):
        if args.folds is None:
            records.append(replace(record, split=splits[0]))
        else:
            records.append(replace(record, split=splits[present[0]], attributes={**record.attributes, FOLD_SPLITS_ATTR: list(splits)}))
    meta = {**dataset.meta, "step_info": {**info, **({"siblings_dropped": dropped} if getattr(args, "on_siblings", None) == "drop" else {})}}
    if args.folds is not None:
        meta[FOLDS_META] = {"k": args.folds, "present": present}
    return replace(dataset, records=records, meta=meta)
