"""Sample a class-organized dataset while preserving or balancing class distribution."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

from ...core import io


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=("preserve", "balance"),
        default="preserve",
        help="How to interpret the per-class sampling limit.",
    )
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument(
        "--max-n-per-class",
        type=int,
        default=None,
        help="Per-class reference cap for sampling.",
    )
    size_group.add_argument(
        "--max-frac-per-class",
        type=float,
        default=None,
        help="Per-class fraction cap for sampling. Must satisfy 0 < F <= 1.",
    )
    parser.add_argument(
        "--with-replacement",
        action="store_true",
        help="Allow duplicate sampling within a class when computed targets exceed available images.",
    )
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Request hard-linked image export when the output command supports it.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic sampling.")


def _record_label(record) -> str | None:
    info = getattr(record, "classification", None)
    label = None if info is None else info.label
    text = str(label or "").strip()
    return text or None


def _ordered_labels(dataset, present_labels: set[str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for raw in getattr(dataset, "classes", []) or []:
        text = str(raw).strip()
        if text and text in present_labels and text not in seen:
            labels.append(text)
            seen.add(text)
    for record in dataset.records:
        label = _record_label(record)
        if label and label in present_labels and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def _has_annotations(dataset) -> bool:
    for record in dataset.records:
        if record.boxes or record.polys or record.kpts or record.labelme_shapes:
            return True
    return False


def _validate_inputs(dataset, args: argparse.Namespace) -> None:
    source_path = getattr(args, "source_path", None)
    if isinstance(source_path, Path) and io.looks_like_annotated_dataset(source_path):
        raise SystemExit("For labeled datasets, use `cvsuite label sample` instead.")
    if _has_annotations(dataset):
        raise SystemExit("For labeled datasets, use `cvsuite label sample` instead.")
    if io.is_multi_class_ingest(dataset):
        raise SystemExit(
            "`cvsuite class sample` requires a single-label class dataset. "
            "Datasets ingested with `--from multi-class` are not supported."
        )

    labels = [_record_label(record) for record in dataset.records]
    if not dataset.records or any(label is None for label in labels):
        raise SystemExit("`cvsuite class sample` requires a fully labeled class dataset. For flat folders, use `cvsuite prep sample` instead.")

    max_n_per_class = getattr(args, "max_n_per_class", None)
    max_frac_per_class = getattr(args, "max_frac_per_class", None)
    if max_n_per_class is not None and int(max_n_per_class) <= 0:
        raise SystemExit("--max-n-per-class must be greater than 0.")
    if max_frac_per_class is not None and (float(max_frac_per_class) <= 0.0 or float(max_frac_per_class) > 1.0):
        raise SystemExit("--max-frac-per-class must be between 0 and 1 (exclusive of 0).")


def _target_count(
    *,
    mode: str,
    class_size: int,
    min_class_size: int,
    max_class_size: int,
    max_n_per_class: int | None,
    max_frac_per_class: float | None,
) -> int:
    if max_n_per_class is not None:
        if mode == "balance":
            return int(max_n_per_class)
        return max(1, int(math.floor(class_size * int(max_n_per_class) / max_class_size)))

    assert max_frac_per_class is not None
    frac = float(max_frac_per_class)
    if mode == "balance":
        return max(1, int(math.floor(min_class_size * frac)))
    return max(1, int(math.floor(class_size * frac)))


def _compute_targets(labels: list[str], by_label: Dict[str, List[object]], args: argparse.Namespace) -> dict[str, int]:
    sizes = {label: len(by_label[label]) for label in labels}
    min_class_size = min(sizes.values())
    max_class_size = max(sizes.values())
    targets: dict[str, int] = {}
    for label in labels:
        targets[label] = _target_count(
            mode=str(getattr(args, "mode", "preserve")),
            class_size=sizes[label],
            min_class_size=min_class_size,
            max_class_size=max_class_size,
            max_n_per_class=getattr(args, "max_n_per_class", None),
            max_frac_per_class=getattr(args, "max_frac_per_class", None),
        )
    return targets


def _sample_bucket(bucket: List[object], *, target: int, with_replacement: bool, rng: random.Random, label: str) -> list[object]:
    size = len(bucket)
    if not with_replacement and target > size:
        raise SystemExit(
            f"Class '{label}' needs {target} samples but only {size} are available. "
            "Use `--with-replacement` to allow duplicate sampling."
        )
    if with_replacement:
        return [bucket[rng.randrange(size)] for _ in range(target)]
    shuffled = list(bucket)
    rng.shuffle(shuffled)
    return shuffled[:target]


def run(dataset, args: argparse.Namespace):
    _validate_inputs(dataset, args)

    by_label: Dict[str, List[object]] = {}
    for record in dataset.records:
        label = _record_label(record)
        assert label is not None
        by_label.setdefault(label, []).append(record)

    rng = random.Random(int(getattr(args, "seed", 0)))
    labels = _ordered_labels(dataset, set(by_label))
    targets = _compute_targets(labels, by_label, args)

    selected: list[object] = []
    for label in labels:
        selected.extend(
            _sample_bucket(
                by_label[label],
                target=targets[label],
                with_replacement=bool(getattr(args, "with_replacement", False)),
                rng=rng,
                label=label,
            )
        )
    rng.shuffle(selected)

    meta = dict(dataset.meta)
    knob_name = "max_n_per_class" if getattr(args, "max_n_per_class", None) is not None else "max_frac_per_class"
    knob_value = getattr(args, knob_name)
    meta["sample"] = {
        "mode": str(getattr(args, "mode", "preserve")),
        "seed": int(getattr(args, "seed", 0)),
        "with_replacement": bool(getattr(args, "with_replacement", False)),
        "hardlink": bool(getattr(args, "hardlink", False)),
        "size_knob": knob_name,
        "size_value": knob_value,
        "targets": dict(targets),
        "class_sizes": {label: len(by_label[label]) for label in labels},
        "sample_size": len(selected),
    }
    return replace(
        dataset,
        records=list(selected),
        classes=_ordered_labels(dataset, set(targets)),
        meta=meta,
    )
