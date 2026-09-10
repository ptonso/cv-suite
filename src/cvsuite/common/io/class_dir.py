"""Adapter for class-folder classification datasets.

This module expects directory structure to carry labels, either as one class
folder per image or duplicate-tree multi-label layouts. In memory it reads and
writes canonical records where classification data lives in
`record.classification`, while geometry and VQA fields stay empty.

Example single-label root:
    dataset/
      cat/
        a.jpg
      dog/
        b.jpg

Example multi-label root:
    dataset/
      red/
        img_001.jpg
      round/
        img_001.jpg
"""

from __future__ import annotations

import math
import os
import random
import re
from pathlib import Path

from cvsuite.common.core import Classification, VisionDataset, VisionRecord
from cvsuite.common.io import common

_SPLIT_DIR_NAMES = {"train", "val", "valid", "validation", "test"}
_CLASS_NAME_MODES = {"basename", "full_path"}


def _iter_images(root: Path, *, max_depth: int = -1) -> list[Path]:
    if root.is_file():
        return [root]
    return [path for path in common.iter_files(root, max_depth=max_depth) if path.suffix.lower() in common.IMG_EXTS]


def _split_scan_roots(root: Path, splits: list[str] | None, *, top_level_split: bool) -> list[Path]:
    """Walk only the requested split subtrees when the tree is split-prefixed."""
    if not (top_level_split and splits):
        return [root]
    requested = {common.canonical_split_id(item) for item in splits}
    roots = [
        entry
        for entry in sorted(root.iterdir())
        if entry.is_dir()
        and entry.name.lower() in _SPLIT_DIR_NAMES
        and common.canonical_split_id(entry.name) in requested
    ]
    return roots or [root]


def _split_and_class_parts(rel: Path) -> tuple[str, list[str]]:
    dirs = list(rel.parts[:-1])
    split = "train"
    if dirs:
        first = dirs[0].lower()
        if first in _SPLIT_DIR_NAMES:
            split = common.canonical_split_id(first)
            dirs = dirs[1:]
    if dirs and dirs[-1].lower() == "images":
        dirs = dirs[:-1]
    return split, dirs


def _validate_class_name_mode(mode: object) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _CLASS_NAME_MODES:
        raise ValueError("class_name_mode must be one of: 'basename', 'full_path'.")
    return normalized


def _resolve_class_name(class_parts: list[str], *, class_name_mode: str) -> str | None:
    if not class_parts:
        return None
    if class_name_mode == "basename":
        return class_parts[-1]
    if class_name_mode == "full_path":
        return Path(*class_parts).as_posix()
    raise ValueError(f"Unsupported class_name_mode: {class_name_mode!r}")


def _infer_split_and_class(root: Path, img: Path, *, class_name_mode: str = "basename") -> tuple[str, str | None]:
    rel = img.relative_to(root)
    split, class_parts = _split_and_class_parts(rel)
    return split, _resolve_class_name(class_parts, class_name_mode=class_name_mode)


def _compare_multi_class_images(key: str, paths: list[Path]) -> tuple[int, int]:
    canonical = paths[0]
    image = common.image_info_from_path(canonical)
    payload = canonical.read_bytes()
    for other in paths[1:]:
        other_image = common.image_info_from_path(other)
        if (other_image.width, other_image.height) != (image.width, image.height):
            raise ValueError(f"Duplicate filename {key!r} maps to different image sizes across classes.")
        if other.read_bytes() != payload:
            raise ValueError(f"Duplicate filename {key!r} maps to different image contents across classes.")
    return image.width, image.height


def _ingest_multi_label(root: Path, *, max_depth: int = -1) -> VisionDataset:
    grouped_paths: dict[str, list[Path]] = {}
    grouped_labels: dict[str, set[str]] = {}
    classes: list[str] = []
    seen_classes: set[str] = set()

    for img in _iter_images(root, max_depth=max_depth):
        rel = img.relative_to(root)
        if len(rel.parts) < 2:
            raise ValueError("Multi-label class-dir input expects images under class folders.")
        first = rel.parts[0].lower()
        if first in _SPLIT_DIR_NAMES:
            raise ValueError("Multi-label class-dir input does not support split-prefixed trees.")
        class_label = Path(*rel.parts[:-1]).as_posix()
        grouped_paths.setdefault(rel.name, []).append(img)
        grouped_labels.setdefault(rel.name, set()).add(class_label)
        if class_label not in seen_classes:
            seen_classes.add(class_label)
            classes.append(class_label)

    records: list[VisionRecord] = []
    for filename in sorted(grouped_paths):
        paths = sorted(grouped_paths[filename])
        width, height = _compare_multi_class_images(filename, paths)
        labels = sorted(grouped_labels[filename])
        probs = {label: 1.0 for label in labels}
        source_rel = common.root_relative_path(root, paths[0])
        rel = Path(filename)
        image = common.image_info_from_path(paths[0])
        image.path = source_rel
        image.width = width
        image.height = height
        records.append(
            VisionRecord(
                image=image,
                split="train",
                rel_image_path=rel,
                classification=Classification(label=labels[0], score=1.0, probs=probs),
                attributes={common.CLASS_DIR_MULTI_LABELS_ATTR: labels},
            )
        )

    dataset = VisionDataset(records=records, classes=classes, root=root)
    dataset.meta["class_dir_ingest_mode"] = "multi-label"
    return dataset


def _group_records_multi_label(root: Path, records: list[VisionRecord], *, class_name_mode: str) -> list[VisionRecord]:
    """Merge per-file records sharing ``(split, basename)`` into one multi-label record.

    Same-basename files placed under different class folders encode multi-label membership;
    their bytes must be identical (enforced by ``_compare_multi_class_images``). Splits are
    preserved by keying on split, so this works on split-prefixed trees that
    ``_ingest_multi_label`` rejects.
    """
    grouped: dict[tuple[str, str], list[VisionRecord]] = {}
    order: list[tuple[str, str]] = []
    for record in records:
        rel = record.rel_image_path
        if rel is None:
            continue
        key = (record.split, rel.name)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(record)

    merged: list[VisionRecord] = []
    for key in order:
        members = grouped[key]
        labels = sorted({
            label
            for member in members
            for label in _coerced_multi_labels(member, class_name_mode=class_name_mode)
        })
        if len(members) > 1:
            _compare_multi_class_images(key[1], [root / member.image.path for member in members])
        base = members[0]
        if labels:
            base.classification = Classification(label=labels[0], score=1.0, probs={label: 1.0 for label in labels})
            base.attributes[common.CLASS_DIR_MULTI_LABELS_ATTR] = labels
        merged.append(base)
    return merged


def _effective_threshold(record: VisionRecord, *, threshold: float | None = None) -> float | None:
    if threshold is not None:
        return float(threshold)
    info = record.classification
    if info is None:
        return None
    meta_threshold = info.meta.get("threshold") if isinstance(info.meta, dict) else None
    if meta_threshold is None:
        return None
    try:
        return float(meta_threshold)
    except (TypeError, ValueError):
        return None


def _threshold_for_label(label: str, *, default_threshold: float | None, class_thresholds: dict[str, float] | None) -> float | None:
    if class_thresholds and label in class_thresholds:
        return float(class_thresholds[label])
    return default_threshold


def _winner_by_score(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def _effective_class_label(
    record: VisionRecord,
    *,
    threshold: float | None = None,
    unlabeled_name: str = "unlabeled",
    class_thresholds: dict[str, float] | None = None,
) -> str:
    info = record.classification
    if info is None:
        return unlabeled_name

    scores = common.classification_scores(info)
    default_threshold = _effective_threshold(record, threshold=threshold)
    overrides = class_thresholds or {}
    if scores:
        winner = _winner_by_score(scores)
        if winner is None:
            return unlabeled_name
        any_pass = False
        for label, score in scores.items():
            label_threshold = _threshold_for_label(label, default_threshold=default_threshold, class_thresholds=overrides)
            if label_threshold is None:
                if not overrides:
                    any_pass = True
                    break
                continue
            if score >= label_threshold:
                any_pass = True
                break
        return winner if any_pass else unlabeled_name

    label = None if info.label is None else str(info.label).strip()
    if not label:
        return unlabeled_name
    label_threshold = _threshold_for_label(label, default_threshold=default_threshold, class_thresholds=overrides)
    if label_threshold is not None and info.score is not None and float(info.score) < label_threshold:
        return unlabeled_name
    return label


def _effective_class_labels(
    dataset: VisionDataset,
    record: VisionRecord,
    *,
    threshold: float | None = None,
    unlabeled_name: str = "unlabeled",
    multi_label: bool = False,
    class_thresholds: dict[str, float] | None = None,
) -> list[str]:
    if multi_label and str(dataset.meta.get("class_dir_ingest_mode") or "").lower() == "multi-label":
        raw = record.attributes.get(common.CLASS_DIR_MULTI_LABELS_ATTR)
        if isinstance(raw, list) and raw:
            return [str(item) for item in raw]
        return [unlabeled_name]
    if not multi_label:
        return [_effective_class_label(record, threshold=threshold, unlabeled_name=unlabeled_name, class_thresholds=class_thresholds)]

    info = record.classification
    if info is None:
        return [unlabeled_name]
    applied_threshold = _effective_threshold(record, threshold=threshold)
    overrides = class_thresholds or {}
    if applied_threshold is None and not overrides:
        return [_effective_class_label(record, threshold=None, unlabeled_name=unlabeled_name, class_thresholds=class_thresholds)]

    probs = common.classification_scores(info)
    if probs:
        passed: list[str] = []
        for label, score in probs.items():
            label_threshold = _threshold_for_label(label, default_threshold=applied_threshold, class_thresholds=overrides)
            if label_threshold is not None and score >= label_threshold:
                passed.append(label)
        return passed or [unlabeled_name]
    return [_effective_class_label(record, threshold=applied_threshold, unlabeled_name=unlabeled_name, class_thresholds=class_thresholds)]


def _split_flags(count: int, *, val_frac: float, test_frac: float) -> list[str]:
    test_n = int(math.floor(count * test_frac))
    val_n = int(math.floor(count * val_frac))
    train_n = max(0, count - test_n - val_n)
    return (["test"] * test_n) + (["val"] * val_n) + (["train"] * train_n)


def _assign_output_splits(
    dataset: VisionDataset,
    *,
    threshold: float | None = None,
    unlabeled_name: str = "unlabeled",
    class_thresholds: dict[str, float] | None = None,
    val_frac: float = 0.0,
    test_frac: float = 0.0,
    seed: int = 0,
    randomized: bool = False,
) -> list[str]:
    rng = random.Random(seed)
    split_by_index: dict[int, str] = {}
    if randomized:
        indices = list(range(len(dataset.records)))
        rng.shuffle(indices)
        flags = _split_flags(len(indices), val_frac=val_frac, test_frac=test_frac)
        split_by_index.update(zip(indices, flags))
    else:
        by_label: dict[str, list[int]] = {}
        for idx, record in enumerate(dataset.records):
            label = _effective_class_label(record, threshold=threshold, unlabeled_name=unlabeled_name, class_thresholds=class_thresholds)
            by_label.setdefault(label, []).append(idx)
        for label in sorted(by_label):
            indices = by_label[label]
            rng.shuffle(indices)
            flags = _split_flags(len(indices), val_frac=val_frac, test_frac=test_frac)
            split_by_index.update(zip(indices, flags))
    return [split_by_index.get(idx, "train") for idx, _record in enumerate(dataset.records)]


def _sanitize_label_parts(label: str) -> list[str]:
    parts: list[str] = []
    for raw_part in re.split(r"[\\/]+", str(label)):
        cleaned = str(raw_part).strip().replace(" ", "_")
        if not cleaned:
            continue
        if cleaned in {".", ".."}:
            cleaned = cleaned.replace(".", "_")
        parts.append(cleaned)
    return parts or ["_"]


def _path_component_labels(record: VisionRecord) -> list[str]:
    rel = record.rel_image_path
    if rel is None:
        return []
    _split, dirs = _split_and_class_parts(rel)
    return [part for part in dirs if part]


def _coerced_multi_labels(record: VisionRecord, *, class_name_mode: str) -> list[str]:
    rel = record.rel_image_path
    if rel is not None:
        _split, dirs = _split_and_class_parts(rel)
        if class_name_mode == "full_path":
            resolved = _resolve_class_name(dirs, class_name_mode=class_name_mode)
            return [resolved] if resolved else []
        return [part for part in dirs if part]
    if record.classification and record.classification.label:
        return [str(record.classification.label)]
    return []


def _multi_label_classes(root: Path, *, class_name_mode: str, max_depth: int = -1) -> list[str]:
    labels: set[str] = set()
    for image_path in _iter_images(root, max_depth=max_depth):
        _split, dirs = _split_and_class_parts(image_path.relative_to(root))
        if class_name_mode == "full_path":
            label = _resolve_class_name(dirs, class_name_mode=class_name_mode)
            if label:
                labels.add(label)
        else:
            labels.update(part for part in dirs if part)
    return sorted(labels)


def _next_output_path(out_dir: Path, image_name: str) -> Path:
    out_path = out_dir / image_name
    if not out_path.exists():
        return out_path
    stem, suffix = out_path.stem, out_path.suffix
    counter = 1
    while True:
        candidate = out_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


class ClassDirAdapter:
    @classmethod
    def matches(cls, src: Path) -> bool:
        if not src.is_dir():
            return False
        class_names: set[str] = set()
        duplicate_name_seen = False
        filename_to_classes: dict[str, set[str]] = {}
        for img in _iter_images(src):
            split, cls_name = _infer_split_and_class(src, img)
            if cls_name is None:
                continue
            class_names.add(cls_name)
            filename_to_classes.setdefault(img.name, set()).add(cls_name)
            if len(filename_to_classes[img.name]) > 1 and split == "train":
                duplicate_name_seen = True
        return duplicate_name_seen or len(class_names) >= 2

    @classmethod
    def read(cls, src: Path, *, task: str = "auto", splits: list[str] | None = None, **options: object) -> VisionDataset:
        root = src.resolve()
        if not root.is_dir():
            raise ValueError(f"class-dir input must be a directory: {root}")
        requested_task = common.canonical_task_id(task) if task not in {"", None} else None
        class_name_mode = _validate_class_name_mode(options.get("class_name_mode", "basename"))
        max_depth = int(options.get("max_depth", -1))

        top_level_split = any((root / name).exists() for name in _SPLIT_DIR_NAMES)
        scan_roots = _split_scan_roots(root, splits, top_level_split=top_level_split)

        # Duplicate-name detection only feeds the non-split multi-label ingest below.
        has_duplicate_names = False
        if not top_level_split:
            filename_to_classes: dict[str, set[str]] = {}
            for img in _iter_images(root, max_depth=max_depth):
                rel = img.relative_to(root)
                if len(rel.parts) < 2:
                    continue
                if rel.parts[0].lower() in _SPLIT_DIR_NAMES:
                    continue
                class_label = Path(*rel.parts[:-1]).as_posix()
                filename_to_classes.setdefault(rel.name, set()).add(class_label)
                if len(filename_to_classes[rel.name]) > 1:
                    has_duplicate_names = True

        dataset = _ingest_multi_label(root, max_depth=max_depth) if has_duplicate_names and not top_level_split else VisionDataset(records=[], root=root)
        if not dataset.records:
            records: list[VisionRecord] = []
            classes: list[str] = []
            seen: set[str] = set()
            for scan_root in scan_roots:
                for img in _iter_images(scan_root, max_depth=max_depth):
                    image = common.image_info_from_path(img)
                    image.path = common.root_relative_path(root, img)
                    split, cls_name = _infer_split_and_class(root, img, class_name_mode=class_name_mode)
                    record = VisionRecord(image=image, split=split, rel_image_path=image.path)
                    if cls_name:
                        if cls_name not in seen:
                            seen.add(cls_name)
                            classes.append(cls_name)
                        record.classification = Classification(label=cls_name)
                    records.append(record)
            dataset = VisionDataset(records=records, classes=classes, root=root, task="cls")
        else:
            dataset.task = "multi-cls"

        if requested_task == "multi-cls" and str(dataset.meta.get("class_dir_ingest_mode") or "").lower() != "multi-label":
            dataset.records = _group_records_multi_label(root, dataset.records, class_name_mode=class_name_mode)
            dataset.classes = _multi_label_classes(root, class_name_mode=class_name_mode, max_depth=max_depth) or dataset.classes
            dataset.meta["class_dir_ingest_mode"] = "multi-label"
            dataset.task = "multi-cls"
        if splits:
            allowed = {common.canonical_split_id(item) for item in splits}
            dataset.records = [record for record in dataset.records if record.split in allowed]
        return dataset

    @classmethod
    def write(
        cls,
        dataset: VisionDataset,
        dst: Path,
        *,
        splits: list[str] | None = None,
        threshold: float | None = None,
        class_thresholds: dict[str, float] | None = None,
        hardlink: bool = False,
        unlabeled_name: str = "unlabeled",
        val_frac: float = 0.0,
        test_frac: float = 0.0,
        seed: int = 0,
        randomized: bool = False,
        multi_label: bool = False,
        preserve_splits: bool = False,
        **options: object,
    ) -> None:
        if val_frac < 0 or test_frac < 0 or val_frac > 1 or test_frac > 1 or (val_frac + test_frac) > 1:
            raise ValueError("Split fractions must satisfy val_frac + test_frac <= 1.0.")

        filtered = dataset.records
        if splits:
            allowed = {common.canonical_split_id(item) for item in splits}
            filtered = [record for record in dataset.records if record.split in allowed]
        split_output = bool(preserve_splits or val_frac or test_frac)
        assigned_splits = [record.split for record in filtered]
        if split_output and not preserve_splits:
            temp_dataset = VisionDataset(records=filtered, classes=dataset.classes, root=dataset.root, meta=dataset.meta)
            assigned_splits = _assign_output_splits(
                temp_dataset,
                threshold=threshold,
                unlabeled_name=unlabeled_name,
                class_thresholds=class_thresholds,
                val_frac=val_frac,
                test_frac=test_frac,
                seed=seed,
                randomized=randomized,
            )

        dst.mkdir(parents=True, exist_ok=True)
        temp_dataset = VisionDataset(records=filtered, classes=dataset.classes, root=dataset.root, meta=dataset.meta)
        for idx, record in enumerate(filtered):
            labels = _effective_class_labels(
                temp_dataset,
                record,
                threshold=threshold,
                unlabeled_name=unlabeled_name,
                multi_label=multi_label,
                class_thresholds=class_thresholds,
            )
            for label in labels:
                parts = _sanitize_label_parts(label)
                if split_output:
                    parts.insert(0, assigned_splits[idx] or "train")
                out_dir = dst.joinpath(*parts)
                out_dir.mkdir(parents=True, exist_ok=True)
                src = common.resolve_record_image_path(dataset, record)
                out_path = _next_output_path(out_dir, src.name)
                if hardlink:
                    try:
                        os.link(src, out_path)
                    except OSError:
                        out_path.write_bytes(src.read_bytes())
                else:
                    out_path.write_bytes(src.read_bytes())
