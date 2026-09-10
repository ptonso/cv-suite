from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set

from PIL import Image
import yaml
from cvsuite.common.core import VisionDataset
from cvsuite.common.io import read_dataset, write_dataset
from cvsuite.common.io import common as class_dir_io

from cvsuite.common.core.enums import IMG_EXTS
from cvsuite.common.core import Classification, ImageRecord, Record
from cvsuite.common.core import normalize_dataset

Fmt = Literal["images", "unstructured", "multi-class"]

MULTI_CLASS_FMT = "multi-class"
CLASSIFY_INGEST_MODE_META = "classify_ingest_mode"
CLASSIFY_MULTI_CLASS_LABELS_ATTR = "classify_multi_class_labels"
_SPLIT_DIR_NAMES = {"train", "val", "valid", "validation", "test"}
VISIONBASE_MULTI_LABEL_META = "class_dir_ingest_mode"
VISIONBASE_MULTI_LABEL_VALUE = "multi-label"


def _load_json_safe(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def detect_format(src: Path, from_hint: str = "auto") -> Fmt:
    if from_hint != "auto":
        return from_hint  # type: ignore[return-value]
    return "images" if src.is_file() else "unstructured"


def looks_like_annotated_dataset(src: Path) -> bool:
    if not src.exists():
        return False

    if src.is_file():
        suffix = src.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return True
        if suffix == ".json":
            data = _load_json_safe(src) or {}
            return ("images" in data and "annotations" in data) or ("shapes" in data)
        return False

    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        if (src / name).exists():
            return True

    split_dirs = [d for d in ("train", "val", "valid", "test") if (src / d).exists()]
    for split_name in split_dirs:
        split_dir = src / split_name
        if (split_dir / "images").exists() and any(
            (split_dir / label_dir).exists()
            for label_dir in ("labels", "labels_det", "labels_seg", "labels_pose")
        ):
            return True

    for json_path in list(src.rglob("*.json"))[:5]:
        data = _load_json_safe(json_path) or {}
        if ("images" in data and "annotations" in data) or ("shapes" in data):
            return True
    return False


def _iter_images(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def _infer_split_and_class(root: Path, img: Path) -> tuple[str, Optional[str]]:
    rel = img.relative_to(root)
    dirs = list(rel.parts[:-1])
    split = "train"
    if dirs:
        first = dirs[0].lower()
        if first in {"train", "val", "valid", "validation", "test"}:
            split = "val" if first in {"val", "valid", "validation"} else first
            dirs = dirs[1:]
    cls_name = None
    if dirs:
        if dirs[-1].lower() == "images" and len(dirs) >= 2:
            cls_name = dirs[-2]
        elif dirs[-1].lower() != "images":
            cls_name = dirs[-1]
    return split, cls_name


def _multi_class_labels(record: Record) -> list[str]:
    raw = record.attributes.get(CLASSIFY_MULTI_CLASS_LABELS_ATTR)
    if not isinstance(raw, list):
        raw = record.attributes.get(class_dir_io.CLASS_DIR_MULTI_LABELS_ATTR)
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        label = str(item).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def is_multi_class_ingest(dataset: VisionDataset) -> bool:
    value = dataset.meta.get(CLASSIFY_INGEST_MODE_META)
    if str(value or "").strip().lower() == MULTI_CLASS_FMT:
        return True
    return str(dataset.meta.get(VISIONBASE_MULTI_LABEL_META) or "").strip().lower() == VISIONBASE_MULTI_LABEL_VALUE


def resolve_output_multi_class(dataset: VisionDataset, *, requested: bool = False) -> bool:
    return bool(requested or is_multi_class_ingest(dataset))


def resolve_output_labels(
    dataset: VisionDataset,
    record: Record,
    *,
    threshold: float | None = None,
    unlabeled_name: str = "unlabeled",
    multi_class: bool = False,
    class_thresholds: Dict[str, float] | None = None,
) -> list[str]:
    if multi_class and is_multi_class_ingest(dataset):
        labels = _multi_class_labels(record)
        return labels or [unlabeled_name]
    return effective_class_labels(
        record,
        threshold=threshold,
        unlabeled_name=unlabeled_name,
        multi_class=multi_class,
        class_thresholds=class_thresholds,
    )


def _compare_multi_class_images(key: str, paths: list[Path]) -> tuple[int, int]:
    canonical = paths[0]
    with Image.open(canonical) as image:
        width, height = image.size
    payload = canonical.read_bytes()
    for other in paths[1:]:
        with Image.open(other) as image:
            other_width, other_height = image.size
        if (other_width, other_height) != (width, height):
            raise SystemExit(
                f"Duplicate filename {key!r} maps to different image sizes across classes. "
                "Rename one of the files or use `--from unstructured`."
            )
        if other.read_bytes() != payload:
            raise SystemExit(
                f"Duplicate filename {key!r} maps to different image contents across classes. "
                "Rename one of the files or use `--from unstructured`."
            )
    return width, height


def _ingest_multi_class(src: Path) -> VisionDataset:
    if not src.is_dir():
        raise SystemExit("`cvsuite class --from multi-class` expects a directory tree rooted at class folders.")

    root = src
    grouped_paths: dict[str, list[Path]] = {}
    grouped_labels: dict[str, set[str]] = {}
    classes: list[str] = []
    seen_classes: set[str] = set()

    for img in _iter_images(src):
        rel = img.relative_to(root)
        if len(rel.parts) < 2:
            raise SystemExit(
                "`cvsuite class --from multi-class` expects images under class folders like "
                "`<src>/<class>/<filename>`. Files at the source root are not supported."
            )
        first = rel.parts[0].lower()
        if first in _SPLIT_DIR_NAMES:
            raise SystemExit(
                "`cvsuite class --from multi-class` does not support split-prefixed trees like train/val/test. "
                "Use `--from unstructured` instead."
            )
        class_label = Path(*rel.parts[:-1]).as_posix()
        grouped_paths.setdefault(rel.name, []).append(img)
        labels = grouped_labels.setdefault(rel.name, set())
        labels.add(class_label)
        if class_label not in seen_classes:
            seen_classes.add(class_label)
            classes.append(class_label)

    records: list[Record] = []
    for filename in sorted(grouped_paths):
        paths = sorted(grouped_paths[filename])
        width, height = _compare_multi_class_images(filename, paths)
        labels = sorted(grouped_labels[filename])
        probs = {label: 1.0 for label in labels}
        records.append(
            Record(
                image=ImageRecord(path=paths[0], width=width, height=height),
                split="train",
                rel_image_path=Path(filename),
                classification=Classification(label=labels[0], score=1.0, probs=probs),
                attributes={
                    CLASSIFY_MULTI_CLASS_LABELS_ATTR: labels,
                    class_dir_io.CLASS_DIR_MULTI_LABELS_ATTR: labels,
                },
            )
        )

    return VisionDataset(
        records=records,
        classes=classes,
        root=root,
        meta={
            "format": MULTI_CLASS_FMT,
            CLASSIFY_INGEST_MODE_META: MULTI_CLASS_FMT,
            VISIONBASE_MULTI_LABEL_META: VISIONBASE_MULTI_LABEL_VALUE,
        },
    )


def ingest(src: Path, from_hint: str = "auto") -> VisionDataset:
    fmt = detect_format(src, from_hint)
    if fmt == MULTI_CLASS_FMT:
        return normalize_dataset(_ingest_multi_class(src))
    dataset = read_dataset(
        src,
        format="flat" if fmt == "images" else "class-dir",
        task="auto",
    )
    dataset.meta["format"] = fmt
    return normalize_dataset(dataset)


def effective_class_label(
    record: Record,
    *,
    threshold: float | None = None,
    unlabeled_name: str = "unlabeled",
    class_thresholds: Dict[str, float] | None = None,
) -> str:
    info = record.classification
    if info is None:
        return unlabeled_name

    scores = _classification_scores(info)
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
    score = info.score
    if not label:
        return unlabeled_name
    label_threshold = _threshold_for_label(label, default_threshold=default_threshold, class_thresholds=overrides)
    if label_threshold is not None and score is not None and float(score) < label_threshold:
        return unlabeled_name
    return label


def _effective_threshold(record: Record, *, threshold: float | None = None) -> float | None:
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


def _classification_scores(info: Classification | None) -> dict[str, float]:
    if info is None:
        return {}
    out: dict[str, float] = {}
    probs = info.probs if isinstance(info.probs, dict) else {}
    for label, score in probs.items():
        text = str(label).strip()
        if not text:
            continue
        try:
            out[text] = float(score)
        except (TypeError, ValueError):
            continue
    return out


def _winner_by_score(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    winner_label, _winner_score = max(scores.items(), key=lambda item: item[1])
    return winner_label


def _threshold_for_label(
    label: str,
    *,
    default_threshold: float | None,
    class_thresholds: Dict[str, float] | None,
) -> float | None:
    if class_thresholds and label in class_thresholds:
        return float(class_thresholds[label])
    return default_threshold


def parse_class_thresholds_yaml(path: Path) -> Dict[str, float]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Class-thresholds YAML must be a flat mapping of class names to floats: {path}")
    out: Dict[str, float] = {}
    for raw_key, raw_value in data.items():
        key = str(raw_key).strip()
        if not key:
            continue
        try:
            out[key] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid threshold for class {key!r} in {path}: {raw_value!r}") from exc
    return out


def parse_class_threshold_overrides(items: Iterable[str] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items or []:
        text = str(item).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Class-threshold override must use class=value syntax: {text!r}")
        raw_key, raw_value = text.split("=", 1)
        key = raw_key.strip()
        if not key:
            raise ValueError(f"Class-threshold override is missing a class name: {text!r}")
        try:
            out[key] = float(raw_value.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid threshold for class {key!r}: {raw_value!r}") from exc
    return out


def resolve_class_thresholds(
    *,
    yaml_path: Path | None = None,
    cli_overrides: Iterable[str] | None = None,
) -> Dict[str, float]:
    resolved: Dict[str, float] = {}
    if yaml_path is not None:
        resolved.update(parse_class_thresholds_yaml(yaml_path))
    resolved.update(parse_class_threshold_overrides(cli_overrides))
    return resolved


def effective_class_labels(
    record: Record,
    *,
    threshold: float | None = None,
    unlabeled_name: str = "unlabeled",
    multi_class: bool = False,
    class_thresholds: Dict[str, float] | None = None,
) -> list[str]:
    if not multi_class:
        return [
            effective_class_label(
                record,
                threshold=threshold,
                unlabeled_name=unlabeled_name,
                class_thresholds=class_thresholds,
            )
        ]

    info = record.classification
    if info is None:
        return [unlabeled_name]

    applied_threshold = _effective_threshold(record, threshold=threshold)
    overrides = class_thresholds or {}
    if applied_threshold is None and not overrides:
        return [
            effective_class_label(
                record,
                threshold=None,
                unlabeled_name=unlabeled_name,
                class_thresholds=class_thresholds,
            )
        ]

    probs = _classification_scores(info)
    if probs:
        passed: list[str] = []
        for label, score in probs.items():
            label_threshold = _threshold_for_label(label, default_threshold=applied_threshold, class_thresholds=overrides)
            if label_threshold is None:
                continue
            if score >= label_threshold:
                passed.append(label)
        if passed:
            seen: set[str] = set()
            out: list[str] = []
            for label in passed:
                if label in seen:
                    continue
                seen.add(label)
                out.append(label)
            return out
        return [unlabeled_name]

    return [
        effective_class_label(
            record,
            threshold=applied_threshold,
            unlabeled_name=unlabeled_name,
            class_thresholds=class_thresholds,
        )
    ]


def validate_split_fractions(*, val_frac: float = 0.0, test_frac: float = 0.0) -> tuple[float, float]:
    val_frac = float(val_frac or 0.0)
    test_frac = float(test_frac or 0.0)
    if val_frac < 0 or test_frac < 0:
        raise ValueError("Split fractions must be non-negative.")
    if val_frac > 1 or test_frac > 1 or (val_frac + test_frac) > 1:
        raise ValueError("Split fractions must satisfy val_frac + test_frac <= 1.0.")
    return val_frac, test_frac


def _split_flags(count: int, *, val_frac: float, test_frac: float) -> list[str]:
    test_n = int(math.floor(count * test_frac))
    val_n = int(math.floor(count * val_frac))
    train_n = max(0, count - test_n - val_n)
    return (["test"] * test_n) + (["val"] * val_n) + (["train"] * train_n)


def assign_output_splits(
    dataset: VisionDataset,
    *,
    threshold: float | None = None,
    unlabeled_name: str = "unlabeled",
    class_thresholds: Dict[str, float] | None = None,
    val_frac: float = 0.0,
    test_frac: float = 0.0,
    seed: int = 0,
    randomized: bool = False,
) -> VisionDataset:
    val_frac, test_frac = validate_split_fractions(val_frac=val_frac, test_frac=test_frac)
    rng = random.Random(seed)

    split_by_index: Dict[int, str] = {}
    if randomized:
        indices = list(range(len(dataset.records)))
        rng.shuffle(indices)
        flags = _split_flags(len(indices), val_frac=val_frac, test_frac=test_frac)
        split_by_index.update(zip(indices, flags))
    else:
        by_label: Dict[str, List[int]] = {}
        for idx, record in enumerate(dataset.records):
            label = effective_class_label(
                record,
                threshold=threshold,
                unlabeled_name=unlabeled_name,
                class_thresholds=class_thresholds,
            )
            by_label.setdefault(label, []).append(idx)
        for label in sorted(by_label):
            indices = by_label[label]
            rng.shuffle(indices)
            flags = _split_flags(len(indices), val_frac=val_frac, test_frac=test_frac)
            split_by_index.update(zip(indices, flags))

    records = [
        replace(record, split=split_by_index.get(idx, "train"))
        for idx, record in enumerate(dataset.records)
    ]
    meta = dict(dataset.meta)
    meta["output_split_assignment"] = {
        "val_frac": val_frac,
        "test_frac": test_frac,
        "seed": seed,
        "randomized": randomized,
        "threshold": threshold,
        "unlabeled_name": unlabeled_name,
        "class_thresholds": dict(class_thresholds or {}),
    }
    return replace(dataset, records=records, meta=meta)


def _strip_excluded_from_record(record: Record, excluded: Set[str]) -> Record | None:
    """Drop excluded class names from a record; return None when nothing is left to write."""
    attrs = dict(record.attributes)
    multi_labels: list[str] | None = None
    for key in (CLASSIFY_MULTI_CLASS_LABELS_ATTR, class_dir_io.CLASS_DIR_MULTI_LABELS_ATTR):
        raw = attrs.get(key)
        if isinstance(raw, list):
            multi_labels = [item for item in raw if str(item).strip() not in excluded]
            attrs[key] = multi_labels
    if multi_labels is not None and not multi_labels:
        return None

    info = record.classification
    if info is not None:
        probs = info.probs if isinstance(info.probs, dict) else {}
        if probs:
            filtered = {k: v for k, v in probs.items() if str(k).strip() not in excluded}
            if not filtered:
                return None
            label = info.label
            if label is None or str(label).strip() in excluded:
                label = _winner_by_score(_classification_scores(replace(info, probs=filtered)))
            info = replace(info, label=label, probs=filtered)
        elif info.label is not None and str(info.label).strip() in excluded:
            if multi_labels:
                info = replace(info, label=multi_labels[0])
            else:
                return None

    return replace(record, classification=info, attributes=attrs)


def prune_excluded_classes(dataset: VisionDataset, excluded: Iterable[str] | None) -> VisionDataset:
    """Remove whole classes from the dataset before it reaches the class-dir writer."""
    excluded_set = {str(name).strip() for name in (excluded or []) if str(name).strip()}
    if not excluded_set:
        return dataset
    records = [
        pruned
        for record in dataset.records
        if (pruned := _strip_excluded_from_record(record, excluded_set)) is not None
    ]
    classes = [cls for cls in dataset.classes if str(cls).strip() not in excluded_set]
    return replace(dataset, records=records, classes=classes)


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


def _next_output_path(out_dir: Path, image_path: Path) -> Path:
    out_path = out_dir / image_path.name
    if not out_path.exists():
        return out_path
    stem, suffix = out_path.stem, out_path.suffix
    counter = 1
    while True:
        candidate = out_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def write_class_dir(
    dataset: VisionDataset,
    dst: Path,
    *,
    threshold: float | None = None,
    class_thresholds: Dict[str, float] | None = None,
    hardlink: bool = False,
    unlabeled_name: str = "unlabeled",
    val_frac: float = 0.0,
    test_frac: float = 0.0,
    seed: int = 0,
    randomized: bool = False,
    multi_class: bool = False,
    preserve_splits: bool = False,
    exclude_classes: Iterable[str] | None = None,
) -> VisionDataset:
    dataset = prune_excluded_classes(dataset, exclude_classes)
    multi_class = resolve_output_multi_class(dataset, requested=multi_class)
    split_output = bool(preserve_splits or val_frac or test_frac)
    if not preserve_splits:
        val_frac, test_frac = validate_split_fractions(val_frac=val_frac, test_frac=test_frac)
    if split_output and not preserve_splits:
        dataset = assign_output_splits(
            dataset,
            threshold=threshold,
            unlabeled_name=unlabeled_name,
            class_thresholds=class_thresholds,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            randomized=randomized,
        )
    write_dataset(
        dataset,
        dst,
        format="class-dir",
        threshold=threshold,
        class_thresholds=class_thresholds,
        hardlink=hardlink,
        unlabeled_name=unlabeled_name,
        multi_label=multi_class,
        preserve_splits=split_output,
    )
    return dataset


def write_classified(dataset: VisionDataset, dst: Path, hardlink: bool = False, unlabeled_name: str = "unlabeled") -> None:
    write_class_dir(dataset, dst, hardlink=hardlink, unlabeled_name=unlabeled_name)
