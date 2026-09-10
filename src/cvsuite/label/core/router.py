from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import yaml

from cvsuite.common.core import BBox, Keypoints, Polygon, VisionDataset, VisionRecord
from cvsuite.common.io import read_dataset

from cvsuite.common.core.enums import IMG_EXTS
from cvsuite.common.core import normalize_dataset

Fmt = Literal["yolo", "labelme", "coco", "images", "unstructured", "class-dir", "semseg-mask"]

_SEMSEG_SPLIT_KEYS = ("train", "val", "valid", "test", "infer")


def _load_json_safe(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _looks_like_semseg_manifest(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if str(payload.get("format", "")).strip().lower() in {"semseg_mask", "semseg-mask"}:
        return True
    for key in _SEMSEG_SPLIT_KEYS:
        value = payload.get(key)
        if isinstance(value, dict) and {"images", "masks"}.issubset(value):
            return True
    return False


def detect_format(src: Path, from_hint: str = "auto") -> Fmt:
    if from_hint != "auto":
        return from_hint  # type: ignore[return-value]

    path = src
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".yml", ".yaml"}:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            fmt = str(data.get("format", "")).lower()
            if fmt == "coco":
                return "coco"
            if fmt == "labelme":
                return "labelme"
            if _looks_like_semseg_manifest(data):
                return "semseg-mask"
            return "yolo"
        if suffix == ".json":
            data = _load_json_safe(path) or {}
            if "images" in data and "annotations" in data:
                return "coco"
            if "shapes" in data:
                return "labelme"
            return "coco"
        if suffix in IMG_EXTS:
            return "images"

    if path.is_dir():
        for name in ("data.yaml", "data.yml"):
            cfg = path / name
            if cfg.exists():
                data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                fmt = str(data.get("format", "")).lower()
                if fmt == "coco":
                    return "coco"
                if fmt == "labelme":
                    return "labelme"
                if _looks_like_semseg_manifest(data):
                    return "semseg-mask"
                return "yolo"
        for json_path in list(path.rglob("*.json"))[:5]:
            data = _load_json_safe(json_path) or {}
            if "shapes" in data:
                return "labelme"
            if "images" in data and "annotations" in data:
                return "coco"
        split_dirs = [name for name in ("train", "val", "valid", "test") if (path / name).exists()]
        for split_name in split_dirs:
            split_dir = path / split_name
            if (split_dir / "images").exists() and any(
                (split_dir / label_dir).exists()
                for label_dir in ("labels", "labels_det", "labels_seg", "labels_pose")
            ):
                return "yolo"
        for subdir in path.iterdir():
            if not subdir.is_dir():
                continue
            if (subdir / "images").exists() and any(
                (subdir / label_dir).exists()
                for label_dir in ("labels", "labels_det", "labels_seg", "labels_pose")
            ):
                return "yolo"
        for item in path.rglob("*"):
            if item.is_file() and item.suffix.lower() in IMG_EXTS:
                return "unstructured"

    raise RuntimeError(f"Could not infer dataset format from: {src}")


def _assign_split(dataset: VisionDataset, split: str) -> None:
    for record in dataset.records:
        record.split = split


def _assign_split_if_missing(dataset: VisionDataset, split: str) -> None:
    if all(record.split for record in dataset.records):
        return
    _assign_split(dataset, split)


def _resolve_common_root(srcs: Sequence[Path], datasets: Sequence[VisionDataset]) -> Path | None:
    bases: list[Path] = []
    for dataset, src in zip(datasets, srcs):
        base = dataset.root or src
        if base.is_file():
            base = base.parent
        bases.append(base.resolve())
    if not bases:
        return None
    try:
        return Path(os.path.commonpath([str(base) for base in bases]))
    except ValueError:
        return None


def _ensure_class_id(classes: list[str], name: str) -> int:
    if name in classes:
        return classes.index(name)
    classes.append(name)
    return len(classes) - 1


def _annotation_class_name(source_classes: Sequence[str], cls_idx: int, label: Optional[str]) -> str:
    if label:
        return label
    if 0 <= cls_idx < len(source_classes):
        return str(source_classes[cls_idx])
    return str(cls_idx)


def _remap_annotation_classes(record: VisionRecord, source_classes: Sequence[str], merged_classes: list[str]) -> None:
    def _remap(items: Sequence[BBox | Polygon | Keypoints]) -> None:
        for item in items:
            name = _annotation_class_name(source_classes, item.cls, getattr(item, "label", None))
            item.cls = _ensure_class_id(merged_classes, name)
            if getattr(item, "label", None) is None:
                item.label = name

    _remap(record.boxes)
    _remap(record.polys)
    _remap(record.kpts)


def _normalize_record_paths(record: VisionRecord, base_root: Path | None) -> None:
    if record.image is not None and not record.image.path.is_absolute():
        anchor = base_root or Path.cwd()
        record.image.path = (anchor / record.image.path).resolve()


def _canonical_task_hint(task: str) -> str:
    raw = str(task or "auto").strip().lower()
    if raw in {"", "auto"}:
        return "auto"
    if raw == "seg":
        return "inst-seg"
    return raw


def _io_format(fmt: Fmt) -> str:
    if fmt in {"images", "unstructured"}:
        return "flat"
    if fmt == "semseg-mask":
        return "semseg_mask"
    return fmt


def ingest_many(
    srcs: Sequence[Path],
    from_hint: str = "auto",
    task: str = "auto",
    keypoints_file: Optional[Path] = None,
    max_depth: int = -1,
) -> VisionDataset:
    if not srcs:
        raise ValueError("At least one source path is required.")
    if len(srcs) == 1:
        return ingest(srcs[0], from_hint=from_hint, task=task, keypoints_file=keypoints_file, max_depth=max_depth)

    datasets = [
        ingest(src, from_hint=from_hint, task=task, keypoints_file=keypoints_file, max_depth=max_depth)
        for src in srcs
    ]

    merged_classes: list[str] = []
    merged_records: list[VisionRecord] = []
    non_null_tasks = {dataset.task for dataset in datasets if dataset.task is not None}
    merged_task = next(iter(non_null_tasks)) if len(non_null_tasks) == 1 else None

    for dataset in datasets:
        for cls_name in dataset.classes:
            _ensure_class_id(merged_classes, str(cls_name))
        for record in dataset.records:
            merged = deepcopy(record)
            _normalize_record_paths(merged, dataset.root or src)
            _remap_annotation_classes(merged, dataset.classes, merged_classes)
            if merged.classification and merged.classification.label:
                _ensure_class_id(merged_classes, str(merged.classification.label))
            merged_records.append(merged)

    meta = {
        "multisrc": True,
        "sources": [str(src) for src in srcs],
        "source_formats": [
            str(dataset.source_format or detect_format(src, from_hint))
            for src, dataset in zip(srcs, datasets)
        ],
    }
    dataset = VisionDataset(
        records=merged_records,
        classes=merged_classes,
        task=merged_task,
        root=_resolve_common_root(srcs, datasets),
        meta=meta,
    )
    return normalize_dataset(dataset)


def ingest(
    src: Path,
    from_hint: str = "auto",
    task: str = "auto",
    keypoints_file: Optional[Path] = None,
    max_depth: int = -1,
) -> VisionDataset:
    fmt = detect_format(src, from_hint)
    dataset = read_dataset(
        src,
        format=_io_format(fmt),
        task=_canonical_task_hint(task),
        keypoints_file=keypoints_file,
        max_depth=max_depth,
    )
    dataset.meta.setdefault("format", fmt)
    if dataset.source_format is None:
        dataset.source_format = _io_format(fmt)
    _assign_split_if_missing(dataset, "train")
    return normalize_dataset(dataset)
