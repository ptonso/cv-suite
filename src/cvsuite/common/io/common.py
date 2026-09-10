"""Shared helpers for cvsuite IO adapters.

These utilities assume canonical `VisionDataset` and `VisionRecord` objects in
memory and handle the cross-format mechanics around YAML/JSON parsing, lazy
image metadata reads, path resolution, sample keys, and writer file copying.
"""

from __future__ import annotations

import io as bytes_io
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import yaml
from PIL import Image

from cvsuite.common.core import Classification, ImageInfo, VisionDataset, VisionRecord, VQA

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".ppm", ".tif", ".tiff", ".webp"}
DEFAULT_MANIFEST_NAMES = ("vqa.json", "manifest.json")

SAMPLE_KEY_ATTR = "vlm_sample_key"
ITEM_META_ATTR = "vlm_item_meta"
IMAGE_ID_ATTR = "vlm_image_id"
PREDOMINANT_LABEL_ATTR = "vlm_predominant_label"
CLEANUP_DIRS_META = "cleanup_dirs"
CLASS_DIR_MULTI_LABELS_ATTR = "class_dir_multi_labels"

LABELME_META_KEY = "cvsuite"
LABELME_PROMOTED_FIELDS = ("score", "prompt", "model", "text")

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_yaml_or_json(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return read_json(path)
    return read_yaml(path)


def ensure_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def canonical_task_id(task: object, *, allow_auto: bool = True) -> str | None:
    if task is None:
        return None
    raw = str(getattr(task, "value", task)).strip().lower()
    if not raw:
        return None
    if raw == "auto" and allow_auto:
        return "auto"
    aliases = {
        "cls": "cls",
        "classification": "cls",
        "multi-cls": "multi-cls",
        "multi_cls": "multi-cls",
        "multi-class": "multi-cls",
        "multiclass": "multi-cls",
        "multi-label": "multi-cls",
        "multilabel": "multi-cls",
        "det": "det",
        "detect": "det",
        "detection": "det",
        "obb": "obb",
        "inst-seg": "inst-seg",
        "inst_seg": "inst-seg",
        "seg": "inst-seg",
        "segment": "inst-seg",
        "segmentation": "inst-seg",
        "sem-seg": "sem-seg",
        "sem_seg": "sem-seg",
        "semantic-seg": "sem-seg",
        "semantic-segmentation": "sem-seg",
        "pose": "pose",
        "keypoint": "pose",
        "keypoints": "pose",
    }
    if raw not in aliases:
        raise ValueError(f"Unsupported task id: {task!r}")
    return aliases[raw]


def yolo_task_name(task: object) -> str:
    canonical = canonical_task_id(task, allow_auto=False)
    mapping = {
        "det": "det",
        "inst-seg": "seg",
        "pose": "pose",
    }
    if canonical not in mapping:
        raise ValueError(f"YOLO format does not support task {canonical!r}")
    return mapping[canonical]


def canonical_split_id(raw: object, *, default: str = "train") -> str:
    value = str(raw or default).strip().lower()
    if not value:
        return default
    aliases = {
        "train": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "eval": "test",
        "infer": "infer",
    }
    return aliases.get(value, value if value in {"train", "val", "test", "infer"} else default)


def image_info_from_path(path: Path) -> ImageInfo:
    return ImageInfo(path=path)


def iter_files(root: Path, *, max_depth: int = -1) -> Iterator[Path]:
    """Yield files under root. max_depth=-1 unlimited; 0 = only files directly in root.
    Depth counts directory levels below root."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if max_depth >= 0 and (len(path.relative_to(root).parts) - 1) > max_depth:
            continue
        yield path


def root_relative_path(root: Path | None, path: Path) -> Path:
    if root is None:
        return Path(path.name)
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def resolve_record_image_path(dataset: VisionDataset, record: VisionRecord) -> Path:
    if record.image is None:
        raise ValueError("VisionRecord.image is required")
    path = record.image.path
    if path.is_absolute():
        return path
    if dataset.root is not None:
        return dataset.root / path
    return path


def resolve_mask_path(dataset: VisionDataset, record: VisionRecord) -> Path:
    if record.semantic_mask is None:
        raise ValueError("VisionRecord.semantic_mask is required")
    path = record.semantic_mask.path
    if path.is_absolute():
        return path
    if dataset.root is not None:
        return dataset.root / path
    return path


def copy_or_link(src: Path, dst: Path, *, hardlink: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def read_image_bytes(dataset: VisionDataset, record: VisionRecord, *, force_jpeg: bool = False) -> tuple[bytes, str]:
    src = resolve_record_image_path(dataset, record)
    suffix = src.suffix.lower() or ".jpg"
    if not force_jpeg and suffix in IMG_EXTS:
        return src.read_bytes(), suffix
    with Image.open(src) as image:
        rgb = image.convert("RGB")
        buffer = bytes_io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue(), ".jpg"


def sanitize_sample_key(value: str) -> str:
    cleaned = _SANITIZE_RE.sub("_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "sample"


def derive_sample_key_from_path(path: Path) -> str:
    no_suffix = path.with_suffix("") if path.suffix else path
    return sanitize_sample_key(no_suffix.as_posix().replace("/", "__"))


def record_relative_image_path(record: VisionRecord) -> Path:
    if record.rel_image_path is not None:
        rel = record.rel_image_path
    elif record.image is not None:
        rel = record.image.path
    else:
        rel = Path("sample")
    if rel.is_absolute():
        return Path(rel.name)
    return rel


def record_sample_key(record: VisionRecord) -> str:
    raw = record.attributes.get(SAMPLE_KEY_ATTR)
    if isinstance(raw, str) and raw.strip():
        return sanitize_sample_key(raw)
    rel = record.rel_image_path
    if isinstance(rel, Path):
        return derive_sample_key_from_path(rel)
    if record.image is None:
        return "sample"
    return derive_sample_key_from_path(record.image.path)


def unique_sample_keys(records: Sequence[VisionRecord]) -> list[str]:
    used: dict[str, int] = {}
    out: list[str] = []
    for record in records:
        base = record_sample_key(record)
        idx = used.get(base, 0)
        key = base if idx == 0 else f"{base}_{idx}"
        while key in used:
            idx += 1
            key = f"{base}_{idx}"
        used[base] = idx + 1
        used[key] = 1
        out.append(key)
    return out


def record_item_meta(record: VisionRecord) -> dict[str, Any]:
    raw = record.attributes.get(ITEM_META_ATTR)
    return dict(raw) if isinstance(raw, dict) else {}


def promoted_item_fields(record: VisionRecord) -> dict[str, Any]:
    item_meta = record_item_meta(record)
    out: dict[str, Any] = {}

    image_id = record.attributes.get(IMAGE_ID_ATTR)
    if image_id is None and "image_id" in item_meta:
        image_id = item_meta["image_id"]
    if image_id is not None:
        out["image_id"] = image_id

    predominant_label = record.attributes.get(PREDOMINANT_LABEL_ATTR)
    if predominant_label is None and "predominant_label" in item_meta:
        predominant_label = item_meta["predominant_label"]
    if predominant_label is not None:
        out["predominant_label"] = predominant_label

    original_filename = item_meta.get("original_filename")
    if original_filename is None and record.rel_image_path is not None:
        original_filename = record.rel_image_path.name
    if original_filename is not None:
        out["original_filename"] = original_filename
    return out


def qa_from_payload(payload: dict[str, Any], *, source: str, question_keys: Sequence[str]) -> VQA:
    question = ""
    for key in question_keys:
        raw_question = payload.get(key)
        if raw_question is not None:
            question = str(raw_question).strip()
            if question:
                break
    if not question:
        raise ValueError("Question payload is missing a question/prompt string.")

    meta: dict[str, Any] = {}
    if payload.get("id") is not None:
        meta["id"] = payload["id"]
    if payload.get("question_id") is not None and "id" not in meta:
        meta["id"] = payload["question_id"]
    if payload.get("label") is not None:
        meta["label"] = payload["label"]
    if payload.get("ground_truth") is not None:
        meta["ground_truth"] = payload["ground_truth"]
    if payload.get("answers") is not None:
        meta["expected_answers"] = payload["answers"]

    extra_meta = payload.get("meta")
    extra = dict(extra_meta) if isinstance(extra_meta, dict) else {}
    source_value = payload.get("source")
    if source_value is None:
        source_value = extra.pop("source", source)
    meta["source"] = str(source_value) if source_value is not None else source
    if extra:
        meta["source_meta"] = extra

    return VQA(
        question=question,
        answer=str(payload.get("answer", "") or ""),
        score=payload.get("score"),
        model=(str(payload["model"]) if payload.get("model") else None),
        meta=meta,
    )


def qa_to_payload(qa: VQA, *, question_key: str, id_key: str = "id") -> dict[str, Any]:
    payload: dict[str, Any] = {question_key: qa.question}
    meta = dict(qa.meta or {})

    qa_id = meta.pop("id", None)
    if qa_id is not None:
        payload[id_key] = qa_id
    label = meta.pop("label", None)
    if label is not None:
        payload["label"] = label
    ground_truth = meta.pop("ground_truth", None)
    if ground_truth is not None:
        payload["ground_truth"] = ground_truth
    expected_answers = meta.pop("expected_answers", None)
    if expected_answers is not None:
        payload["answers"] = expected_answers
    source = meta.pop("source", None)
    source_meta = meta.pop("source_meta", None)

    if qa.answer:
        payload["answer"] = qa.answer
    if qa.score is not None:
        payload["score"] = qa.score
    if qa.model:
        payload["model"] = qa.model

    merged_meta: dict[str, Any] = {}
    if isinstance(source_meta, dict):
        merged_meta.update(source_meta)
    merged_meta.update(meta)
    if source is not None:
        merged_meta["source"] = source
    if merged_meta:
        payload["meta"] = merged_meta
    return payload


def ensure_has_vqas(dataset: VisionDataset, *, allow_empty: bool = False) -> None:
    if allow_empty and not dataset.records:
        return
    if any(record.vqas for record in dataset.records):
        return
    raise ValueError("This output requires at least one question-answer entry in the dataset.")


def classification_scores(info: Classification | None) -> dict[str, float]:
    if info is None or not isinstance(info.probs, dict):
        return {}
    out: dict[str, float] = {}
    for label, score in info.probs.items():
        text = str(label).strip()
        if not text:
            continue
        try:
            out[text] = float(score)
        except (TypeError, ValueError):
            continue
    return out


def cleanup_dataset_resources(dataset: VisionDataset | None) -> None:
    if dataset is None:
        return
    raw = dataset.meta.get(CLEANUP_DIRS_META, [])
    if not isinstance(raw, list):
        return
    for item in raw:
        if item:
            shutil.rmtree(Path(str(item)), ignore_errors=True)
