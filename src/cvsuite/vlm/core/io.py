from __future__ import annotations

import io as bytes_io
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from PIL import Image

from cvsuite.common.core.enums import IMG_EXTS
from cvsuite.common.core import ImageRecord, Record, VQA
from cvsuite.common.core import VisionDataset

DEFAULT_MANIFEST_NAMES = ("vqa.json", "manifest.json")
CAPTION_PROMPT = "Describe this image in one concise sentence."

SAMPLE_KEY_ATTR = "vlm_sample_key"
ITEM_META_ATTR = "vlm_item_meta"
IMAGE_ID_ATTR = "vlm_image_id"
PREDOMINANT_LABEL_ATTR = "vlm_predominant_label"
ANSWER_BUCKET_ATTR = "vlm_answer_bucket"
CLEANUP_DIRS_META = "cleanup_dirs"
ANSWER_BUCKETS_META = "vlm_answer_buckets"

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_VQA_STD_KEYS = {"id", "label", "ground_truth", "expected_answers", "source", "source_meta"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml_or_json(path: Path) -> Any:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def image_record_from_path(path: Path) -> ImageRecord:
    with Image.open(path) as image:
        width, height = image.size
    return ImageRecord(path=path, width=width, height=height)


def normalize_split(raw: object) -> str:
    value = str(raw or "train").strip()
    return value or "train"


def sanitize_sample_key(value: str) -> str:
    cleaned = _SANITIZE_RE.sub("_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "sample"


def derive_sample_key_from_path(path: Path) -> str:
    no_suffix = path.with_suffix("") if path.suffix else path
    return sanitize_sample_key(no_suffix.as_posix().replace("/", "__"))


def record_relative_image_path(record: Record) -> Path:
    rel = record.rel_image_path or record.image.path
    if rel.is_absolute():
        return Path(rel.name)
    return rel


def record_sample_key(record: Record) -> str:
    raw = record.attributes.get(SAMPLE_KEY_ATTR)
    if isinstance(raw, str) and raw.strip():
        return sanitize_sample_key(raw)
    rel = record.rel_image_path
    if isinstance(rel, Path):
        return derive_sample_key_from_path(rel)
    return derive_sample_key_from_path(record.image.path)


def unique_sample_keys(records: Sequence[Record]) -> list[str]:
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


def resolve_record_image_path(dataset: VisionDataset, record: Record) -> Path:
    path = record.image.path
    if path.is_absolute():
        return path
    if dataset.root is not None:
        return Path(dataset.root) / path
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


def read_image_bytes(dataset: VisionDataset, record: Record, *, force_jpeg: bool = False) -> tuple[bytes, str]:
    src = resolve_record_image_path(dataset, record)
    suffix = src.suffix.lower() or ".jpg"
    if not force_jpeg and suffix in IMG_EXTS:
        return src.read_bytes(), suffix
    with Image.open(src) as image:
        rgb = image.convert("RGB")
        buffer = bytes_io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue(), ".jpg"


def ensure_has_vqas(dataset: VisionDataset, *, allow_empty: bool = False) -> None:
    if allow_empty and not dataset.records:
        return
    if any(record.vqas for record in dataset.records):
        return
    raise SystemExit("This output requires at least one question-answer entry in the dataset.")


def parse_model_args(values: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw in values:
        key, sep, value = raw.partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"Invalid --model-arg {raw!r}; expected key=value.")
        parsed: Any
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = value
        out[key] = parsed
    return out


def load_prompt_spec(raw_prompt: str) -> list[tuple[str, str | None]]:
    prompt = str(raw_prompt or "").strip()
    if not prompt:
        raise SystemExit("--prompt is required.")

    path = Path(prompt)
    if path.suffix.lower() in {".yaml", ".yml", ".json"} and path.exists():
        data = read_yaml_or_json(path)
        return normalize_prompt_spec(data)

    try:
        data = json.loads(prompt)
    except Exception:
        return [(prompt, None)]
    return normalize_prompt_spec(data)


def normalize_prompt_spec(data: object) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    if isinstance(data, list):
        for item in data:
            question = str(item).strip()
            if question:
                out.append((question, None))
    elif isinstance(data, dict):
        for raw_label, raw_questions in data.items():
            label = str(raw_label).strip() or None
            if isinstance(raw_questions, list):
                for raw_question in raw_questions:
                    question = str(raw_question).strip()
                    if question:
                        out.append((question, label))
            else:
                question = str(raw_questions).strip()
                if question:
                    out.append((question, label))
    else:
        raise SystemExit("Prompt payload must be a string, a list of strings, or a label-to-questions mapping.")

    deduped: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for question, label in out:
        key = (question, label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((question, label))
    if not deduped:
        raise SystemExit("Prompt payload did not contain any non-empty questions.")
    return deduped


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
    if isinstance(extra_meta, dict):
        extra = dict(extra_meta)
    else:
        extra = {}

    source_value = payload.get("source")
    if source_value is None:
        source_value = extra.pop("source", source)
    if source_value is not None:
        meta["source"] = str(source_value)
    else:
        meta["source"] = source
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


def record_item_meta(record: Record) -> dict[str, Any]:
    raw = record.attributes.get(ITEM_META_ATTR)
    return dict(raw) if isinstance(raw, dict) else {}


def promoted_item_fields(record: Record) -> dict[str, Any]:
    item_meta = record_item_meta(record)
    out: dict[str, Any] = {}
    image_id = record.attributes.get(IMAGE_ID_ATTR)
    if image_id is None and "image_id" in item_meta:
        image_id = item_meta.get("image_id")
    if image_id is not None:
        out["image_id"] = image_id

    predominant_label = record.attributes.get(PREDOMINANT_LABEL_ATTR)
    if predominant_label is None and "predominant_label" in item_meta:
        predominant_label = item_meta.get("predominant_label")
    if predominant_label is not None:
        out["predominant_label"] = predominant_label

    original_filename = item_meta.get("original_filename")
    if original_filename is None and record.rel_image_path is not None:
        original_filename = record.rel_image_path.name
    if original_filename is not None:
        out["original_filename"] = original_filename
    return out


def cleanup_dataset_resources(dataset: VisionDataset | None) -> None:
    if dataset is None:
        return
    raw = dataset.meta.get(CLEANUP_DIRS_META, [])
    if not isinstance(raw, list):
        return
    for item in raw:
        if not item:
            continue
        shutil.rmtree(Path(str(item)), ignore_errors=True)
