from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

from cvsuite.common.core.enums import IMG_EXTS
from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset

from .. import io

_IMAGE_ID_RE = re.compile(r"(\d+)$")


def _json_candidates(src: Path) -> list[Path]:
    src = src.resolve()
    if src.is_file():
        if src.suffix.lower() != ".json":
            raise ValueError(f"Expected a JSON manifest, got: {src}")
        return [src]
    if not src.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {src}")
    return [path for path in sorted(src.rglob("*.json")) if path.is_file()]


def _load_json_payloads(src: Path) -> list[tuple[Path, Any]]:
    out: list[tuple[Path, Any]] = []
    for path in _json_candidates(src):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append((path, payload))
    return out


def _is_normalized_manifest(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("items"), list)


def _question_entries(payload: Any) -> list[dict[str, Any]] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
        return None
    entries = [item for item in payload["questions"] if isinstance(item, dict)]
    if not entries:
        return []
    if not any(item.get("question") is not None and item.get("question_id") is not None and item.get("image_id") is not None for item in entries):
        return None
    return entries


def _annotation_entries(payload: Any) -> list[dict[str, Any]] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("annotations"), list):
        return None
    entries = [item for item in payload["annotations"] if isinstance(item, dict)]
    if not entries:
        return []
    if not any(item.get("question_id") is not None and item.get("image_id") is not None for item in entries):
        return None
    return entries


def _prediction_entries(payload: Any) -> list[dict[str, Any]] | None:
    candidates: Any = None
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        for key in ("predictions", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = value
                break
    if not isinstance(candidates, list):
        return None
    entries = [item for item in candidates if isinstance(item, dict)]
    if not entries:
        return []
    if not any(item.get("question_id") is not None and item.get("answer") is not None for item in entries):
        return None
    return entries


def _discover_normalized_manifests(src: Path) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for path, payload in _load_json_payloads(src):
        if _is_normalized_manifest(payload):
            out.append((path, payload))
    return out


def _discover_raw_vqa_jsons(src: Path) -> tuple[list[tuple[Path, dict[str, Any], list[dict[str, Any]]]], list[tuple[Path, dict[str, Any], list[dict[str, Any]]]], list[tuple[Path, Any, list[dict[str, Any]]]]]:
    questions: list[tuple[Path, dict[str, Any], list[dict[str, Any]]]] = []
    annotations: list[tuple[Path, dict[str, Any], list[dict[str, Any]]]] = []
    predictions: list[tuple[Path, Any, list[dict[str, Any]]]] = []
    for path, payload in _load_json_payloads(src):
        question_entries = _question_entries(payload)
        if question_entries is not None:
            questions.append((path, payload, question_entries))
            continue
        annotation_entries = _annotation_entries(payload)
        if annotation_entries is not None:
            annotations.append((path, payload, annotation_entries))
            continue
        prediction_entries = _prediction_entries(payload)
        if prediction_entries is not None:
            predictions.append((path, payload, prediction_entries))
    return questions, annotations, predictions


def _resolve_manifest(src: Path) -> tuple[Path, dict[str, Any]]:
    manifests = _discover_normalized_manifests(src)
    if not manifests:
        raise FileNotFoundError(f"No VLM manifest found under {src}. Expected one of: {', '.join(io.DEFAULT_MANIFEST_NAMES)}")

    src_path = src.resolve()
    preferred_names = {src_path / name for name in io.DEFAULT_MANIFEST_NAMES} if src_path.is_dir() else {src_path}
    preferred = [item for item in manifests if item[0] in preferred_names]
    if len(preferred) == 1:
        return preferred[0]
    if len(manifests) == 1:
        return manifests[0]
    raise RuntimeError(f"Multiple VLM manifests found under {src}: {', '.join(str(path) for path, _payload in manifests)}")


def _coerce_int(value: Any, *, field: str, path: Path) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Expected integer {field!r} in {path}, got: {value!r}") from exc


def _extract_image_id(path: Path) -> int | None:
    match = _IMAGE_ID_RE.search(path.stem)
    if match is not None:
        return int(match.group(1))
    digit_groups = re.findall(r"\d+", path.stem)
    if digit_groups:
        return int(digit_groups[-1])
    return None


def _discover_images_by_id(root: Path) -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        image_id = _extract_image_id(path)
        if image_id is None:
            continue
        out[image_id].append(path)
    return out


def _select_image_path(image_id: int, images_by_id: dict[int, list[Path]], *, data_subtype: str | None, root: Path) -> Path:
    candidates = images_by_id.get(image_id, [])
    if not candidates:
        raise FileNotFoundError(f"No image found for image_id={image_id} under {root}")
    if len(candidates) == 1:
        return candidates[0]
    if data_subtype:
        filtered = [path for path in candidates if data_subtype in path.as_posix()]
        if len(filtered) == 1:
            return filtered[0]
    raise ValueError(f"Multiple images found for image_id={image_id}: {', '.join(str(path.relative_to(root)) for path in candidates)}")


def _question_payload_from_raw(
    question_item: dict[str, Any],
    *,
    question_path: Path,
    annotation_item: dict[str, Any] | None,
    prediction_item: dict[str, Any] | None,
) -> dict[str, Any]:
    if question_item.get("question") is None:
        raise ValueError(f"Question entry is missing 'question' in {question_path}")

    payload: dict[str, Any] = {
        "question": str(question_item["question"]),
        "question_id": question_item.get("question_id"),
    }
    meta: dict[str, Any] = {}

    question_meta = {key: value for key, value in question_item.items() if key not in {"question", "question_id", "image_id"}}
    meta.update(question_meta)

    if annotation_item is not None:
        if annotation_item.get("multiple_choice_answer") is not None:
            payload["ground_truth"] = annotation_item["multiple_choice_answer"]
        answers = annotation_item.get("answers")
        if isinstance(answers, list):
            expected_answers = [item.get("answer") for item in answers if isinstance(item, dict) and item.get("answer") is not None]
            if expected_answers:
                payload["answers"] = expected_answers
        annotation_meta = {
            key: value
            for key, value in annotation_item.items()
            if key not in {"answers", "multiple_choice_answer", "question_id", "image_id"}
        }
        meta.update(annotation_meta)

    if prediction_item is not None:
        if prediction_item.get("answer") is not None:
            payload["answer"] = str(prediction_item["answer"])
        if prediction_item.get("model") is not None:
            payload["model"] = str(prediction_item["model"])
        if prediction_item.get("score") is not None:
            payload["score"] = prediction_item["score"]
        prediction_meta = {
            key: value
            for key, value in prediction_item.items()
            if key not in {"answer", "model", "score", "question_id", "image_id"}
        }
        meta.update(prediction_meta)

    if meta:
        payload["meta"] = meta
    return payload


def _ingest_raw_vqa(src: Path) -> VisionDataset:
    root = src.resolve() if src.is_dir() else src.resolve().parent
    question_docs, annotation_docs, prediction_docs = _discover_raw_vqa_jsons(src)
    if not question_docs:
        raise ValueError(f"No VQA questions JSON found under {src}")
    if not any(entries for _path, _payload, entries in question_docs):
        raise ValueError(f"No VQA questions found inside question JSONs under {src}")

    images_by_id = _discover_images_by_id(root)
    if not images_by_id:
        raise FileNotFoundError(f"No images with VQA-style image ids found under {root}")

    annotations_by_qid: dict[int, dict[str, Any]] = {}
    for ann_path, _payload, entries in annotation_docs:
        for entry in entries:
            question_id = entry.get("question_id")
            if question_id is None:
                continue
            annotations_by_qid[_coerce_int(question_id, field="question_id", path=ann_path)] = entry

    predictions_by_qid: dict[int, dict[str, Any]] = {}
    for pred_path, _payload, entries in prediction_docs:
        for entry in entries:
            question_id = entry.get("question_id")
            if question_id is None:
                continue
            predictions_by_qid[_coerce_int(question_id, field="question_id", path=pred_path)] = entry

    records_by_image_id: dict[int, Record] = {}
    seen_questions: set[int] = set()
    records: list[Record] = []
    source_jsons = [path for path, _payload, _entries in question_docs]
    source_jsons.extend(path for path, _payload, _entries in annotation_docs)
    source_jsons.extend(path for path, _payload, _entries in prediction_docs)

    for question_path, question_payload, entries in question_docs:
        data_subtype = question_payload.get("data_subtype")
        data_type = question_payload.get("data_type")
        split = io.normalize_split(data_subtype or question_payload.get("split"))

        for entry in entries:
            question_id = entry.get("question_id")
            image_id = entry.get("image_id")
            if question_id is None or image_id is None:
                continue
            qid = _coerce_int(question_id, field="question_id", path=question_path)
            if qid in seen_questions:
                continue
            seen_questions.add(qid)
            image_id_value = _coerce_int(image_id, field="image_id", path=question_path)

            record = records_by_image_id.get(image_id_value)
            if record is None:
                image_abs = _select_image_path(image_id_value, images_by_id, data_subtype=str(data_subtype) if data_subtype is not None else None, root=root)
                image_rel = image_abs.relative_to(root) if image_abs.is_relative_to(root) else Path(image_abs.name)
                image = io.image_record_from_path(image_abs)
                image.path = image_rel

                item_meta: dict[str, Any] = {}
                if data_type is not None:
                    item_meta["data_type"] = data_type
                if data_subtype is not None:
                    item_meta["data_subtype"] = data_subtype
                attributes = {
                    io.SAMPLE_KEY_ATTR: io.derive_sample_key_from_path(image_rel),
                    io.ITEM_META_ATTR: item_meta,
                    io.IMAGE_ID_ATTR: image_id_value,
                }
                record = Record(
                    image=image,
                    split=split,
                    rel_image_path=image_rel,
                    attributes=attributes,
                    vqas=[],
                )
                records_by_image_id[image_id_value] = record
                records.append(record)

            annotation_item = annotations_by_qid.get(qid)
            prediction_item = predictions_by_qid.get(qid)
            qa_payload = _question_payload_from_raw(
                entry,
                question_path=question_path,
                annotation_item=annotation_item,
                prediction_item=prediction_item,
            )
            record.vqas.append(io.qa_from_payload(qa_payload, source="vqa-style", question_keys=("question", "prompt")))

    if not records:
        raise ValueError(f"No VQA records could be constructed from question JSONs under {src}")

    return VisionDataset(
        records=records,
        root=root,
        meta={
            "format": "vqa-style",
            "source": "raw-vqa",
            "question_jsons": [str(path.relative_to(root)) for path, _payload, _entries in question_docs],
            "annotation_jsons": [str(path.relative_to(root)) for path, _payload, _entries in annotation_docs],
            "prediction_jsons": [str(path.relative_to(root)) for path, _payload, _entries in prediction_docs],
            "source_jsons": [str(path.relative_to(root)) for path in source_jsons],
        },
    )


class VQAStyleIO:
    @classmethod
    def matches(cls, src: Path) -> bool:
        manifests = _discover_normalized_manifests(src)
        if manifests:
            return True
        question_docs, _annotation_docs, _prediction_docs = _discover_raw_vqa_jsons(src)
        return bool(question_docs)

    @classmethod
    def ingest(cls, src: Path) -> VisionDataset:
        manifests = _discover_normalized_manifests(src)
        question_docs, _annotation_docs, _prediction_docs = _discover_raw_vqa_jsons(src)
        if question_docs:
            if manifests:
                raise RuntimeError(f"Ambiguous vqa-style input under {src}: found both normalized manifests and raw VQA question files.")
            return _ingest_raw_vqa(src)

        manifest_path, manifest = _resolve_manifest(src)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("items"), list):
            raise ValueError(f"Manifest must be a JSON object with an 'items' list: {manifest_path}")

        images_root_value = manifest.get("images_root", "images")
        images_root = Path(str(images_root_value))
        if not images_root.is_absolute():
            images_root = (manifest_path.parent / images_root).resolve()

        records: list[Record] = []
        for item_idx, item in enumerate(manifest["items"]):
            if not isinstance(item, dict):
                raise ValueError(f"Manifest item {item_idx} must be an object.")
            raw_image = item.get("image")
            if raw_image is None:
                raise ValueError(f"Manifest item {item_idx} is missing 'image'.")
            image_rel = Path(str(raw_image))
            image_abs = image_rel if image_rel.is_absolute() else images_root / image_rel
            if not image_abs.exists():
                raise FileNotFoundError(f"Image referenced by manifest does not exist: {image_abs}")
            image = io.image_record_from_path(image_abs)
            image.path = image_rel

            raw_qas = item.get("qas", [])
            if not isinstance(raw_qas, list):
                raise ValueError(f"Manifest item {item_idx} 'qas' must be a list.")
            vqas = [io.qa_from_payload(dict(qa), source="vqa-style", question_keys=("prompt", "question")) for qa in raw_qas if isinstance(qa, dict)]

            item_meta = dict(item.get("meta") or {})
            attributes = {
                io.SAMPLE_KEY_ATTR: io.derive_sample_key_from_path(image_rel),
                io.ITEM_META_ATTR: item_meta,
            }
            if item_meta.get("image_id") is not None:
                attributes[io.IMAGE_ID_ATTR] = item_meta["image_id"]
            if item_meta.get("predominant_label") is not None:
                attributes[io.PREDOMINANT_LABEL_ATTR] = item_meta["predominant_label"]

            record = Record(
                image=image,
                split=io.normalize_split(item.get("split")),
                rel_image_path=image_rel,
                attributes=attributes,
                vqas=vqas,
            )
            records.append(record)

        return VisionDataset(
            records=records,
            root=images_root,
            meta={
                "format": "vqa-style",
                "manifest_path": str(manifest_path),
                "images_root": str(images_root_value),
                "version": manifest.get("version", "1"),
            },
        )

    @classmethod
    def write(cls, dataset: VisionDataset, dst: Path, *, hardlink: bool = False, skip_images: bool = False) -> None:
        io.ensure_has_vqas(dataset, allow_empty=True)
        images_root = dst / "images"
        dst.mkdir(parents=True, exist_ok=True)
        manifest_path = dst / "vqa.json"
        if manifest_path.exists():
            raise FileExistsError(f"Destination already exists: {manifest_path}")
        if not skip_images:
            images_root.mkdir(parents=True, exist_ok=True)

        sample_keys = io.unique_sample_keys(dataset.records)
        rel_paths: list[Path] = []
        seen: set[str] = set()
        for record, sample_key in zip(dataset.records, sample_keys):
            src = io.resolve_record_image_path(dataset, record)
            rel = record.rel_image_path if record.rel_image_path is not None and not record.rel_image_path.is_absolute() else None
            candidate = rel if rel is not None else Path(record.split or "train") / f"{sample_key}{src.suffix.lower() or '.jpg'}"
            key = candidate.as_posix()
            if key in seen:
                candidate = Path(record.split or "train") / f"{sample_key}{src.suffix.lower() or '.jpg'}"
                key = candidate.as_posix()
            if key in seen:
                raise FileExistsError(f"Conflicting output image path: {candidate}")
            seen.add(key)
            rel_paths.append(candidate)

        items: list[dict[str, Any]] = []
        for record, rel_path in zip(dataset.records, rel_paths):
            src = io.resolve_record_image_path(dataset, record)
            if not skip_images:
                io.copy_or_link(src, images_root / rel_path, hardlink=hardlink)

            qas = [io.qa_to_payload(qa, question_key="prompt") for qa in record.vqas]
            item: dict[str, Any] = {
                "image": rel_path.as_posix(),
                "split": record.split or "train",
                "qas": qas,
            }
            item_meta = io.record_item_meta(record)
            image_id = record.attributes.get(io.IMAGE_ID_ATTR)
            if image_id is not None and "image_id" not in item_meta:
                item_meta["image_id"] = image_id
            predominant_label = record.attributes.get(io.PREDOMINANT_LABEL_ATTR)
            if predominant_label is not None and "predominant_label" not in item_meta:
                item_meta["predominant_label"] = predominant_label
            if item_meta:
                item["meta"] = item_meta
            items.append(item)

        payload = {
            "version": "1",
            "task": "vlm",
            "images_root": "images",
            "items": items,
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
