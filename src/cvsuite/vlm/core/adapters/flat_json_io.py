from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from cvsuite.common.core.enums import IMG_EXTS
from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset

from .. import io


def _image_candidates_by_stem(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            out.setdefault(path.stem, []).append(path)
    return out


def _json_sidecars(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*.json")) if path.is_file()]


def _sidecar_payload(record: Record, *, sample_key: str, image_filename: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": "1",
        "task": "vlm_sample",
        "sample_key": sample_key,
        "split": record.split or "train",
        "qa_pairs": [io.qa_to_payload(qa, question_key="question", id_key="question_id") for qa in record.vqas],
    }
    rel = record.rel_image_path
    if rel is not None and not rel.is_absolute():
        payload["rel_image_path"] = rel.as_posix()

    promoted = io.promoted_item_fields(record)
    if promoted.get("image_id") is not None:
        payload["image_id"] = promoted["image_id"]
    if promoted.get("predominant_label") is not None:
        payload["predominant_label"] = promoted["predominant_label"]
    if promoted.get("original_filename") is not None:
        payload["original_filename"] = promoted["original_filename"]
    elif image_filename:
        payload["original_filename"] = image_filename

    item_meta = io.record_item_meta(record)
    item_meta.pop("image_id", None)
    item_meta.pop("predominant_label", None)
    item_meta.pop("original_filename", None)
    if item_meta:
        payload["meta"] = item_meta
    return payload


class FlatJsonIO:
    @classmethod
    def matches(cls, src: Path) -> bool:
        if not src.is_dir():
            return False
        image_map = _image_candidates_by_stem(src)
        for path in _json_sidecars(src):
            if path.stem in image_map:
                return True
        return False

    @classmethod
    def ingest(cls, src: Path) -> VisionDataset:
        src = src.resolve()
        if not src.is_dir():
            raise ValueError(f"Flat JSON input must be a directory: {src}")

        image_map = _image_candidates_by_stem(src)
        records: list[Record] = []
        for sidecar in _json_sidecars(src):
            images = image_map.get(sidecar.stem, [])
            if not images:
                continue
            if len(images) > 1:
                raise ValueError(f"Multiple images found for {sidecar}: {', '.join(str(path.relative_to(src)) for path in images)}")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Sample sidecar must be a JSON object: {sidecar}")
            image_path = images[0]
            image = io.image_record_from_path(image_path)
            image_rel_path = image_path.relative_to(src)
            image.path = image_rel_path
            rel_image = payload.get("rel_image_path")
            rel_path = Path(str(rel_image)) if rel_image else image_rel_path

            qas_payload = payload.get("qa_pairs", payload.get("qas", []))
            if qas_payload is None:
                qas_payload = []
            if not isinstance(qas_payload, list):
                raise ValueError(f"'qa_pairs' must be a list in {sidecar}")

            vqas = [io.qa_from_payload(dict(qa), source="json", question_keys=("question", "prompt")) for qa in qas_payload if isinstance(qa, dict)]

            item_meta = dict(payload.get("meta") or {})
            original_filename = payload.get("original_filename")
            if original_filename is not None:
                item_meta.setdefault("original_filename", original_filename)

            attributes = {
                io.SAMPLE_KEY_ATTR: str(payload.get("sample_key") or io.derive_sample_key_from_path(rel_path)),
                io.ITEM_META_ATTR: item_meta,
            }
            if payload.get("image_id") is not None:
                attributes[io.IMAGE_ID_ATTR] = payload["image_id"]
            if payload.get("predominant_label") is not None:
                attributes[io.PREDOMINANT_LABEL_ATTR] = payload["predominant_label"]

            record = Record(
                image=image,
                split=io.normalize_split(payload.get("split")),
                rel_image_path=rel_path,
                attributes=attributes,
                vqas=vqas,
            )
            records.append(record)

        return VisionDataset(records=records, root=src, meta={"format": "json"})

    @classmethod
    def write(cls, dataset: VisionDataset, dst: Path, *, hardlink: bool = False, skip_images: bool = False) -> None:
        io.ensure_has_vqas(dataset, allow_empty=True)
        dst.mkdir(parents=True, exist_ok=True)
        if not dataset.records:
            return

        sample_keys = io.unique_sample_keys(dataset.records)
        stem_counts = Counter(io.record_relative_image_path(record).stem for record in dataset.records)

        for record, sample_key in zip(dataset.records, sample_keys):
            src = io.resolve_record_image_path(dataset, record)
            rel = io.record_relative_image_path(record)
            preferred_stem = rel.stem
            preferred_name = rel.name
            if not preferred_name or stem_counts[preferred_stem] != 1:
                preferred_name = f"{sample_key}{src.suffix.lower() or '.jpg'}"

            img_dst = dst / preferred_name
            json_dst = dst / f"{Path(preferred_name).stem}.json"
            if json_dst.exists():
                raise FileExistsError(f"Destination already exists: {json_dst}")
            if not skip_images:
                if img_dst.exists():
                    raise FileExistsError(f"Destination already exists: {img_dst}")
                io.copy_or_link(src, img_dst, hardlink=hardlink)
            payload = _sidecar_payload(record, sample_key=sample_key, image_filename=preferred_name)
            json_dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
