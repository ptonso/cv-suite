"""Adapter for image-plus-JSON VLM sidecar datasets.

This module expects each sample on disk to be an image paired with a same-stem
JSON sidecar carrying VQA-style fields. In memory it reads and writes canonical
`VisionRecord` objects with `image`, `rel_image_path`, `sample_id`, metadata,
and `vqas` already structured.

Example root:
    dataset/
      sample_001.jpg
      sample_001.json
      nested/
        sample_002.png
        sample_002.json
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from cvsuite.common.core import VisionDataset, VisionRecord
from cvsuite.common.io import common


def _matches_requested_split(actual: str, requested: list[str]) -> bool:
    actual_lower = str(actual or "train").lower()
    requested_lower = [str(item).lower() for item in requested]
    return any(actual_lower == token or actual_lower.startswith(token) for token in requested_lower)


def _image_candidates_by_stem(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in common.IMG_EXTS:
            out.setdefault(path.stem, []).append(path)
    return out


def _resolve_sidecar_image(sidecar: Path, root: Path, image_map: dict[str, list[Path]]) -> Path | None:
    """Pick the image that belongs to `sidecar`.

    Prefers a same-stem image in the sidecar's own directory (so a
    bucket-partitioned export -- e.g. `map-answers ... to-json`, one
    sub-directory per answer bucket -- stays re-ingestable even when the same
    stem lands in several buckets). Falls back to a tree-wide match, and for a
    parallel `labels*/ <-> images/` split layout picks the mirror-path image.
    """
    same_dir = [p for p in sidecar.parent.iterdir() if p.is_file() and p.stem == sidecar.stem and p.suffix.lower() in common.IMG_EXTS]
    if len(same_dir) == 1:
        return same_dir[0]
    if len(same_dir) > 1:
        joined = ", ".join(sorted(str(p.relative_to(root)) for p in same_dir))
        raise ValueError(f"Multiple images found for {sidecar}: {joined}")

    candidates = image_map.get(sidecar.stem, [])
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    rel_parts = sidecar.relative_to(root).parent.parts
    mirror = tuple("images" if part == "labels" or part.startswith("labels_") or part.startswith("labels-") else part for part in rel_parts)
    mirrored = [p for p in candidates if p.relative_to(root).parent.parts == mirror]
    if len(mirrored) == 1:
        return mirrored[0]
    joined = ", ".join(sorted(str(p.relative_to(root)) for p in candidates))
    raise ValueError(f"Multiple images found for {sidecar}: {joined}")


def _json_sidecars(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*.json")) if path.is_file()]


def _sidecar_payload(record: VisionRecord, *, sample_key: str, image_filename: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": "1",
        "task": "vlm_sample",
        "sample_key": sample_key,
        "split": record.split or "train",
        "qa_pairs": [common.qa_to_payload(qa, question_key="question", id_key="question_id") for qa in record.vqas],
    }
    rel = record.rel_image_path
    if rel is not None and not rel.is_absolute():
        payload["rel_image_path"] = rel.as_posix()

    promoted = common.promoted_item_fields(record)
    if promoted.get("image_id") is not None:
        payload["image_id"] = promoted["image_id"]
    if promoted.get("predominant_label") is not None:
        payload["predominant_label"] = promoted["predominant_label"]
    if promoted.get("original_filename") is not None:
        payload["original_filename"] = promoted["original_filename"]
    elif image_filename:
        payload["original_filename"] = image_filename

    item_meta = common.record_item_meta(record)
    item_meta.pop("image_id", None)
    item_meta.pop("predominant_label", None)
    item_meta.pop("original_filename", None)
    if item_meta:
        payload["meta"] = item_meta
    return payload


class FlatVlmJsonAdapter:
    @classmethod
    def matches(cls, src: Path) -> bool:
        if not src.is_dir():
            return False
        root = src.resolve()
        image_map = _image_candidates_by_stem(root)
        for sidecar in _json_sidecars(root):
            try:
                if _resolve_sidecar_image(sidecar, root, image_map) is not None:
                    return True
            except ValueError:
                return True
        return False

    @classmethod
    def read(cls, src: Path, *, task: str = "auto", splits: list[str] | None = None, **options: object) -> VisionDataset:
        root = src.resolve()
        if not root.is_dir():
            raise ValueError(f"Flat JSON input must be a directory: {root}")

        image_map = _image_candidates_by_stem(root)
        records: list[VisionRecord] = []
        for sidecar in _json_sidecars(root):
            image_match = _resolve_sidecar_image(sidecar, root, image_map)
            if image_match is None:
                continue
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Sample sidecar must be a JSON object: {sidecar}")

            image_path = image_match.resolve()
            image = common.image_info_from_path(image_path)
            image_rel_path = image_path.relative_to(root)
            image.path = image_rel_path
            rel_image = payload.get("rel_image_path")
            rel_path = Path(str(rel_image)) if rel_image else image_rel_path

            qas_payload = payload.get("qa_pairs", payload.get("qas", []))
            if qas_payload is None:
                qas_payload = []
            if not isinstance(qas_payload, list):
                raise ValueError(f"'qa_pairs' must be a list in {sidecar}")

            vqas = [
                common.qa_from_payload(dict(qa), source="json", question_keys=("question", "prompt"))
                for qa in qas_payload
                if isinstance(qa, dict)
            ]

            item_meta = dict(payload.get("meta") or {})
            original_filename = payload.get("original_filename")
            if original_filename is not None:
                item_meta.setdefault("original_filename", original_filename)

            sample_key = str(payload.get("sample_key") or common.derive_sample_key_from_path(rel_path))
            attributes = {
                common.SAMPLE_KEY_ATTR: sample_key,
                common.ITEM_META_ATTR: item_meta,
            }
            if payload.get("image_id") is not None:
                attributes[common.IMAGE_ID_ATTR] = payload["image_id"]
            if payload.get("predominant_label") is not None:
                attributes[common.PREDOMINANT_LABEL_ATTR] = payload["predominant_label"]

            records.append(
                VisionRecord(
                    sample_id=sample_key,
                    image=image,
                    split=str(payload.get("split") or "train").strip() or "train",
                    rel_image_path=rel_path,
                    attributes=attributes,
                    vqas=vqas,
                )
            )

        dataset = VisionDataset(records=records, root=root)
        if splits:
            dataset.records = [record for record in dataset.records if _matches_requested_split(record.split or "train", splits)]
        return dataset

    @classmethod
    def write(
        cls,
        dataset: VisionDataset,
        dst: Path,
        *,
        splits: list[str] | None = None,
        hardlink: bool = False,
        skip_images: bool = False,
        **options: object,
    ) -> None:
        filtered = dataset.records
        if splits:
            filtered = [record for record in dataset.records if _matches_requested_split(record.split or "train", splits)]
        common.ensure_has_vqas(VisionDataset(records=filtered, root=dataset.root), allow_empty=True)
        dst.mkdir(parents=True, exist_ok=True)
        if not filtered:
            return

        sample_keys = common.unique_sample_keys(filtered)
        # Counted over the records written into *this* directory: a partitioned
        # export writes one bucket per call, so a stem shared across buckets is
        # still unique here and keeps its natural filename. Read is dir-scoped,
        # so the same stem living in several bucket dirs no longer collides.
        stem_counts = Counter(common.record_relative_image_path(record).stem for record in filtered)

        for record, sample_key in zip(filtered, sample_keys):
            src = common.resolve_record_image_path(dataset, record)
            rel = common.record_relative_image_path(record)
            preferred_stem = rel.stem
            preferred_name = rel.name
            if not preferred_name or stem_counts[preferred_stem] != 1:
                preferred_name = f"{sample_key}{src.suffix.lower() or '.jpg'}"

            img_dst = dst / preferred_name
            json_dst = dst / f"{Path(preferred_name).stem}.json"
            if json_dst.exists():
                raise FileExistsError(f"Destination already exists: {json_dst}")
            if not skip_images:
                common.copy_or_link(src, img_dst, hardlink=hardlink)
            payload = _sidecar_payload(record, sample_key=sample_key, image_filename=preferred_name)
            json_dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
