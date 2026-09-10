from __future__ import annotations

import io as bytes_io
import json
from pathlib import Path
import tarfile
import tempfile
from typing import Any

from cvsuite.common.core.enums import IMG_EXTS
from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset

from .. import io


def _shard_paths(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    if not src.is_dir():
        raise FileNotFoundError(f"Shard input not found: {src}")
    shards_root = src / "shards" if (src / "shards").is_dir() else src
    shards = sorted(path for path in shards_root.iterdir() if path.is_file() and path.suffix.lower() == ".tar")
    if not shards:
        raise FileNotFoundError(f"No shard archives found under: {shards_root}")
    return shards


def _sample_payload(record: Record, *, sample_key: str) -> dict[str, Any]:
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

    item_meta = io.record_item_meta(record)
    item_meta.pop("image_id", None)
    item_meta.pop("predominant_label", None)
    item_meta.pop("original_filename", None)
    if item_meta:
        payload["meta"] = item_meta
    return payload


class ShardsIO:
    @classmethod
    def matches(cls, src: Path) -> bool:
        if src.is_file():
            return src.suffix.lower() == ".tar"
        if not src.is_dir():
            return False
        shards_root = src / "shards" if (src / "shards").is_dir() else src
        return any(path.is_file() and path.suffix.lower() == ".tar" for path in shards_root.iterdir())

    @classmethod
    def ingest(cls, src: Path) -> VisionDataset:
        shards = _shard_paths(src.resolve())
        temp_dir = Path(tempfile.mkdtemp(prefix="cvsuite-vlm-shards-"))
        records: list[Record] = []

        for shard_path in shards:
            with tarfile.open(shard_path, mode="r") as handle:
                pending_json: dict[str, dict[str, Any]] = {}
                pending_images: dict[str, tuple[str, bytes]] = {}
                for member in handle:
                    if not member.isfile():
                        continue
                    suffix = Path(member.name).suffix.lower()
                    if suffix not in IMG_EXTS and suffix != ".json":
                        continue
                    payload_file = handle.extractfile(member)
                    if payload_file is None:
                        continue
                    raw = payload_file.read()
                    sample_key = Path(member.name).stem
                    if suffix == ".json":
                        payload = json.loads(raw.decode("utf-8"))
                        if not isinstance(payload, dict):
                            raise ValueError(f"Shard sidecar must be a JSON object: {shard_path}:{member.name}")
                        pending_json[sample_key] = payload
                    else:
                        pending_images[sample_key] = (suffix, raw)

                    if sample_key not in pending_json or sample_key not in pending_images:
                        continue

                    image_suffix, image_bytes = pending_images.pop(sample_key)
                    payload = pending_json.pop(sample_key)
                    image_dir = temp_dir / shard_path.stem
                    image_dir.mkdir(parents=True, exist_ok=True)
                    image_rel = Path(shard_path.stem) / f"{sample_key}{image_suffix}"
                    image_abs = temp_dir / image_rel
                    image_abs.write_bytes(image_bytes)
                    image = io.image_record_from_path(image_abs)
                    image.path = image_rel

                    rel_image_value = payload.get("rel_image_path")
                    if rel_image_value is not None:
                        rel_image_path = Path(str(rel_image_value))
                    elif payload.get("original_filename") is not None:
                        rel_image_path = Path(str(payload["original_filename"]))
                    else:
                        rel_image_path = Path(image_rel.name)

                    qas_payload = payload.get("qa_pairs", [])
                    if not isinstance(qas_payload, list):
                        raise ValueError(f"'qa_pairs' must be a list in {shard_path}:{sample_key}.json")
                    vqas = [io.qa_from_payload(dict(qa), source="shards", question_keys=("question", "prompt")) for qa in qas_payload if isinstance(qa, dict)]

                    item_meta = dict(payload.get("meta") or {})
                    item_meta["shard_name"] = shard_path.name
                    original_filename = payload.get("original_filename")
                    if original_filename is not None:
                        item_meta.setdefault("original_filename", original_filename)

                    attributes = {
                        io.SAMPLE_KEY_ATTR: str(payload.get("sample_key") or sample_key),
                        io.ITEM_META_ATTR: item_meta,
                    }
                    if payload.get("image_id") is not None:
                        attributes[io.IMAGE_ID_ATTR] = payload["image_id"]
                    if payload.get("predominant_label") is not None:
                        attributes[io.PREDOMINANT_LABEL_ATTR] = payload["predominant_label"]

                    record = Record(
                        image=image,
                        split=io.normalize_split(payload.get("split")),
                        rel_image_path=rel_image_path,
                        attributes=attributes,
                        vqas=vqas,
                    )
                    records.append(record)

                if pending_json or pending_images:
                    raise ValueError(f"Incomplete sample pairs found in shard: {shard_path}")

        return VisionDataset(
            records=records,
            root=temp_dir,
            meta={"format": "shards", io.CLEANUP_DIRS_META: [str(temp_dir)]},
        )

    @classmethod
    def write(
        cls,
        dataset: VisionDataset,
        dst: Path,
        *,
        target_shard_size_mb: int = 500,
        max_samples_per_shard: int = 10_000,
    ) -> None:
        io.ensure_has_vqas(dataset, allow_empty=True)
        shards_root = dst / "shards"
        shards_root.mkdir(parents=True, exist_ok=True)
        if not dataset.records:
            return
        target_bytes = int(target_shard_size_mb) * 1024 * 1024
        sample_keys = io.unique_sample_keys(dataset.records)

        shard_index = 0
        shard_sample_count = 0
        shard_bytes = 0
        tar_handle: tarfile.TarFile | None = None
        tar_path: Path | None = None

        def open_shard(index: int) -> tuple[tarfile.TarFile, Path]:
            path = shards_root / f"shard-{index:06d}.tar"
            if path.exists():
                raise FileExistsError(f"Destination already exists: {path}")
            return tarfile.open(path, mode="w"), path

        def close_shard(handle: tarfile.TarFile | None) -> None:
            if handle is not None:
                handle.close()

        tar_handle, tar_path = open_shard(shard_index)
        try:
            for record, sample_key in zip(dataset.records, sample_keys):
                image_bytes, _suffix = io.read_image_bytes(dataset, record, force_jpeg=True)
                payload = _sample_payload(record, sample_key=sample_key)
                json_bytes = json.dumps(payload, ensure_ascii=False, indent=None).encode("utf-8")
                estimated_pair_bytes = len(image_bytes) + len(json_bytes)

                if shard_sample_count > 0 and (
                    shard_sample_count >= max_samples_per_shard
                    or shard_bytes + estimated_pair_bytes > target_bytes
                ):
                    close_shard(tar_handle)
                    shard_index += 1
                    shard_sample_count = 0
                    shard_bytes = 0
                    tar_handle, tar_path = open_shard(shard_index)

                assert tar_handle is not None
                assert tar_path is not None
                img_name = f"{sample_key}.jpg"
                json_name = f"{sample_key}.json"
                img_info = tarfile.TarInfo(name=img_name)
                img_info.size = len(image_bytes)
                tar_handle.addfile(img_info, bytes_io.BytesIO(image_bytes))
                json_info = tarfile.TarInfo(name=json_name)
                json_info.size = len(json_bytes)
                tar_handle.addfile(json_info, bytes_io.BytesIO(json_bytes))
                shard_sample_count += 1
                shard_bytes += estimated_pair_bytes
        finally:
            close_shard(tar_handle)
