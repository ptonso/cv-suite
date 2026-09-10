"""Canonical dataset containers: `VisionRecord` and `VisionDataset`.

One `VisionRecord` is one image-centric sample. `VisionDataset` is the record list
plus dataset-wide provenance. JSON round-trip (`to_json` / `from_json`) is used for
FM subprocess payloads and checkpoint state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from cvsuite.common.core.enums import Task
from cvsuite.common.core.types import (
    BBox,
    Classification,
    Embedding,
    FMRequest,
    FormatId,
    ImageInfo,
    Keypoints,
    LabelMeShape,
    Polygon,
    SemanticMaskRef,
    SplitId,
    TaskId,
    VQA,
)
from cvsuite.common.core.utils import _decode_dataclass, _encode


@dataclass
class VisionRecord:
    sample_id: str | None = None
    image: ImageInfo | None = None
    split: SplitId = "train"
    task: TaskId | None = None

    boxes: list[BBox] = field(default_factory=list)
    polys: list[Polygon] = field(default_factory=list)
    kpts: list[Keypoints] = field(default_factory=list)

    classification: Classification | None = None
    vqas: list[VQA] = field(default_factory=list)
    semantic_mask: SemanticMaskRef | None = None

    rel_image_path: Path | None = None
    rel_label_path: Path | None = None
    source_format: FormatId | None = None

    round_trip: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    # cvsuite runtime overlay
    embeddings: list[Embedding] = field(default_factory=list)
    labelme_shapes: list[LabelMeShape] = field(default_factory=list)


@dataclass
class VisionDataset:
    records: list[VisionRecord]
    classes: list[str] = field(default_factory=list)
    task: TaskId | None = None
    root: Path | None = None
    source_format: FormatId | None = None

    round_trip: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # cvsuite runtime overlay
    fm_request: FMRequest | None = None

    def by_split(self) -> dict[str, list[VisionRecord]]:
        out: dict[str, list[VisionRecord]] = {}
        for record in self.records:
            out.setdefault(record.split or "train", []).append(record)
        return out

    def resized(self, size: int | tuple[int, int]) -> "VisionDataset":
        """Return a copy with image dimensions overwritten. Normalized geometry is left untouched."""
        width, height = (size, size) if isinstance(size, int) else size
        records: list[VisionRecord] = []
        for record in self.records:
            if record.image is None:
                records.append(record)
                continue
            image = ImageInfo(path=record.image.path, width=width, height=height, source=record.image.source)
            attributes = {**record.attributes, "resize_to": (width, height)}
            records.append(replace(record, image=image, attributes=attributes))
        meta = {**self.meta, "resize_to": (width, height)}
        return normalize_dataset(replace(self, records=records, meta=meta))

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_encode(self), ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: Path) -> "VisionDataset":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return normalize_dataset(_decode_dataclass(cls, raw))


Record = VisionRecord


def _coerce_task(value: Any) -> Any:
    if value is None or isinstance(value, Task):
        return value
    try:
        return Task.from_raw(value)
    except ValueError:
        return value


def normalize_dataset(dataset: VisionDataset) -> VisionDataset:
    """Coerce dataset/record task hints onto the canonical `Task` vocabulary in place."""
    dataset.task = _coerce_task(dataset.task)
    for record in dataset.records:
        record.task = _coerce_task(record.task)
    return dataset


def copy_dataset(dataset: VisionDataset, **changes: Any) -> VisionDataset:
    return normalize_dataset(replace(dataset, **changes))


def copy_record(record: VisionRecord, **changes: Any) -> VisionRecord:
    updated = replace(record, **changes)
    updated.task = _coerce_task(updated.task)
    return updated


__all__ = [
    "Record",
    "VisionDataset",
    "VisionRecord",
    "copy_dataset",
    "copy_record",
    "normalize_dataset",
]
