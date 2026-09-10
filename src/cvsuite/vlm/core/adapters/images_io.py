from __future__ import annotations

from pathlib import Path

from cvsuite.common.core.enums import IMG_EXTS
from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset

from .. import io


class ImagesIO:
    @classmethod
    def matches(cls, src: Path) -> bool:
        if src.is_file():
            return src.suffix.lower() in IMG_EXTS
        if not src.is_dir():
            return False
        return any(path.is_file() and path.suffix.lower() in IMG_EXTS for path in src.rglob("*"))

    @classmethod
    def ingest(cls, src: Path) -> VisionDataset:
        src = src.resolve()
        root = src.parent if src.is_file() else src
        image_paths = [src] if src.is_file() else [path for path in sorted(src.rglob("*")) if path.is_file() and path.suffix.lower() in IMG_EXTS]
        records: list[Record] = []
        for image_path in image_paths:
            image = io.image_record_from_path(image_path)
            rel_path = image_path.relative_to(root) if image_path.is_relative_to(root) else Path(image_path.name)
            image.path = rel_path
            record = Record(
                image=image,
                split="train",
                rel_image_path=rel_path,
                attributes={io.SAMPLE_KEY_ATTR: io.derive_sample_key_from_path(rel_path)},
            )
            records.append(record)
        return VisionDataset(records=records, root=root, meta={"format": "images"})
