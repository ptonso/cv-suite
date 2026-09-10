"""Reader for loose image collections.

This adapter expects one image file or a directory tree of images on disk and
produces canonical `VisionRecord` items with image metadata only. In memory it
works with unlabeled records whose identity comes from relative image paths and
derived sample keys.

Example root:
    dataset/
      a.jpg
      nested/
        b.png
"""

from __future__ import annotations

from pathlib import Path

from cvsuite.common.core import VisionDataset, VisionRecord
from cvsuite.common.io import common


class FlatAdapter:
    @classmethod
    def matches(cls, src: Path) -> bool:
        if src.is_file():
            return src.suffix.lower() in common.IMG_EXTS
        if not src.is_dir():
            return False
        return any(path.is_file() and path.suffix.lower() in common.IMG_EXTS for path in src.rglob("*"))

    @classmethod
    def read(cls, src: Path, *, task: str = "auto", splits: list[str] | None = None, **options: object) -> VisionDataset:
        max_depth = int(options.get("max_depth", -1))
        root = src.resolve().parent if src.is_file() else src.resolve()
        image_paths = (
            [src.resolve()]
            if src.is_file()
            else [path.resolve() for path in common.iter_files(src, max_depth=max_depth) if path.suffix.lower() in common.IMG_EXTS]
        )
        records: list[VisionRecord] = []
        for image_path in image_paths:
            image = common.image_info_from_path(image_path)
            rel_path = common.root_relative_path(root, image_path)
            image.path = rel_path
            sample_key = common.derive_sample_key_from_path(rel_path)
            records.append(
                VisionRecord(
                    sample_id=sample_key,
                    image=image,
                    split="train",
                    rel_image_path=rel_path,
                    attributes={common.SAMPLE_KEY_ATTR: sample_key},
                )
            )
        dataset = VisionDataset(records=records, root=root)
        if splits:
            allowed = {common.canonical_split_id(item) for item in splits}
            dataset.records = [record for record in dataset.records if record.split in allowed]
        return dataset
