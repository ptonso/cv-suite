"""Adapter for semantic-segmentation image/mask manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cvsuite.common.core import SemanticMaskRef, VisionDataset, VisionRecord
from cvsuite.common.io import common

_MANIFEST_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
_SPLIT_KEYS = ("train", "val", "valid", "test", "infer")


def _resolve_manifest(src: Path) -> Path:
    if src.is_file() and src.suffix.lower() in {".yaml", ".yml"}:
        return src.resolve()
    if src.is_dir():
        for name in _MANIFEST_NAMES:
            candidate = src / name
            if candidate.exists():
                return candidate.resolve()
        hit = next((path for path in src.rglob("data.yaml") if path.is_file()), None)
        if hit is not None:
            return hit.resolve()
    raise FileNotFoundError(str(src))


def _looks_like_semseg_spec(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in _SPLIT_KEYS:
        value = payload.get(key)
        if isinstance(value, dict) and {"images", "masks"}.issubset(value):
            return True
    return False


def _resolve_dir(root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _resolve_mask_path(mask_dir: Path, image_dir: Path, image_path: Path) -> Path | None:
    rel = image_path.relative_to(image_dir)
    for suffix in (".png", ".pgm", ".tif", ".tiff"):
        candidate = (mask_dir / rel).with_suffix(suffix)
        if candidate.exists():
            return candidate
    for suffix in (".png", ".pgm", ".tif", ".tiff"):
        candidate = mask_dir / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


class SemSegMaskAdapter:
    @classmethod
    def matches(cls, src: Path) -> bool:
        try:
            manifest = _resolve_manifest(src)
        except FileNotFoundError:
            return False
        return _looks_like_semseg_spec(common.read_yaml(manifest) or {})

    @classmethod
    def read(cls, src: Path, *, task: str = "auto", splits: list[str] | None = None, **options: Any) -> VisionDataset:
        del task, options
        manifest = _resolve_manifest(src)
        payload = common.read_yaml(manifest) or {}
        if not _looks_like_semseg_spec(payload):
            raise ValueError(f"Semantic-segmentation manifest must define split images/masks mappings: {manifest}")

        root = manifest.parent
        requested = [common.canonical_split_id(item) for item in (splits or ["train", "val", "test"])]
        records: list[VisionRecord] = []

        for split in requested:
            raw = payload.get(split)
            if raw is None and split == "val":
                raw = payload.get("valid")
            if not isinstance(raw, dict):
                continue
            image_dir = _resolve_dir(root, raw.get("images"))
            mask_dir = _resolve_dir(root, raw.get("masks"))
            for image_path in sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in common.IMG_EXTS):
                image = common.image_info_from_path(image_path)
                rel_img = common.root_relative_path(root, image_path)
                image.path = rel_img
                mask_path = _resolve_mask_path(mask_dir, image_dir, image_path)
                rel_mask = common.root_relative_path(root, mask_path) if mask_path is not None else None
                records.append(
                    VisionRecord(
                        image=image,
                        split=split,
                        task="sem-seg",
                        semantic_mask=SemanticMaskRef(path=rel_mask) if rel_mask is not None else None,
                        rel_image_path=rel_img,
                        rel_label_path=rel_mask,
                    )
                )

        return VisionDataset(
            records=records,
            classes=[str(item) for item in payload.get("names", [])],
            task="sem-seg",
            root=root,
        )
