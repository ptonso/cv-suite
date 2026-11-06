"""The image table every section reads: one row per record, plus in-memory thumbnails. Contract: `specs/eda/report.md#imagescsv`."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from cvsuite.common.core import VisionDataset
from cvsuite.common.io.common import resolve_record_image_path
from cvsuite.common.keys import group_records, layout_path, parse_key
from cvsuite.common.similarity import hash_image
from cvsuite.common.transforms.split import EMPTY, record_labels

COLUMNS = ("path", "split", "group", "class", "n_instances", "width", "height", "file_bytes", "ext", "brightness", "sharpness", "phash", "mtime")
THUMB = 160


@dataclass
class ImageTable:
    rows: list[dict]
    thumbs: list[np.ndarray | None]
    unreadable: list[tuple[str, str]] = field(default_factory=list)

    def readable(self) -> list[int]:
        return [i for i, row in enumerate(self.rows) if row["width"] != ""]

    def write(self, path: Path, *, with_group: bool) -> Path:
        columns = [c for c in COLUMNS if with_group or c != "group"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
        return path


def _classes_of(record, dataset: VisionDataset) -> tuple[str, int]:
    labels = record_labels(record, dataset.classes)
    if not labels:
        return EMPTY, 0
    return ";".join(sorted(set(labels))), len(labels)


def _pixels(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size
        if width == 0 or height == 0:
            raise ValueError("zero-size image")
        thumb = np.asarray(image.convert("L").resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float32)
        digest = hash_image(image).tobytes().hex()
    return {
        "width": width,
        "height": height,
        "brightness": round(float(thumb.mean()), 3),
        "sharpness": round(float(cv2.Laplacian(thumb.astype(np.float64), cv2.CV_64F).var()), 3),
        "phash": digest,
        "thumb": thumb,
    }


def _group_names(dataset: VisionDataset, group_by: list[str], rows: list[dict]) -> list[str]:
    """The key value for one spec, so a group reads `pcb12`; the smallest member path when specs are joined.

    A `similar:` spec reads the hashes already in the rows, so no image is hashed twice.
    """
    if not group_by:
        return [""] * len(dataset.records)
    if len(group_by) == 1 and not group_by[0].startswith("similar:"):
        return parse_key(group_by[0]).values(dataset)
    bits = None
    if any(spec.startswith("similar:") for spec in group_by):
        missing = [row["path"] for row in rows if row["phash"] == ""]
        if missing:
            raise ValueError(f"Cannot group by similarity: image {missing[0]} is unreadable.")
        bits = hash_array([row["phash"] for row in rows])
    return group_records(dataset, group_by, bits=bits)


def hash_array(hexes: list[str]) -> np.ndarray:
    """Packed hash bytes of table `phash` values, one row each."""
    return np.frombuffer(bytes.fromhex("".join(hexes)), dtype=np.uint8).reshape(len(hexes), -1)


def build_table(dataset: VisionDataset, group_by: list[str], *, progress: bool = True) -> ImageTable:
    rows: list[dict] = []
    thumbs: list[np.ndarray | None] = []
    unreadable: list[tuple[str, str]] = []
    for record in tqdm(dataset.records, desc="images", disable=not progress, leave=False):
        label, n = _classes_of(record, dataset)
        path = resolve_record_image_path(dataset, record)
        row = {
            "path": layout_path(record, dataset),
            "split": record.split or "train",
            "group": "",
            "class": label,
            "n_instances": n,
            "width": "",
            "height": "",
            "file_bytes": "",
            "ext": path.suffix.lower().lstrip("."),
            "brightness": "",
            "sharpness": "",
            "phash": "",
            "mtime": "",
        }
        thumb = None
        try:
            stat = path.stat()
            row["file_bytes"], row["mtime"] = stat.st_size, int(stat.st_mtime)
            pixels = _pixels(path)
            thumb = pixels.pop("thumb")
            row.update(pixels)
        except Exception as exc:
            unreadable.append((row["path"], str(exc)))
            for key in ("width", "height", "brightness", "sharpness", "phash"):
                row[key] = ""
        rows.append(row)
        thumbs.append(thumb)
    for row, group in zip(rows, _group_names(dataset, group_by, rows)):
        row["group"] = group
    return ImageTable(rows, thumbs, unreadable)
