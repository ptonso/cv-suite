from __future__ import annotations

from pathlib import Path
from typing import Literal

from cvsuite.common.core import VisionDataset
from cvsuite.common.io import read_dataset
from cvsuite.common.io.flat import FlatAdapter
from cvsuite.common.io.flat_vlm_json import FlatVlmJsonAdapter
from cvsuite.common.io.shards_vlm import ShardsVlmAdapter
from cvsuite.common.io.vqa_style import VQAStyleAdapter

from cvsuite.common.core import normalize_dataset

Fmt = Literal["images", "json", "shards", "vqa-style"]


def detect_format(src: Path, from_hint: str = "auto") -> Fmt:
    if from_hint != "auto":
        return from_hint  # type: ignore[return-value]

    path = src.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")

    if path.is_file():
        if ShardsVlmAdapter.matches(path):
            return "shards"
        if VQAStyleAdapter.matches(path):
            return "vqa-style"
        if FlatAdapter.matches(path):
            return "images"
        raise RuntimeError(f"Could not infer dataset format from: {src}")

    recognized: list[Fmt] = []
    if FlatVlmJsonAdapter.matches(path):
        recognized.append("json")
    if ShardsVlmAdapter.matches(path):
        recognized.append("shards")
    if VQAStyleAdapter.matches(path):
        recognized.append("vqa-style")
    if len(recognized) > 1:
        raise RuntimeError(f"Ambiguous dataset format for {src}: {', '.join(recognized)}")
    if recognized:
        return recognized[0]
    if FlatAdapter.matches(path):
        return "images"
    raise RuntimeError(f"Could not infer dataset format from: {src}")


def ingest(src: Path, from_hint: str = "auto") -> VisionDataset:
    fmt = detect_format(src, from_hint=from_hint)
    format_name = {
        "images": "flat",
        "json": "flat_vlm_json",
        "shards": "shards_vlm",
        "vqa-style": "vqa_style",
    }[fmt]
    dataset = read_dataset(src, format=format_name, task="auto")
    dataset.meta.setdefault("format", fmt)
    return normalize_dataset(dataset)
