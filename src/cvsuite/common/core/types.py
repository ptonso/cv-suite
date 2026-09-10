"""Canonical in-memory dataset types owned by cvsuite.

`cvsuite.common.core` is the single home of the shared dataset contract.
`types.py` holds the leaf value types (geometry, image reference, classification,
VQA) plus the cvsuite runtime dataclasses (`FMRequest`, `Embedding`, `LabelMeShape`);
`records.py` holds the containers (`VisionRecord`, `VisionDataset`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

TaskId: TypeAlias = Literal[
    "cls",
    "multi-cls",
    "det",
    "obb",
    "inst-seg",
    "sem-seg",
    "pose",
]

SplitId: TypeAlias = Literal["train", "val", "test", "infer"]

FormatId: TypeAlias = Literal[
    "semseg_mask",
    "yolo",
    "labelme",
    "coco",
    "class-dir",
    "flat",
    "flat_vlm_json",
    "vqa_style",
    "shards_vlm",
]

Point: TypeAlias = tuple[float, float]
KeypointValue: TypeAlias = tuple[float, float, float]


class ImageInfo:
    """Image reference whose dimensions are read from `source` on first access.

    `path` is the relocatable (often root-relative) reference stored on records; `source`
    is the absolute path opened for lazy dimension reads, so reassigning `path` does not
    break them. Eager construction with explicit `width`/`height` skips the open.
    """

    __slots__ = ("path", "source", "_width", "_height")

    def __init__(
        self,
        path: Path,
        width: int | None = None,
        height: int | None = None,
        source: Path | None = None,
    ) -> None:
        self.path = path
        self.source = source if source is not None else path
        self._width = width
        self._height = height

    def _resolve(self) -> None:
        from PIL import Image

        with Image.open(self.source) as image:
            self._width, self._height = image.size

    @property
    def width(self) -> int:
        if self._width is None:
            self._resolve()
        return self._width

    @width.setter
    def width(self, value: int) -> None:
        self._width = value

    @property
    def height(self) -> int:
        if self._height is None:
            self._resolve()
        return self._height

    @height.setter
    def height(self, value: int) -> None:
        self._height = value

    def __repr__(self) -> str:
        return f"ImageInfo(path={self.path!r}, width={self._width!r}, height={self._height!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageInfo):
            return NotImplemented
        return (self.path, self._width, self._height) == (other.path, other._width, other._height)


@dataclass
class BBox:
    cx: float
    cy: float
    w: float
    h: float
    cls: int
    label: str | None = None
    score: float | None = None
    id: int | str | None = None
    group_id: int | str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    # cvsuite annotation overlay
    kind: str = "det"
    prompt: str | None = None
    model: str | None = None
    text: str | None = None


@dataclass
class Polygon:
    points: list[Point]
    cls: int
    label: str | None = None
    score: float | None = None
    id: int | str | None = None
    group_id: int | str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    # cvsuite annotation overlay
    prompt: str | None = None
    model: str | None = None


@dataclass
class Keypoints:
    points: list[KeypointValue]
    cls: int
    label: str | None = None
    score: float | None = None
    id: int | str | None = None
    group_id: int | str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    # cvsuite annotation overlay
    kind: str = "pose"


@dataclass
class Classification:
    label: str | None = None
    score: float | None = None
    probs: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class VQA:
    question: str
    answer: str
    score: float | None = None
    model: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticMaskRef:
    path: Path
    encoding: str = "class-index"
    ignore_index: int | None = 255
    attributes: dict[str, Any] = field(default_factory=dict)


# --- cvsuite runtime dataclasses ---------------------------------------


@dataclass
class LabelMeShape:
    label: str
    shape_type: str
    points: list[tuple[float, float]]
    flags: dict[str, Any] = field(default_factory=dict)
    group_id: int | str | None = None
    description: str | None = None
    other_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Embedding:
    vector: list[float] = field(default_factory=list)
    model: str | None = None
    task: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FMRequest:
    task: str = ""
    provider: str = ""
    prompt: str = ""
    config_path: Path | None = None
    device: str = "auto"
    precision: str = "fp32"
    max_gpu_memory: str | None = None
    batch_size: int = 1
    weights_dir: Path | None = None
    model_args: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def model(self) -> str:
        return self.provider

    @model.setter
    def model(self, value: str) -> None:
        self.provider = value


ImageRecord = ImageInfo

__all__ = [
    "BBox",
    "Classification",
    "Embedding",
    "FMRequest",
    "FormatId",
    "ImageInfo",
    "ImageRecord",
    "KeypointValue",
    "Keypoints",
    "LabelMeShape",
    "Point",
    "Polygon",
    "SemanticMaskRef",
    "SplitId",
    "TaskId",
    "VQA",
]
