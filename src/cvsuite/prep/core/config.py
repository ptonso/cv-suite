from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


FlattenMode = Literal["none", "plain", "enc-prefix", "enc-suffix"]
DedupMode = Literal["none", "exact", "perceptual"]
LinkMode = Literal["copy", "move", "hard"]
NonImagePolicy = Literal["skip", "collect", "keep"]
VideoSampleMode = Literal["skip", "collect", "keep"]

ResizeMode = Literal["none", "long", "short", "square", "cap-long", "cap-short"]
SquareMode = Literal["pad", "crop", "distort"]
TransferMode = Literal["copy", "move"]
ImageBackend = Literal["pillow", "opencv"]

@dataclass(frozen=True)
class ArrangeConfig:
    src: Path
    dst: Path
    unzip: bool
    flatten: FlattenMode
    dedup_mode: DedupMode
    perceptual_threshold: Optional[int]
    rename_seq: bool
    dst_subdir: Optional[str]
    link_mode: LinkMode
    non_image: NonImagePolicy
    manifest: bool
    dry_run: bool
    verbose: int


@dataclass(frozen=True)
class ProcessConfig:
    src: Path
    dst: Path
    orient: bool
    resize_mode: ResizeMode
    size: Optional[int]
    square_mode: Optional[SquareMode]
    grayscale: bool
    to_format: Optional[str]
    backend: ImageBackend
    jpeg_quality: int
    transfer: TransferMode
    non_image: NonImagePolicy
    video_sample_mode: VideoSampleMode
    video_sample_k: Optional[int]
    video_sample_min_gap: Optional[float]
    manifest: bool
    dry_run: bool
    verbose: int


@dataclass(frozen=True)
class OrientConfig:
    src: Path
    dst: Path
    try_hardlink: bool
    batch: int
    device: str
    precision: str
    weights: Optional[Path]
    dry_run: bool
    verbose: int
    no_resume: bool = False


@dataclass(frozen=True)
class SampleConfig:
    srcs: tuple[Path, ...]
    dst: Path
    count: Optional[int]
    frac: Optional[float]
    hardlink: bool
    seed: Optional[int]
    dry_run: bool
    verbose: int
