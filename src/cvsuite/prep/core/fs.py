
import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, Iterator

import imagehash
from PIL import Image

from .config import LinkMode


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
    ".webp",
}

# Artefact names that prep commands write into their destination trees. Other
# prep commands treat these as non-content and skip them when re-reading a tree.
PREP_MANIFEST_NAME = "manifest.yaml"
PREP_NON_IMAGE_DIR = "_non_images"
PREP_VIDEO_FRAMES_DIR = "_video_frames"
PREP_ARTIFACT_NAMES = frozenset({PREP_MANIFEST_NAME, PREP_NON_IMAGE_DIR, PREP_VIDEO_FRAMES_DIR})


def iter_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        base = Path(dirpath)
        for name in sorted(filenames):
            yield base / name


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def require_image_source(src: Path, *, label: str = "source", allow_video: bool = False, allow_archive: bool = False) -> None:
    """Fail fast when a prep source directory is missing, not a directory, or empty of content."""
    if not src.exists():
        raise SystemExit(f"{label} not found: {src}")
    if not src.is_dir():
        raise SystemExit(f"{label} must be a directory: {src}")
    if allow_archive:
        return
    from .process.vsample import is_video  # local import: avoids a cycle at module load

    for path in iter_files(src):
        if is_image(path) or (allow_video and is_video(path)):
            return
    raise SystemExit(f"No images found in {label.lower()}: {src}")


def file_size(path: Path) -> int:
    return path.stat().st_size


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as img:
        return imagehash.phash(img)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def materialize_files_copy(
    pairs: Iterable[tuple[Path, Path]],
) -> None:
    for src, dst in pairs:
        ensure_parent(dst)
        shutil.copy2(src, dst)


def materialize_files_move(
    pairs: Iterable[tuple[Path, Path]],
) -> None:
    for src, dst in pairs:
        ensure_parent(dst)
        shutil.move(src, dst)


def materialize_files_hardlink(
    pairs: Iterable[tuple[Path, Path]],
) -> None:
    for src, dst in pairs:
        ensure_parent(dst)
        if dst.exists():
            dst.unlink()
        os.link(src, dst)


def materialize_files(
    pairs: Iterable[tuple[Path, Path]],
    mode: LinkMode,
) -> None:
    if mode == "copy":
        materialize_files_copy(pairs)
    elif mode == "move":
        materialize_files_move(pairs)
    elif mode == "hard":
        materialize_files_hardlink(pairs)
    else:
        raise ValueError(f"Unknown link mode: {mode}")
