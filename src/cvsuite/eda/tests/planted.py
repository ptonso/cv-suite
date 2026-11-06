"""Datasets with planted structure for the split and leak sections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PADDING = (0, 6000, 12000)


def noise(seed: int, size: int = 64) -> np.ndarray:
    return (np.random.default_rng(seed).random((size, size, 3)) * 255).astype("uint8")


def save(array: np.ndarray, path: Path, quality: int, padding: int = 0) -> None:
    """Write a JPEG; `padding` bytes of comment change the file size and no pixel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, quality=quality, comment=b"x" * padding)


def planted(
    root: Path,
    *,
    classes: int = 3,
    per_class: int = 12,
    test_per_class: int = 4,
    twins: bool = False,
    size_leak: bool = False,
    sequential_names: bool = False,
    test_classes: int | None = None,
    groups: int | None = None,
    splits: tuple[str, str] = ("train", "test"),
) -> Path:
    """A class-folder dataset. Twins re-encode a train image into test; a size leak encodes the class in the JPEG quality."""
    next_seed = 0

    def stem(seed: int, default: str) -> str:
        return f"g{seed % groups}_{seed}" if groups else default

    for c in range(classes):
        padding = PADDING[c % len(PADDING)] if size_leak else 0
        train = []
        for i in range(per_class):
            next_seed += 1
            array = noise(next_seed)
            train.append(array)
            name = f"img_{c * 100 + i}.jpg" if sequential_names else f"{stem(next_seed, f't{next_seed}')}.jpg"
            save(array, root / splits[0] / f"c{c}" / name, 75, padding)
        if test_classes is not None and c >= test_classes:
            continue
        for i in range(test_per_class):
            next_seed += 1
            array = train[i] if twins else noise(next_seed)
            name = f"img_{c * 100 + per_class + i}.jpg" if sequential_names else f"{stem(next_seed, f'x{next_seed}')}.jpg"
            save(array, root / splits[1] / f"c{c}" / name, 75 if size_leak else 70, padding)
    return root
