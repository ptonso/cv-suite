"""Perceptual hashes, Hamming-distance twins and SSIM, as described in `specs/common/similarity.md`."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import imagehash
import numpy as np
from PIL import Image

from cvsuite.prep.core.fs import perceptual_hash

MAX_BITS = 64
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_SSIM_C1 = (0.01 * 255) ** 2
_SSIM_C2 = (0.03 * 255) ** 2


def check_bits(value: int) -> int:
    if not 0 <= value <= MAX_BITS:
        raise ValueError(f"A hash threshold must be in [0, {MAX_BITS}] bits, got {value}.")
    return value


def check_ssim(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"An SSIM threshold must be in [0, 1], got {value}.")
    return value


def hash_image(image: Image.Image) -> np.ndarray:
    """The 8 packed bytes of an opened image's perceptual hash."""
    return np.packbits(imagehash.phash(image).hash.reshape(-1))


def hash_bits(paths: Sequence[Path]) -> np.ndarray:
    out = np.empty((len(paths), MAX_BITS // 8), dtype=np.uint8)
    for index, path in enumerate(paths):
        try:
            out[index] = np.packbits(perceptual_hash(Path(path)).hash.reshape(-1))
        except Exception as exc:
            raise ValueError(f"Cannot hash image {path}: {exc}") from exc
    return out


def _block_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _POPCOUNT[left[:, None, :] ^ right[None, :, :]].sum(axis=2, dtype=np.int16)


def nearest_distance(query: np.ndarray, reference: np.ndarray, *, chunk: int = 1024) -> np.ndarray:
    if len(reference) == 0:
        raise ValueError("nearest_distance needs at least one reference hash.")
    out = np.full(len(query), MAX_BITS + 1, dtype=np.int16)
    for qs in range(0, len(query), chunk):
        block = query[qs : qs + chunk]
        best = np.full(len(block), MAX_BITS + 1, dtype=np.int16)
        for rs in range(0, len(reference), chunk):
            best = np.minimum(best, _block_distance(block, reference[rs : rs + chunk]).min(axis=1))
        out[qs : qs + chunk] = best
    return out


def nearest(query: np.ndarray, reference: np.ndarray, k: int = 1, *, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """The `k` closest reference rows of every query row: distances and reference indices, ordered by distance then index."""
    if len(reference) == 0:
        raise ValueError("nearest needs at least one reference hash.")
    k = min(k, len(reference))
    n_ref = len(reference)
    rows = max(1, min(chunk, 16_000_000 // n_ref))
    distances = np.empty((len(query), k), dtype=np.int16)
    indices = np.empty((len(query), k), dtype=np.int64)
    columns = np.arange(n_ref, dtype=np.int64)
    for qs in range(0, len(query), rows):
        block = query[qs : qs + rows]
        full = np.concatenate([_block_distance(block, reference[rs : rs + chunk]) for rs in range(0, n_ref, chunk)], axis=1)
        keyed = full.astype(np.int64) * n_ref + columns
        best = np.sort(np.partition(keyed, k - 1, axis=1)[:, :k], axis=1)
        distances[qs : qs + rows], indices[qs : qs + rows] = best // n_ref, best % n_ref
    return distances, indices


def pairs_within(bits: np.ndarray, max_distance: int, *, chunk: int = 1024) -> list[tuple[int, int]]:
    check_bits(max_distance)
    pairs: list[tuple[int, int]] = []
    for qs in range(0, len(bits), chunk):
        block = bits[qs : qs + chunk]
        for rs in range(qs, len(bits), chunk):
            rows, cols = np.nonzero(_block_distance(block, bits[rs : rs + chunk]) <= max_distance)
            pairs.extend((qs + int(i), rs + int(j)) for i, j in zip(rows, cols) if qs + i < rs + j)
    return sorted(pairs)


def components(n: int, pairs: Sequence[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n)]


def gray_thumb(path: Path, size: int = 160) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Cannot read image {path}: {exc}") from exc


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"ssim needs arrays of equal shape, got {a.shape} and {b.shape}.")
    if min(a.shape) < 11:
        raise ValueError(f"ssim needs at least 11 pixels per side, got {a.shape}.")
    a, b = a.astype(np.float64), b.astype(np.float64)

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + _SSIM_C1) * (2 * cov + _SSIM_C2)) / ((mu_a**2 + mu_b**2 + _SSIM_C1) * (var_a + var_b + _SSIM_C2))
    return float(score[5:-5, 5:-5].mean())
