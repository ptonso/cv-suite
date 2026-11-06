"""File-level cues that could predict a label, and their binning. Contract: `specs/eda/report.md#sections` (leak)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

CUES = ("resolution", "aspect", "file_bytes", "container", "brightness", "sharpness", "name_digits", "file_index", "mtime_minute")
NUMERIC = frozenset({"aspect", "file_bytes", "brightness", "sharpness", "name_digits", "file_index"})
COMBINED = ("file_bytes", "brightness", "sharpness")
ABSENT = "none"
UNSEEN = "unseen"
_STD_LUMA_MEAN = 57.625  # mean of the libjpeg quality-50 luminance table


def jpeg_quality(table: list[int]) -> int:
    """The libjpeg quality (rounded to 5) that scales the standard luminance table to `table`."""
    scale = 100.0 * float(np.mean(table)) / _STD_LUMA_MEAN
    quality = (200.0 - scale) / 2 if scale <= 100 else 5000.0 / scale
    return int(min(100, max(5, round(quality / 5) * 5)))


def container(path: Path) -> str:
    """The extension, with the JPEG quality and EXIF presence when the file is a JPEG."""
    ext = path.suffix.lower().lstrip(".")
    with Image.open(path) as image:
        if image.format != "JPEG":
            return ext
        quality = jpeg_quality(list(image.quantization[0]))
        return f"{ext}|q{quality}|{'exif' if len(image.getexif()) else 'noexif'}"


def _digits(stem: str) -> float | None:
    runs = re.findall(r"\d+", stem)
    return float(int(runs[-1])) if runs else None


def extract(rows: list[dict], paths: list[Path]) -> dict[str, list]:
    """Raw cue values per row (None where a cue is absent); binning is fitted separately on the fitting set."""
    stems = [Path(row["path"]).stem for row in rows]
    position = {key: n for n, key in enumerate(sorted(zip(stems, [row["path"] for row in rows])))}
    return {
        "resolution": [f"{row['width']}x{row['height']}" for row in rows],
        "aspect": [row["width"] / row["height"] for row in rows],
        "file_bytes": [float(row["file_bytes"]) for row in rows],
        "container": [container(path) for path in paths],
        "brightness": [float(row["brightness"]) for row in rows],
        "sharpness": [float(row["sharpness"]) for row in rows],
        "name_digits": [_digits(stem) for stem in stems],
        "file_index": [float(position[(stem, row["path"])]) for stem, row in zip(stems, rows)],
        "mtime_minute": [str(int(row["mtime"]) // 60) for row in rows],
    }


def _edge_label(edges: np.ndarray, bin_id: int) -> str:
    if len(edges) == 0:
        return "all"
    if bin_id == 0:
        return f"<={edges[0]:.4g}"
    if bin_id == len(edges):
        return f">{edges[-1]:.4g}"
    return f"({edges[bin_id - 1]:.4g}, {edges[bin_id]:.4g}]"


def encode(raw: list, numeric: bool, fit: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Integer bin id per row and the label of each id. Id 0 is a value the fitting rows never showed.

    Numeric cues use quartile bins of the fitting rows (absent values form the `none` bin); the others use exact values.
    """
    if numeric:
        values = np.array([np.nan if v is None else v for v in raw], dtype=float)
        known = values[fit & ~np.isnan(values)]
        edges = np.unique(np.quantile(known, [0.25, 0.5, 0.75])) if len(known) else np.array([])
        ids = np.where(np.isnan(values), len(edges) + 2, 1 + np.searchsorted(edges, np.nan_to_num(values), side="left"))
        return ids.astype(np.int64), [UNSEEN, *[_edge_label(edges, b) for b in range(len(edges) + 1)], ABSENT]
    values = [ABSENT if v is None else str(v) for v in raw]
    fitted = [v for v, f in zip(values, fit) if f]
    counts = {v: fitted.count(v) for v in set(fitted)}
    seen = sorted(counts, key=lambda v: (-counts[v], v))
    index = {v: n + 1 for n, v in enumerate(seen)}
    return np.array([index.get(v, 0) for v in values], dtype=np.int64), [UNSEEN, *seen]


def joint(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """One id per distinct pair of ids, numbered densely."""
    return np.unique(left * (int(right.max()) + 1) + right, return_inverse=True)[1].astype(np.int64)
