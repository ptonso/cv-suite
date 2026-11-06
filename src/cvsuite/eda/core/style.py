"""One plotting style and the cap rule shared by every eda figure. Contract: `specs/eda/report.md`."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

INSTALL_HINT = 'pip install "cv-suite[eda]"'
PALETTE = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#9d755d", "#bab0ac")
BASELINE = "#888888"
OTHER = "#c9c9c9"


def ensure_matplotlib():
    """Import matplotlib with the Agg backend, or exit with the install command."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(f"cvsuite eda draws its figures with matplotlib. Install it with: {INSTALL_HINT}") from None
    return plt


def apply_style() -> None:
    plt = ensure_matplotlib()
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.dpi": 100,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def top_entries(counts: Mapping[str, float], n: int) -> tuple[list[tuple[str, float]], int]:
    """The `n` largest entries, plus one `other (k)` entry for the k left out. Returns the entries and k."""
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept, rest = ordered[:n], ordered[n:]
    if rest:
        kept.append((f"other ({len(rest)})", float(sum(v for _, v in rest))))
    return kept, len(rest)


def spans_orders(values: Sequence[float], factor: float = 100.0) -> bool:
    positive = [v for v in values if v > 0]
    return bool(positive) and max(positive) / min(positive) >= factor


def new_figure(width: float = 7.0, height: float = 4.0):
    plt = ensure_matplotlib()
    apply_style()
    return plt.subplots(figsize=(width, height))


def save_figure(fig, path: Path) -> Path:
    plt = ensure_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def tile(images: Sequence[np.ndarray], cols: int, size: int, labels: Sequence[str] | None = None) -> np.ndarray:
    """A grid of BGR images resized to `size` squares; missing cells are black."""
    rows = max(1, -(-len(images) // cols))
    canvas = np.zeros((rows * size, cols * size, 3), dtype=np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, cols)
        cell = cv2.resize(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), (size, size), interpolation=cv2.INTER_AREA)
        if labels and labels[i]:
            cv2.putText(cell, labels[i][:28], (4, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[r * size : (r + 1) * size, c * size : (c + 1) * size] = cell
    return canvas


def labeled_rows(rows: Sequence[tuple[str, Sequence[np.ndarray]]], cols: int, size: int, label_width: int = 170) -> np.ndarray:
    """One labeled row of up to `cols` cells per entry."""
    out = np.zeros((len(rows) * size, label_width + cols * size, 3), dtype=np.uint8)
    for r, (label, images) in enumerate(rows):
        cv2.putText(out, label[:26], (6, r * size + size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if images:
            out[r * size : (r + 1) * size, label_width : label_width + len(images) * size] = tile(images[:cols], len(images[:cols]), size)
    return out


def write_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write {path}")
    return path


def bars(ax, entries: Sequence[tuple[str, float]], *, horizontal: bool = True, log: bool = False, color: str | None = None) -> None:
    """Bars with the value printed on each; the `other (k)` entry is grayed."""
    names = [n for n, _ in entries]
    values = [v for _, v in entries]
    colors = [OTHER if n.startswith("other (") else (color or PALETTE[0]) for n in names]
    positions = np.arange(len(names))
    if horizontal:
        ax.barh(positions, values, color=colors)
        ax.set_yticks(positions, names)
        ax.invert_yaxis()
        if log:
            ax.set_xscale("log")
        for p, v in zip(positions, values):
            ax.text(v, p, f" {v:g}", va="center", fontsize=7)
    else:
        ax.bar(positions, values, color=colors)
        ax.set_xticks(positions, names, rotation=60, ha="right")
        if log:
            ax.set_yscale("log")
        for p, v in zip(positions, values):
            ax.text(p, v, f"{v:g}", ha="center", va="bottom", fontsize=7)
