from pathlib import Path
from typing import Optional
import os
import cv2
import numpy as np

from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset
from .overlay import render_overlay
from .utils import select_indices

_QT_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/dejavu"),
    Path("/usr/share/fonts/truetype/liberation2"),
    Path("/usr/share/fonts/liberation"),
)


def _fix_size(im: np.ndarray, size: int) -> np.ndarray:
    h, w = im.shape[:2]
    if h == size and w == size:
        return im
    interp = cv2.INTER_NEAREST if (h < size or w < size) else cv2.INTER_AREA
    return cv2.resize(im, (size, size), interpolation=interp)


def _make_vis(
    split: str,
    rec: Record,
    classes,
    imgsz: Optional[int],
    strict: bool,
    names_mode: str,
    root: Optional[Path],
    *,
    show_confidence: bool,
    show_prompt: bool,
):
    vis = render_overlay(
        rec,
        classes,
        names_mode=names_mode,
        imgsz=imgsz,
        root=root,
        show_confidence=show_confidence,
        show_prompt=show_prompt,
    )
    if vis is None:
        return None
    if strict and (vis.shape[0] != imgsz or vis.shape[1] != imgsz):
        vis = _fix_size(vis, imgsz)
    return vis


def _prepare_qt_environment() -> None:
    plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if plugin_path and not Path(plugin_path).exists():
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    font_dir = os.environ.get("QT_QPA_FONTDIR")
    if font_dir and Path(font_dir).exists():
        return

    for candidate in _QT_FONT_CANDIDATES:
        if candidate.exists():
            os.environ["QT_QPA_FONTDIR"] = str(candidate)
            return

    os.environ.pop("QT_QPA_FONTDIR", None)


def run_preview(
    ds: VisionDataset,
    imgsz: int,
    max_items: int,
    names_mode: str = "smart",
    *,
    show_confidence: bool = False,
    show_prompt: bool = False,
) -> None:
    sets = ds.by_split()
    if not sets:
        raise RuntimeError("Empty dataset.")

    order = [(sp, it) for sp, items in sorted(sets.items()) for it in items]
    if not order:
        raise RuntimeError("No items to preview.")
    idxs = select_indices(len(order), max_items) if max_items else list(range(len(order)))
    order = [order[i] for i in idxs]

    win = "inspect"
    strict = imgsz is not None
    _prepare_qt_environment()
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE if strict else cv2.WINDOW_NORMAL)

    idx = 0
    n = len(order)

    root = ds.root
    while True:
        split, rec = order[idx]
        vis = _make_vis(
            split,
            rec,
            ds.classes,
            imgsz,
            strict,
            names_mode,
            root,
            show_confidence=show_confidence,
            show_prompt=show_prompt,
        )
        if vis is None:
            idx = (idx + 1) % n
            if idx == 0:
                raise RuntimeError("No readable images.")
            continue

        title = f"{split} [{idx+1}/{n}] – {Path(rec.image.path).name}"
        cv2.setWindowTitle(win, title)
        cv2.imshow(win, vis)

        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord("q")):
            break
        if k in (81, ord("a")):
            idx = (idx - 1) % n
            continue
        if k in (83, ord("d")):
            idx = (idx + 1) % n
            continue

    cv2.destroyAllWindows()
