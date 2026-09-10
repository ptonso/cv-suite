from __future__ import annotations

from pathlib import Path
import os

from PIL import Image, ImageDraw

from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset

from . import io

_QT_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/dejavu"),
    Path("/usr/share/fonts/truetype/liberation2"),
    Path("/usr/share/fonts/liberation"),
)


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


def _wrap_text(text: str, width: int = 48) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    out: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
            continue
        out.append(current)
        current = word
    out.append(current)
    return out


def _text_lines(record: Record) -> list[str]:
    lines = [Path(io.record_relative_image_path(record)).as_posix(), f"split: {record.split or 'train'}", ""]
    for idx, qa in enumerate(record.vqas, start=1):
        lines.append(f"Q{idx}:")
        for line in _wrap_text(qa.question or "(empty question)"):
            lines.append(f"  {line}")
        lines.append(f"  answer: {qa.answer or '<pending>'}")
        meta = dict(qa.meta or {})
        ground_truth = meta.get("ground_truth")
        if ground_truth is not None:
            lines.append(f"  ground truth: {ground_truth}")
        expected_answers = meta.get("expected_answers")
        if expected_answers is not None:
            lines.append(f"  expected: {expected_answers}")
        label = meta.get("label")
        if label is not None:
            lines.append(f"  label: {label}")
        if qa.model:
            lines.append(f"  model: {qa.model}")
        lines.append("")
    return lines


def render_record(dataset: VisionDataset, record: Record, *, max_height: int = 720, panel_width: int = 700) -> np.ndarray:
    import numpy as np

    src = io.resolve_record_image_path(dataset, record)
    with Image.open(src) as image:
        vis = image.convert("RGB")
    width, height = vis.size
    scale = min(1.0, float(max_height) / float(height)) if height else 1.0
    if scale != 1.0:
        vis = vis.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)

    panel = Image.new("RGB", (panel_width, vis.size[1]), color=(248, 248, 248))
    draw = ImageDraw.Draw(panel)
    y = 18
    for line in _text_lines(record):
        draw.text((16, y), line, fill=(20, 20, 20))
        y += 20
        if y > panel.size[1] - 24:
            draw.text((16, panel.size[1] - 20), "...", fill=(20, 20, 20))
            break

    canvas = Image.new("RGB", (vis.size[0] + panel.size[0], vis.size[1]), color=(255, 255, 255))
    canvas.paste(vis, (0, 0))
    canvas.paste(panel, (vis.size[0], 0))
    return np.array(canvas)


def run_preview(dataset: VisionDataset, *, splits: list[str] | None = None, max_items: int = 0) -> None:
    import cv2

    order = [record for record in dataset.records if not splits or record.split in set(splits)]
    if max_items > 0:
        order = order[:max_items]
    if not order:
        raise RuntimeError("No items to inspect.")

    _prepare_qt_environment()
    window = "cvsuite-vlm-inspect"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    index = 0

    while True:
        record = order[index]
        canvas = render_record(dataset, record)
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.setWindowTitle(window, f"{record.split} [{index + 1}/{len(order)}] - {Path(record.image.path).name}")
        cv2.imshow(window, canvas_bgr)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (81, ord("a")):
            index = (index - 1) % len(order)
            continue
        if key in (83, ord("d")):
            index = (index + 1) % len(order)
            continue

    cv2.destroyAllWindows()
