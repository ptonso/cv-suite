from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np

from cvsuite.common.core import BBox

Rect = Tuple[float, float, float, float]
IntRect = Tuple[int, int, int, int]


def resolve_image_path(path: Path, root: Path | None) -> Path:
    if path.is_absolute():
        return path
    if root is not None:
        return (root / path).resolve()
    return path.resolve()


def annotation_label(cls_idx: int, classes: Sequence[str], explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if 0 <= cls_idx < len(classes):
        class_name = str(classes[cls_idx]).strip()
        if class_name:
            return class_name
    return str(cls_idx)


def sanitize_label(name: str) -> str:
    sanitized = name.replace("/", "_").replace(" ", "_")
    return sanitized or "unnamed"


def bbox_xyxy_from_norm(cx: float, cy: float, w: float, h: float, *, img_w: int, img_h: int) -> Rect:
    box_w = w * img_w
    box_h = h * img_h
    center_x = cx * img_w
    center_y = cy * img_h
    x1 = center_x - box_w / 2.0
    y1 = center_y - box_h / 2.0
    x2 = center_x + box_w / 2.0
    y2 = center_y + box_h / 2.0
    return x1, y1, x2, y2


def expand_rect_xyxy(rect: Rect, *, margin_frac: float = 0.0) -> Rect:
    if margin_frac <= 0.0:
        return rect

    x1, y1, x2, y2 = rect
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    margin_x = width * margin_frac
    margin_y = height * margin_frac
    return x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y


def clamp_crop_xyxy(x1: float, y1: float, x2: float, y2: float, *, img_w: int, img_h: int) -> IntRect | None:
    if img_w <= 0 or img_h <= 0:
        return None
    if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
        return None

    left = max(0, int(math.floor(x1)))
    top = max(0, int(math.floor(y1)))
    right = min(img_w, int(math.ceil(x2)))
    bottom = min(img_h, int(math.ceil(y2)))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def bbox_crop_xyxy(
    box: BBox,
    *,
    img_w: int,
    img_h: int,
    margin_frac: float = 0.0,
) -> tuple[Rect, IntRect | None]:
    source_xyxy = bbox_xyxy_from_norm(box.cx, box.cy, box.w, box.h, img_w=img_w, img_h=img_h)
    crop_xyxy = clamp_crop_xyxy(*expand_rect_xyxy(source_xyxy, margin_frac=margin_frac), img_w=img_w, img_h=img_h)
    return source_xyxy, crop_xyxy


def next_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def resize_long_side(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("--size must be a positive integer.")

    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= 0 or long_side == size:
        return image

    scale = float(size) / float(long_side)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)
