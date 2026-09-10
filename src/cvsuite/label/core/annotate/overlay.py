from pathlib import Path
from typing import Iterable, List, Tuple
import cv2
import numpy as np

from cvsuite.common.core import Record, BBox, Polygon, Keypoints
from .letterbox import letterbox

# Simple but pleasant BGR palette
_PALETTE = [
    (255, 99, 71),
    (56, 161, 243),
    (120, 195, 26),
    (255, 178, 29),
    (204, 46, 72),
    (140, 86, 75),
    (86, 90, 214),
    (0, 192, 255),
    (255, 140, 0),
    (46, 204, 113),
    (155, 89, 182),
]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_FONT_THICKNESS = 2
_SUB_FONT_SCALE = 0.45
_SUB_FONT_THICKNESS = 1
_TEXT_PAD = 3
_TEXT_GAP = 2


def _color(idx: int) -> Tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


def _load_image(path: Path, root: Path | None = None) -> np.ndarray | None:
    resolved = path if path.is_absolute() else (root / path if root else path)
    if not resolved.exists():
        return None
    img = cv2.imread(str(resolved))
    return img


def _target_shape(rec: Record, imgsz: int | None) -> Tuple[int, int] | None:
    if imgsz:
        return (imgsz, imgsz)
    if rec.attributes:
        resize_to = rec.attributes.get("resize_to")
        if resize_to and isinstance(resize_to, (list, tuple)) and len(resize_to) == 2:
            # resize_to stored as (w, h)
            return (int(resize_to[1]), int(resize_to[0]))
    return None


def _scale_point(x: float, y: float, w: int, h: int, ratio: float, pad: Tuple[float, float]) -> Tuple[int, int]:
    px = x * w * ratio + pad[0]
    py = y * h * ratio + pad[1]
    return int(round(px)), int(round(py))


def _xywh_to_xyxy(b: BBox, w: int, h: int, ratio: float, pad: Tuple[float, float]) -> Tuple[int, int, int, int]:
    cx = b.cx * w
    cy = b.cy * h
    bw = b.w * w
    bh = b.h * h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    x1, y1 = _scale_point(x1 / w, y1 / h, w, h, ratio, pad)
    x2, y2 = _scale_point(x2 / w, y2 / h, w, h, ratio, pad)
    return x1, y1, x2, y2


def _poly_points(poly: Polygon, w: int, h: int, ratio: float, pad: Tuple[float, float]) -> np.ndarray:
    pts = [(_scale_point(x, y, w, h, ratio, pad)) for x, y in poly.points]
    if not pts:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(pts, dtype=np.int32)


def _name(cls_idx: int, names: List[str]) -> str:
    if cls_idx < len(names):
        return names[cls_idx]
    return str(cls_idx)


def _maybe_letterbox(img, rec: Record, imgsz: int | None):
    target = _target_shape(rec, imgsz)
    if target:
        canvas, ratio, pad = letterbox(img, target)
        return canvas, ratio, pad
    return img, 1.0, (0.0, 0.0)


def _text_color(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (0, 0, 0) if sum(bg) > 420 else (255, 255, 255)


def _annotation_name(annotation: BBox | Polygon, names: List[str]) -> str:
    label = getattr(annotation, "label", None)
    if isinstance(label, str) and label.strip():
        return label.strip()
    return _name(annotation.cls, names)


def _draw_label(
    img,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int],
    *,
    subtext: str | None = None,
) -> None:
    """Draw a primary label box and an optional secondary prompt line beneath it."""
    (tw, th), base = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
    main_w = tw + _TEXT_PAD * 2
    main_h = th + base + _TEXT_PAD * 2

    sub_w = sub_h = sub_base = sub_th = 0
    if subtext:
        (sub_tw, sub_th), sub_base = cv2.getTextSize(subtext, _FONT, _SUB_FONT_SCALE, _SUB_FONT_THICKNESS)
        sub_w = sub_tw + _TEXT_PAD * 2
        sub_h = sub_th + sub_base + _TEXT_PAD * 2

    total_h = main_h + (_TEXT_GAP + sub_h if subtext else 0)
    x, y = org
    x0 = max(x, 0)
    top = max(y - total_h, 0)

    main_x1 = min(x0 + main_w, img.shape[1] - 1)
    main_y1 = min(top + main_h, img.shape[0] - 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, top), (main_x1, main_y1), color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, dst=img)
    cv2.putText(
        img,
        text,
        (x0 + _TEXT_PAD, top + _TEXT_PAD + th),
        _FONT,
        _FONT_SCALE,
        _text_color(color),
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )
    if not subtext:
        return

    sub_top = min(main_y1 + _TEXT_GAP, img.shape[0] - 1)
    sub_x1 = min(x0 + sub_w, img.shape[1] - 1)
    sub_y1 = min(sub_top + sub_h, img.shape[0] - 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, sub_top), (sub_x1, sub_y1), (44, 44, 44), -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, dst=img)
    cv2.putText(
        img,
        subtext,
        (x0 + _TEXT_PAD, sub_top + _TEXT_PAD + sub_th),
        _FONT,
        _SUB_FONT_SCALE,
        (190, 190, 190),
        _SUB_FONT_THICKNESS,
        cv2.LINE_AA,
    )


def _draw_annotation_label(
    img,
    annotation: BBox | Polygon,
    org: Tuple[int, int],
    color: Tuple[int, int, int],
    names: List[str],
    *,
    show_confidence: bool,
    show_prompt: bool,
) -> None:
    label = _annotation_name(annotation, names)
    if show_confidence and annotation.score is not None:
        label = f"{label} {annotation.score:.2f}"
    prompt = None
    if show_prompt and isinstance(annotation.prompt, str):
        prompt = annotation.prompt.strip() or None
    _draw_label(img, label, org, color, subtext=prompt)


def render_overlay(
    rec: Record,
    names: List[str],
    names_mode: str = "smart",
    imgsz: int | None = None,
    root: Path | None = None,
    *,
    show_confidence: bool = False,
    show_prompt: bool = False,
):
    """Draw boxes/polygons/keypoints on top of the source image."""
    img = _load_image(rec.image.path, root)
    if img is None:
        return None
    canvas, ratio, pad = _maybe_letterbox(img, rec, imgsz)
    h, w = img.shape[:2]

    show_det_names = names_mode in {"all", "det", "smart"}
    show_seg_names = names_mode in {"all", "seg"} or (names_mode == "smart" and not rec.boxes)

    # Segmentation
    for poly in rec.polys:
        pts = _poly_points(poly, w, h, ratio, pad)
        if pts.size == 0:
            continue
        color = _color(poly.cls)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color)
        canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)
        cv2.polylines(canvas, [pts], True, color, 2, lineType=cv2.LINE_AA)
        if show_seg_names:
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            _draw_annotation_label(
                canvas,
                poly,
                (int(cx), int(cy)),
                color,
                names,
                show_confidence=show_confidence,
                show_prompt=show_prompt,
            )

    # Boxes
    for b in rec.boxes:
        color = _color(b.cls)
        x1, y1, x2, y2 = _xywh_to_xyxy(b, w, h, ratio, pad)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        if show_det_names:
            _draw_annotation_label(
                canvas,
                b,
                (x1, max(15, y1 - 5)),
                color,
                names,
                show_confidence=show_confidence,
                show_prompt=show_prompt,
            )

    # Keypoints
    for kp in rec.kpts:
        color = _color(kp.cls)
        for x, y, v in kp.points:
            if v <= 0:
                continue
            px, py = _scale_point(x, y, w, h, ratio, pad)
            cv2.circle(canvas, (px, py), 3, color, -1, lineType=cv2.LINE_AA)

    return canvas


def detection_crops(rec: Record, names: List[str], imgsz: int | None = None, root: Path | None = None) -> Iterable[Tuple[str, np.ndarray]]:
    """Yield (class_name, crop) for each detection box."""
    img = _load_image(rec.image.path, root)
    if img is None:
        return []
    canvas, ratio, pad = _maybe_letterbox(img, rec, imgsz)
    h, w = img.shape[:2]
    out: List[Tuple[str, np.ndarray]] = []
    for b in rec.boxes:
        if b.kind == "pose":
            continue
        x1, y1, x2, y2 = _xywh_to_xyxy(b, w, h, ratio, pad)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, canvas.shape[1] - 1)
        y2 = min(y2, canvas.shape[0] - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = canvas[y1:y2, x1:x2].copy()
        out.append((_name(b.cls, names), crop))
    return out


def class_masks(rec: Record, names: List[str], imgsz: int | None = None, root: Path | None = None) -> Iterable[Tuple[str, np.ndarray]]:
    """Yield (class_name, mask) binary masks for polygons."""
    if not rec.polys:
        return []
    img = _load_image(rec.image.path, root)
    if img is None:
        return []
    canvas, ratio, pad = _maybe_letterbox(img, rec, imgsz)
    h, w = img.shape[:2]
    masks_by_cls: dict[int, np.ndarray] = {}
    for poly in rec.polys:
        pts = _poly_points(poly, w, h, ratio, pad)
        if pts.size == 0:
            continue
        cls_mask = masks_by_cls.setdefault(poly.cls, np.zeros(canvas.shape[:2], dtype=np.uint8))
        cv2.fillPoly(cls_mask, [pts], 255)
    out: List[Tuple[str, np.ndarray]] = []
    for cls_idx, mask in masks_by_cls.items():
        out.append((_name(cls_idx, names), mask))
    return out


def keypoint_overlay(rec: Record, names: List[str], imgsz: int | None = None, root: Path | None = None):
    """Overlay only keypoints onto the image."""
    img = _load_image(rec.image.path, root)
    if img is None:
        return None
    canvas, ratio, pad = _maybe_letterbox(img, rec, imgsz)
    h, w = img.shape[:2]
    for kp in rec.kpts:
        color = _color(kp.cls)
        for x, y, v in kp.points:
            if v <= 0:
                continue
            px, py = _scale_point(x, y, w, h, ratio, pad)
            cv2.circle(canvas, (px, py), 3, color, -1, lineType=cv2.LINE_AA)
    return canvas
