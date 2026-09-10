from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np, cv2

from .types import Array


def bbox_xyxy_from_norm(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return x1, y1, x2, y2


def polygon_to_mask(points: List[Tuple[float, float]], img_w: int, img_h: int) -> Array:
    m = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(points) >= 3:
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [arr.astype(np.int32)], 1)
    return m


def polygon_bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def nms_xyxy(boxes: Array, scores: Array, iou_thr: float) -> Array:
    if boxes.size == 0:
        return np.zeros((0,), dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)


def cross_class_suppress_xyxy(boxes: Array, scores: Array, labels: Array, iou_thr: float) -> Array:
    if boxes.size == 0:
        return np.zeros((0,), dtype=int)
    order = np.argsort(scores)[::-1]
    keep: List[int] = []
    for idx in order:
        bi = boxes[idx]
        li = labels[idx]
        ok = True
        for k in keep:
            if labels[k] == li:
                continue
            a = bi; b = boxes[k]
            xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
            xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
            w = max(0.0, xx2 - xx1 + 1.0)
            h = max(0.0, yy2 - yy1 + 1.0)
            inter = w * h
            if inter > 0.0:
                aa = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0)
                bb = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0)
                iou = inter / (aa + bb - inter + 1e-9)
                if iou > iou_thr:
                    ok = False
                    break
        if ok:
            keep.append(int(idx))
    keep.sort()
    return np.asarray(keep, dtype=int)


def box_iou(b1: Array, b2: Array) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / max(a1 + a2 - inter, 1e-12)


def load_rgb_safe(path: Path) -> Optional[Array]:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

