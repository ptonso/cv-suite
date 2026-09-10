from __future__ import annotations
from typing import List
from cvsuite.common.core import Record, BBox

def rebox_item(it: Record, mode: str, margin: float, edge_tol: float) -> Record:
    if not it.kpts:
        return it
    xs, ys = [], []
    for kp in it.kpts:
        for x, y, v in kp.points:
            if v > 0:
                xs.append(x)
                ys.append(y)
    if not xs:
        return it
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if mode == "expand":
        w *= 1 + margin * 2
        h *= 1 + margin * 2
    elif mode == "adaptive":
        w *= 1 + margin * 2
        h *= 1 + margin * 2
    it.boxes = [BBox(cx=cx, cy=cy, w=w, h=h, cls=it.kpts[0].cls)]
    return it
