from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np, cv2

from .types import Features

Array = np.ndarray


def _box_features(H: int, W: int, boxes: Array) -> Dict[str, Array]:
    if boxes.size == 0:
        z = np.zeros((0,), dtype=float)
        return {"box_area_frac": z, "short_side_frac": z, "aspect_ratio": z}
    fh, fw = float(H), float(W)
    wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1).clip(min=1.0)
    area = wh[:, 0] * wh[:, 1]
    box_area_frac = area / (fh * fw + 1e-12)
    short_side_frac = np.minimum(wh[:, 0], wh[:, 1]) / (min(fw, fh) + 1e-12)
    aspect_ratio = (np.maximum(wh[:, 0], wh[:, 1]) / np.maximum(np.minimum(wh[:, 0], wh[:, 1]), 1e-12)).clip(min=1.0)
    return {
        "box_area_frac": box_area_frac.astype(float),
        "short_side_frac": short_side_frac.astype(float),
        "aspect_ratio": aspect_ratio.astype(float),
    }


def _hsv_from_rgb01(rgb01: Array) -> Tuple[Array, Array, Array]:
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    cmax = np.max(rgb01, axis=-1)
    cmin = np.min(rgb01, axis=-1)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    m = delta > 1e-12
    rc = ((cmax == r) & m)
    gc = ((cmax == g) & m)
    bc = ((cmax == b) & m)
    h[rc] = ((g - b)[rc] / (delta[rc] + 1e-12)) % 6.0
    h[gc] = ((b - r)[gc] / (delta[gc] + 1e-12)) + 2.0
    h[bc] = ((r - g)[bc] / (delta[bc] + 1e-12)) + 4.0
    h = h / 6.0
    s = np.where(cmax <= 1e-12, 0.0, delta / (cmax + 1e-12))
    v = cmax
    return h, s, v


def _rgb_to_hsv01(rgb: Array) -> Tuple[Array, Array, Array]:
    arr = rgb.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return _hsv_from_rgb01(arr)


def _color_masks(h: Array, s: Array, v: Array) -> Dict[str, Array]:
    masks: Dict[str, Array] = {}
    masks["white"] = (s <= 0.20) & (v >= 0.85)
    masks["black"] = v <= 0.12

    def hue_band(center: float, width: float = 0.06) -> Array:
        lo = (center - width) % 1.0
        hi = (center + width) % 1.0
        if lo < hi:
            return (h >= lo) & (h <= hi)
        return (h >= lo) | (h <= hi)

    masks["yellow"] = (hue_band(1.0 / 6.0, 0.06)) & (s >= 0.25) & (v >= 0.35)
    masks["red"] = ((hue_band(0.0, 0.06) | hue_band(1.0, 1.0))) & (s >= 0.25) & (v >= 0.35)
    masks["green"] = hue_band(1.0 / 3.0, 0.06) & (s >= 0.25) & (v >= 0.25)
    masks["blue"] = hue_band(2.0 / 3.0, 0.06) & (s >= 0.25) & (v >= 0.25)
    return masks


def _color_fracs(rgb: Optional[Array], boxes: Array, masks: Optional[List[Array]]) -> Optional[Dict[str, Array]]:
    if rgb is None or boxes.size == 0:
        return None
    H, W = rgb.shape[:2]
    h, s, v = _rgb_to_hsv01(rgb)
    cm = _color_masks(h, s, v)
    out: Dict[str, Array] = {k: np.zeros((boxes.shape[0],), dtype=float) for k in cm.keys()}

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].astype(int)
        x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        if masks is not None and i < len(masks) and masks[i] is not None:
            mm = (masks[i].astype(np.uint8) > 0)
            m_crop = mm[y1:y2, x1:x2]
            denom = float(m_crop.sum())
            if denom < 1.0:
                denom = float((y2 - y1) * (x2 - x1))
                m_crop = np.ones((y2 - y1, x2 - x1), dtype=bool)
        else:
            m_crop = np.ones((y2 - y1, x2 - x1), dtype=bool)
            denom = float(m_crop.size)
        for k, k_mask in cm.items():
            region = k_mask[y1:y2, x1:x2]
            out[k][i] = float((region & m_crop).sum()) / max(denom, 1.0)
    return out


def _mask_geometry_stats(masks: Optional[List[Array]], boxes: Array) -> Dict[str, Array]:
    N = boxes.shape[0]
    per_box = np.zeros((N,), dtype=float)
    circ = np.zeros((N,), dtype=float)
    sol = np.zeros((N,), dtype=float)
    holes = np.zeros((N,), dtype=float)
    mar = np.zeros((N,), dtype=float)
    if masks is None or len(masks) == 0 or boxes.size == 0:
        return {
            "mask_per_box": per_box,
            "mask_circularity": circ,
            "mask_solidity": sol,
            "mask_holes_frac": holes,
            "mask_ar": mar,
        }
    for i in range(N):
        if i >= len(masks) or masks[i] is None:
            continue
        mb = (masks[i].astype(np.uint8) > 0).astype(np.uint8)
        area = float(mb.sum())
        ys, xs = np.nonzero(mb)
        if ys.size == 0:
            continue
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        bw = float(x2 - x1 + 1)
        bh = float(y2 - y1 + 1)
        mar[i] = (max(bw, bh) / max(min(bw, bh), 1.0))
        bx1, by1, bx2, by2 = boxes[i]
        bb_area = max((bx2 - bx1 + 1.0) * (by2 - by1 + 1.0), 1.0)
        per_box[i] = area / bb_area
        cnts, hier = cv2.findContours(mb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(c, True) for c in cnts) if cnts else 0.0
        circ[i] = (4.0 * np.pi * area) / max(perim * perim, 1e-9) if perim > 0 else 0.0
        hull_area = 0.0
        for c in cnts or []:
            hull = cv2.convexHull(c)
            hull_area += float(cv2.contourArea(hull))
        sol[i] = area / max(hull_area, 1e-9) if hull_area > 0 else 0.0
        holes_area = 0.0
        if hier is not None and len(hier) > 0:
            hinfo = hier[0]
            for j, (_, _, _, parent) in enumerate(hinfo):
                if parent != -1:
                    holes_area += float(cv2.contourArea(cnts[j]))
        holes[i] = holes_area / max(area, 1.0)
    return {
        "mask_per_box": np.clip(per_box, 0.0, 1.0),
        "mask_circularity": np.clip(circ, 0.0, 1.0),
        "mask_solidity": np.clip(sol, 0.0, 1.0),
        "mask_holes_frac": np.clip(holes, 0.0, 1.0),
        "mask_ar": mar,
    }


class FeatureExtractor:
    def compute(
        self,
        H: int,
        W: int,
        boxes: Array,
        scores: Array,
        labels: Array,
        masks: Optional[List[Array]],
        rgb: Optional[Array] = None,
    ) -> Features:
        N = len(labels)

        mlist: Optional[List[Array]] = None
        if masks is not None:
            mlist = list(masks)
            if len(mlist) < N:
                mlist = mlist + [None] * (N - len(mlist))
            elif len(mlist) > N:
                mlist = mlist[:N]

        geom = _box_features(H, W, boxes)
        score = scores.astype(float)
        mask_stats = _mask_geometry_stats(mlist, boxes)
        color = _color_fracs(rgb, boxes, mlist)

        def _fill(arr: Optional[Array]) -> Array:
            if arr is None or arr.shape[0] != N:
                return np.zeros((N,), dtype=float)
            return arr.astype(float)

        return Features(
            box_area_frac=_fill(geom["box_area_frac"]),
            short_side_frac=_fill(geom["short_side_frac"]),
            aspect_ratio=_fill(geom["aspect_ratio"]),
            score=score,
            labels=labels,
            boxes=boxes,
            masks=mlist,
            mask_per_box=_fill(mask_stats["mask_per_box"]),
            mask_circularity=_fill(mask_stats["mask_circularity"]),
            mask_solidity=_fill(mask_stats["mask_solidity"]),
            mask_holes_frac=_fill(mask_stats["mask_holes_frac"]),
            mask_ar=_fill(mask_stats["mask_ar"]),
            white_frac=_fill(None if color is None else color.get("white")),
            black_frac=_fill(None if color is None else color.get("black")),
            yellow_frac=_fill(None if color is None else color.get("yellow")),
            red_frac=_fill(None if color is None else color.get("red")),
            green_frac=_fill(None if color is None else color.get("green")),
            blue_frac=_fill(None if color is None else color.get("blue")),
        )
