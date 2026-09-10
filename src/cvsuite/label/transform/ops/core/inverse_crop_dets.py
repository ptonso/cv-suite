from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from cvsuite.common.core import BBox, ImageRecord, Polygon, Record, VisionDataset
from cvsuite.label.core.crops import (
    IntRect,
    Rect,
    bbox_crop_xyxy,
    clamp_crop_xyxy,
    expand_rect_xyxy,
    resolve_image_path,
)
from cvsuite.label.transform.ops.core.crop_dets import CROP_DETS_ATTR, CVSUITE_ATTR, CropRef, parse_crop_id


@dataclass(frozen=True)
class CropApplication:
    crop_id: str
    crop_path: Path
    record_index: int
    source_kind: str
    source_index: int
    source_xyxy: Rect
    crop_xyxy: IntRect
    pad_type: str | None
    flags: dict[str, bool]


def _metadata(annotation: BBox | Polygon) -> dict[str, Any] | None:
    attrs = annotation.attributes if isinstance(annotation.attributes, dict) else {}
    ns = attrs.get(CVSUITE_ATTR)
    if not isinstance(ns, dict):
        return None
    meta = ns.get(CROP_DETS_ATTR)
    return meta if isinstance(meta, dict) else None


def _flags(annotation: BBox | Polygon) -> dict[str, bool]:
    raw = annotation.attributes.get("flags", {}) if isinstance(annotation.attributes, dict) else {}
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("annotation.attributes['flags'] must be a dict[str, bool].")
    if any(not isinstance(key, str) or not isinstance(value, bool) for key, value in raw.items()):
        raise TypeError("annotation.attributes['flags'] must be a dict[str, bool].")
    return dict(raw)


def _crop_anchor(rec: Record) -> tuple[BBox | Polygon, dict[str, Any]]:
    matches: list[tuple[BBox | Polygon, dict[str, Any]]] = []
    for annotation in [*rec.boxes, *rec.polys]:
        meta = _metadata(annotation)
        if meta is not None:
            matches.append((annotation, meta))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one crop-dets anchor in modified crop: {rec.image.path}")
    return matches[0]




def _margin_frac(meta: dict[str, Any], crop_id: str) -> float:
    if "margin_frac" not in meta:
        raise ValueError(f"Crop {crop_id!r} is missing required margin_frac metadata.")
    raw = meta["margin_frac"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"Crop {crop_id!r} margin_frac must be a number.")
    margin_frac = float(raw)
    if margin_frac < 0:
        raise ValueError(f"Crop {crop_id!r} margin_frac must be non-negative.")
    return margin_frac


def _pad_type(meta: dict[str, Any], crop_id: str) -> str | None:
    raw = meta.get("pad_type")
    if raw is None:
        return None
    if raw != "background":
        raise ValueError(f"Crop {crop_id!r} pad_type may only be 'background' when present.")
    return "background"


def _record_keys(rec: Record, root: Path | None) -> set[str]:
    path = Path(rec.image.path)
    abs_path = resolve_image_path(path, root).resolve()
    keys = {str(abs_path), abs_path.as_posix(), path.as_posix()}
    if root is not None:
        try:
            keys.add(abs_path.relative_to(root.resolve()).as_posix())
        except ValueError:
            pass
    return keys


def _record_index(ds: VisionDataset, ref: CropRef, key_map: dict[tuple[str, str], list[int]]) -> int:
    candidates = set(key_map.get((ref.split, ref.image_ref), []))
    if len(candidates) != 1:
        raise ValueError(
            f"Could not resolve crop source {ref.split}:{ref.image_ref!r} to exactly one original record."
        )
    return next(iter(candidates))


def _crop_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(str(path))
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return image[:, :, :3]
    if image.shape[2] != 3:
        raise ValueError(f"Unsupported crop image channel count: {path}")
    return image


def _source_geometry(rec: Record, ref: CropRef, margin_frac: float, crop_id: str) -> tuple[Rect, IntRect]:
    if ref.source_kind == "box":
        if not (0 <= ref.source_index < len(rec.boxes)):
            raise IndexError(f"Crop {crop_id!r} source box index is out of range.")
        box = rec.boxes[ref.source_index]
        if box.kind == "pose":
            raise ValueError(f"Crop {crop_id!r} points at a pose box, which crop-dets does not support.")
        source_xyxy, crop_xyxy = bbox_crop_xyxy(
            box,
            img_w=rec.image.width,
            img_h=rec.image.height,
            margin_frac=margin_frac,
        )
    elif ref.source_kind == "polygon":
        if not (0 <= ref.source_index < len(rec.polys)):
            raise IndexError(f"Crop {crop_id!r} source polygon index is out of range.")
        source_xyxy = _polygon_bbox(rec.polys[ref.source_index], img_w=rec.image.width, img_h=rec.image.height)
        crop_xyxy = clamp_crop_xyxy(
            *expand_rect_xyxy(source_xyxy, margin_frac=margin_frac),
            img_w=rec.image.width,
            img_h=rec.image.height,
        )
    else:
        raise ValueError(f"Unsupported crop source kind: {ref.source_kind!r}")
    if crop_xyxy is None:
        raise ValueError(f"Crop {crop_id!r} recomputed to an empty source crop.")
    return source_xyxy, crop_xyxy


def _applications(original: VisionDataset, crops: VisionDataset) -> list[CropApplication]:
    key_map: dict[tuple[str, str], list[int]] = {}
    for idx, rec in enumerate(original.records):
        split = rec.split or "train"
        for key in _record_keys(rec, original.root):
            key_map.setdefault((split, key), []).append(idx)

    seen: dict[str, Path] = {}
    out: list[CropApplication] = []
    for rec in crops.records:
        anchor, meta = _crop_anchor(rec)
        crop_id = str(meta.get("crop_id") or "")
        if not crop_id:
            raise ValueError(f"Missing crop_id in modified crop: {rec.image.path}")
        crop_path = resolve_image_path(Path(rec.image.path), crops.root).resolve()
        if crop_id in seen:
            raise ValueError(f"Duplicate crop_id {crop_id!r}: {seen[crop_id]} and {crop_path}")
        seen[crop_id] = crop_path
        ref = parse_crop_id(crop_id)
        record_index = _record_index(original, ref, key_map)
        source_xyxy, crop_xyxy = _source_geometry(
            original.records[record_index],
            ref,
            _margin_frac(meta, crop_id),
            crop_id,
        )

        out.append(
            CropApplication(
                crop_id=crop_id,
                crop_path=crop_path,
                record_index=record_index,
                source_kind=ref.source_kind,
                source_index=ref.source_index,
                source_xyxy=source_xyxy,
                crop_xyxy=crop_xyxy,
                pad_type=_pad_type(meta, crop_id),
                flags=_flags(anchor),
            )
        )
    return out


def _bbox_xyxy(box: BBox, *, img_w: int, img_h: int) -> Rect:
    w = box.w * img_w
    h = box.h * img_h
    cx = box.cx * img_w
    cy = box.cy * img_h
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def _polygon_bbox(poly: Polygon, *, img_w: int, img_h: int) -> Rect:
    xs = [x * img_w for x, _ in poly.points]
    ys = [y * img_h for _, y in poly.points]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a: Rect, b: Rect) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _best_by_iou(
    items: Sequence[BBox | Polygon],
    source_xyxy: Rect,
    *,
    img_w: int,
    img_h: int,
) -> BBox | Polygon | None:
    best: BBox | Polygon | None = None
    best_iou = 0.0
    for item in items:
        xyxy = (
            _bbox_xyxy(item, img_w=img_w, img_h=img_h)
            if isinstance(item, BBox)
            else _polygon_bbox(item, img_w=img_w, img_h=img_h)
        )
        iou = _iou(source_xyxy, xyxy)
        if iou > best_iou:
            best = item
            best_iou = iou
    return best


def _matched_counterpart(rec: Record, anchor: BBox | Polygon, app: CropApplication) -> BBox | Polygon | None:
    if isinstance(anchor, BBox):
        candidates = [poly for poly in rec.polys if poly.cls == anchor.cls]
    else:
        candidates = [box for box in rec.boxes if box.cls == anchor.cls and box.kind != "pose"]
    exact = [
        item for item in candidates
        if (anchor.id is not None and item.id == anchor.id)
        or (anchor.group_id is not None and item.group_id == anchor.group_id)
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        candidates = exact
    return _best_by_iou(candidates, app.source_xyxy, img_w=rec.image.width, img_h=rec.image.height)


def _merge_flags(annotation: BBox | Polygon, flags: dict[str, bool]) -> None:
    if not flags:
        return
    attrs = dict(annotation.attributes or {})
    current = attrs.get("flags", {})
    if current is None:
        current = {}
    if not isinstance(current, dict):
        raise TypeError("annotation.attributes['flags'] must be a dict[str, bool].")
    attrs["flags"] = {**current, **flags}
    annotation.attributes = attrs


def _apply_flags(rec: Record, app: CropApplication) -> None:
    if app.source_kind == "box":
        if not (0 <= app.source_index < len(rec.boxes)):
            raise IndexError(f"Crop {app.crop_id!r} source box index is out of range.")
        anchor: BBox | Polygon = rec.boxes[app.source_index]
    elif app.source_kind == "polygon":
        if not (0 <= app.source_index < len(rec.polys)):
            raise IndexError(f"Crop {app.crop_id!r} source polygon index is out of range.")
        anchor = rec.polys[app.source_index]
    else:
        raise ValueError(f"Unsupported crop source_kind: {app.source_kind!r}")

    _merge_flags(anchor, app.flags)
    counterpart = _matched_counterpart(rec, anchor, app)
    if counterpart is not None:
        _merge_flags(counterpart, app.flags)


def _background_source_xyxy(crop_xyxy: IntRect) -> Rect:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
    crop_w = max(1, crop_x2 - crop_x1)
    crop_h = max(1, crop_y2 - crop_y1)
    side = max(crop_w, crop_h)
    pad_left = max(0, (side - crop_w) // 2)
    pad_right = max(0, side - crop_w - pad_left)
    pad_top = max(0, (side - crop_h) // 2)
    pad_bottom = max(0, side - crop_h - pad_top)
    return (
        float(crop_x1 - pad_left),
        float(crop_y1 - pad_top),
        float(crop_x2 + pad_right),
        float(crop_y2 + pad_bottom),
    )


def _paste_xyxy(app: CropApplication) -> Rect:
    if app.pad_type == "background":
        return _background_source_xyxy(app.crop_xyxy)
    x1, y1, x2, y2 = app.crop_xyxy
    return float(x1), float(y1), float(x2), float(y2)


def _paste_area(app: CropApplication) -> float:
    x1, y1, x2, y2 = _paste_xyxy(app)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _render_mapping(crop: np.ndarray, app: CropApplication) -> tuple[Rect, float, float, float, float]:
    crop_x1, crop_y1, crop_x2, crop_y2 = app.crop_xyxy
    crop_w = max(1, crop_x2 - crop_x1)
    crop_h = max(1, crop_y2 - crop_y1)
    out_h, out_w = crop.shape[:2]
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"Crop {app.crop_id!r} image has empty dimensions.")
    if app.pad_type == "background":
        side = max(crop_w, crop_h)
        return _background_source_xyxy(app.crop_xyxy), out_w / side, out_h / side, 0.0, 0.0

    scale = min(out_w / crop_w, out_h / crop_h)
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    pad_x = max(0, (out_w - resized_w) // 2)
    pad_y = max(0, (out_h - resized_h) // 2)
    return (
        (float(crop_x1), float(crop_y1), float(crop_x2), float(crop_y2)),
        resized_w / crop_w,
        resized_h / crop_h,
        float(pad_x),
        float(pad_y),
    )


def _paste(image: np.ndarray, app: CropApplication) -> None:
    crop = _crop_image(app.crop_path)
    img_h, img_w = image.shape[:2]
    render_source_xyxy, scale_x, scale_y, pad_x, pad_y = _render_mapping(crop, app)
    src_x1, src_y1, src_x2, src_y2 = render_source_xyxy
    dst_x1 = max(0, int(np.floor(src_x1)))
    dst_y1 = max(0, int(np.floor(src_y1)))
    dst_x2 = min(img_w, int(np.ceil(src_x2)))
    dst_y2 = min(img_h, int(np.ceil(src_y2)))
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        raise ValueError(f"Crop {app.crop_id!r} has no pasteable area in the source image.")

    crop_x1 = int(round(pad_x + (dst_x1 - src_x1) * scale_x))
    crop_y1 = int(round(pad_y + (dst_y1 - src_y1) * scale_y))
    crop_x2 = int(round(pad_x + (dst_x2 - src_x1) * scale_x))
    crop_y2 = int(round(pad_y + (dst_y2 - src_y1) * scale_y))
    crop_x1 = max(0, min(crop.shape[1], crop_x1))
    crop_y1 = max(0, min(crop.shape[0], crop_y1))
    crop_x2 = max(0, min(crop.shape[1], crop_x2))
    crop_y2 = max(0, min(crop.shape[0], crop_y2))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError(f"Crop {app.crop_id!r} has no mapped crop pixels.")

    patch = crop[crop_y1:crop_y2, crop_x1:crop_x2]
    patch = cv2.resize(patch, (dst_x2 - dst_x1, dst_y2 - dst_y1), interpolation=cv2.INTER_LINEAR)
    image[dst_y1:dst_y2, dst_x1:dst_x2] = patch


def _set_provenance(rec: Record, apps: Sequence[CropApplication]) -> None:
    attrs = dict(rec.attributes or {})
    raw_ns = attrs.get(CVSUITE_ATTR)
    if raw_ns is not None and not isinstance(raw_ns, dict):
        raise TypeError(f"record.attributes['{CVSUITE_ATTR}'] must be a dict.")
    ns = dict(raw_ns or {})
    ns["inverse_crop_dets"] = {
        "applied": [
            {
                "crop_id": app.crop_id,
                "crop_path": str(app.crop_path),
                "flags": dict(app.flags),
            }
            for app in apps
        ]
    }
    attrs[CVSUITE_ATTR] = ns
    rec.attributes = attrs


def inverse_crop_dets_dataset(original: VisionDataset, modified_crops: VisionDataset) -> VisionDataset:
    apps = _applications(original, modified_crops)
    apps_by_record: dict[int, list[CropApplication]] = {}
    for app in apps:
        apps_by_record.setdefault(app.record_index, []).append(app)

    tmp_root = Path(tempfile.mkdtemp(prefix="vislabel_inverse_crop_"))
    out_records: list[Record] = []
    for idx, rec in enumerate(original.records):
        out_rec = deepcopy(rec)
        src_path = resolve_image_path(Path(rec.image.path), original.root).resolve()
        rec_apps = sorted(apps_by_record.get(idx, []), key=lambda item: (-_paste_area(item), item.crop_id))
        if not rec_apps:
            out_rec.image.path = src_path
            out_records.append(out_rec)
            continue

        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(str(src_path))
        for app in rec_apps:
            _paste(image, app)
            _apply_flags(out_rec, app)

        split = out_rec.split or "train"
        dst = tmp_root / split / src_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst), image):
            raise RuntimeError(f"Failed to write inverse crop image: {dst}")
        out_rec.image = ImageRecord(path=dst, width=rec.image.width, height=rec.image.height)
        _set_provenance(out_rec, rec_apps)
        out_records.append(out_rec)

    meta = dict(original.meta)
    meta.update(
        {
            "transform": "inverse-crop-dets",
            "modified_crop_count": len(apps),
            "modified_record_count": len(apps_by_record),
            "inverse_crop_root": tmp_root,
        }
    )
    return VisionDataset(
        records=out_records,
        classes=list(original.classes),
        task=original.task,
        root=tmp_root,
        meta=meta,
        fm_request=original.fm_request,
    )
