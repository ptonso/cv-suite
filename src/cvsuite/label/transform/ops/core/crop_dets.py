from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import List, Literal, Sequence, Tuple

import cv2
import numpy as np

from cvsuite.common.core import BBox, Classification, ImageRecord, Polygon, Record
from cvsuite.common.core import VisionDataset
from cvsuite.label.core.crops import (
    IntRect,
    Rect,
    annotation_label,
    bbox_crop_xyxy,
    clamp_crop_xyxy,
    expand_rect_xyxy,
    next_output_path,
    resolve_image_path,
    sanitize_label,
)

Point = Tuple[float, float]
PadType = Literal["background", "black", "gray", "white"]
CVSUITE_ATTR = "cvsuite"
CROP_DETS_ATTR = "crop_dets"
PAD_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "gray": (127, 127, 127),
    "white": (255, 255, 255),
}


@dataclass(frozen=True)
class CropAnchor:
    cls: int
    label: str | None
    score: float | None
    source_kind: str
    source_box_index: int
    source_xyxy: Rect
    crop_xyxy: IntRect
    box: BBox | None = None
    polygon_index: int | None = None


@dataclass(frozen=True)
class CropRenderSpec:
    source_xyxy: Rect
    out_w: int
    out_h: int
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float


def _source_image_ref(rec: Record, img_path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return img_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return rec.image.path.as_posix()


@dataclass(frozen=True)
class CropRef:
    split: str
    image_ref: str
    source_kind: str
    source_index: int


def build_crop_id(split: str, source_image_ref: str, source_kind: str, source_index: int) -> str:
    return f"{split}:{source_image_ref}:{source_kind}:{source_index}"


def parse_crop_id(crop_id: str) -> CropRef:
    try:
        prefix, source_kind, raw_index = crop_id.rsplit(":", 2)
        split, image_ref = prefix.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            f"crop_id {crop_id!r} must have format '<split>:<image_ref>:<box|polygon>:<index>'."
        ) from exc
    if not split or not image_ref:
        raise ValueError(f"crop_id {crop_id!r} has an empty split or image ref.")
    if source_kind not in {"box", "polygon"}:
        raise ValueError(f"crop_id {crop_id!r} has unsupported source kind {source_kind!r}.")
    try:
        source_index = int(raw_index)
    except ValueError as exc:
        raise ValueError(f"crop_id {crop_id!r} source index must be an integer.") from exc
    if source_index < 0:
        raise ValueError(f"crop_id {crop_id!r} source index must be non-negative.")
    return CropRef(split=split, image_ref=image_ref, source_kind=source_kind, source_index=source_index)


def _crop_metadata(
    *,
    rec: Record,
    root: Path | None,
    img_path: Path,
    split: str,
    anchor: CropAnchor,
    imgsz: int | None,
    pad_type: PadType,
    margin_frac: float,
) -> dict[str, object]:
    source_ref = _source_image_ref(rec, img_path, root)
    meta: dict[str, object] = {
        "crop_id": build_crop_id(split, source_ref, anchor.source_kind, anchor.source_box_index),
        "margin_frac": margin_frac,
    }
    if imgsz is not None and pad_type == "background":
        meta["pad_type"] = "background"
    return meta


def _set_crop_metadata(box: BBox, metadata: dict[str, object]) -> BBox:
    attrs = dict(box.attributes or {})
    raw_ns = attrs.get(CVSUITE_ATTR)
    if raw_ns is not None and not isinstance(raw_ns, dict):
        raise TypeError(f"annotation.attributes['{CVSUITE_ATTR}'] must be a dict.")
    ns = dict(raw_ns or {})
    ns[CROP_DETS_ATTR] = metadata
    attrs[CVSUITE_ATTR] = ns
    box.attributes = attrs
    return box


def _polygon_bbox_xyxy(poly: Polygon, *, img_w: int, img_h: int) -> Rect:
    xs = [x * img_w for x, _ in poly.points]
    ys = [y * img_h for _, y in poly.points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(a: Rect, b: Rect) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return inter / (area_a + area_b - inter + 1e-9)


def _anchors_for_record(
    rec: Record,
    classes: Sequence[str],
    *,
    margin_frac: float = 0.0,
) -> tuple[List[CropAnchor], int]:
    anchors: List[CropAnchor] = []
    skipped_invalid = 0
    img_w = rec.image.width
    img_h = rec.image.height

    for idx, box in enumerate(rec.boxes):
        if box.kind == "pose":
            continue
        source_xyxy, crop_xyxy = bbox_crop_xyxy(
            box,
            img_w=img_w,
            img_h=img_h,
            margin_frac=margin_frac,
        )
        if crop_xyxy is None:
            skipped_invalid += 1
            continue
        anchors.append(
            CropAnchor(
                cls=box.cls,
                label=annotation_label(box.cls, classes, box.label),
                score=box.score,
                source_kind="box",
                source_box_index=idx,
                source_xyxy=source_xyxy,
                crop_xyxy=crop_xyxy,
                box=box,
            )
        )

    if anchors or not rec.polys:
        return anchors, skipped_invalid

    for idx, poly in enumerate(rec.polys):
        source_xyxy = _polygon_bbox_xyxy(poly, img_w=img_w, img_h=img_h)
        crop_xyxy = clamp_crop_xyxy(
            *expand_rect_xyxy(source_xyxy, margin_frac=margin_frac),
            img_w=img_w,
            img_h=img_h,
        )
        if crop_xyxy is None:
            skipped_invalid += 1
            continue
        anchors.append(
            CropAnchor(
                cls=poly.cls,
                label=annotation_label(poly.cls, classes, poly.label),
                score=poly.score,
                source_kind="polygon",
                source_box_index=idx,
                source_xyxy=source_xyxy,
                crop_xyxy=crop_xyxy,
                polygon_index=idx,
            )
        )

    return anchors, skipped_invalid


def _match_polygon(rec: Record, anchor: CropAnchor) -> Polygon | None:
    if anchor.polygon_index is not None and 0 <= anchor.polygon_index < len(rec.polys):
        return rec.polys[anchor.polygon_index]

    polys = [poly for poly in rec.polys if poly.cls == anchor.cls]
    if not polys:
        return None

    if anchor.box is not None:
        exact = []
        for poly in polys:
            same_id = anchor.box.id is not None and poly.id == anchor.box.id
            same_group = anchor.box.group_id is not None and poly.group_id == anchor.box.group_id
            if same_id or same_group:
                exact.append(poly)
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            polys = exact

    best_poly: Polygon | None = None
    best_iou = -1.0
    for poly in polys:
        poly_bbox = _polygon_bbox_xyxy(poly, img_w=rec.image.width, img_h=rec.image.height)
        iou = _bbox_iou(anchor.source_xyxy, poly_bbox)
        if iou > best_iou:
            best_iou = iou
            best_poly = poly
    if best_poly is None:
        return None
    if best_iou > 0 or (anchor.box is not None and polys and len(polys) == 1):
        return best_poly
    return None


def _near(a: Point, b: Point, eps: float = 1e-6) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def _dedupe_points(points: Sequence[Point]) -> List[Point]:
    out: List[Point] = []
    for pt in points:
        if out and _near(out[-1], pt):
            continue
        out.append(pt)
    if len(out) >= 2 and _near(out[0], out[-1]):
        out.pop()
    return out


def _clip_against_edge(
    points: Sequence[Point],
    *,
    inside,
    intersect,
) -> List[Point]:
    if not points:
        return []

    out: List[Point] = []
    prev = points[-1]
    prev_inside = inside(prev)
    for curr in points:
        curr_inside = inside(curr)
        if curr_inside:
            if not prev_inside:
                out.append(intersect(prev, curr))
            out.append(curr)
        elif prev_inside:
            out.append(intersect(prev, curr))
        prev = curr
        prev_inside = curr_inside
    return out


def _line_intersection_at_x(p1: Point, p2: Point, x_edge: float) -> Point:
    x1, y1 = p1
    x2, y2 = p2
    if abs(x2 - x1) <= 1e-9:
        return x_edge, y1
    t = (x_edge - x1) / (x2 - x1)
    return x_edge, y1 + t * (y2 - y1)


def _line_intersection_at_y(p1: Point, p2: Point, y_edge: float) -> Point:
    x1, y1 = p1
    x2, y2 = p2
    if abs(y2 - y1) <= 1e-9:
        return x1, y_edge
    t = (y_edge - y1) / (y2 - y1)
    return x1 + t * (x2 - x1), y_edge


def _clip_polygon_to_rect(points: Sequence[Point], rect: IntRect) -> List[Point]:
    x1, y1, x2, y2 = rect
    clipped = list(points)
    clipped = _clip_against_edge(
        clipped,
        inside=lambda pt: pt[0] >= x1,
        intersect=lambda p1, p2: _line_intersection_at_x(p1, p2, x1),
    )
    clipped = _clip_against_edge(
        clipped,
        inside=lambda pt: pt[0] <= x2,
        intersect=lambda p1, p2: _line_intersection_at_x(p1, p2, x2),
    )
    clipped = _clip_against_edge(
        clipped,
        inside=lambda pt: pt[1] >= y1,
        intersect=lambda p1, p2: _line_intersection_at_y(p1, p2, y1),
    )
    clipped = _clip_against_edge(
        clipped,
        inside=lambda pt: pt[1] <= y2,
        intersect=lambda p1, p2: _line_intersection_at_y(p1, p2, y2),
    )
    return _dedupe_points(clipped)


def _polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _choose_interpolation(src_w: int, src_h: int, dst_w: int, dst_h: int) -> int:
    if dst_w < src_w or dst_h < src_h:
        return cv2.INTER_AREA
    return cv2.INTER_LINEAR


def _resize_exact(image: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    if image.shape[1] == dst_w and image.shape[0] == dst_h:
        return image
    interpolation = _choose_interpolation(image.shape[1], image.shape[0], dst_w, dst_h)
    return cv2.resize(image, (dst_w, dst_h), interpolation=interpolation)


def _background_square_crop(image: np.ndarray, crop_xyxy: IntRect) -> tuple[np.ndarray, Rect]:
    img_h, img_w = image.shape[:2]
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
    crop_w = max(1, crop_x2 - crop_x1)
    crop_h = max(1, crop_y2 - crop_y1)
    side = max(crop_w, crop_h)

    pad_left = max(0, (side - crop_w) // 2)
    pad_right = max(0, side - crop_w - pad_left)
    pad_top = max(0, (side - crop_h) // 2)
    pad_bottom = max(0, side - crop_h - pad_top)

    left = crop_x1 - pad_left
    top = crop_y1 - pad_top
    right = crop_x2 + pad_right
    bottom = crop_y2 + pad_bottom

    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(img_w, right)
    src_bottom = min(img_h, bottom)
    bg = image[src_top:src_bottom, src_left:src_right]
    if bg.size == 0:
        empty_shape = (side, side) if image.ndim == 2 else (side, side, image.shape[2])
        return np.zeros(empty_shape, dtype=image.dtype), (float(left), float(top), float(right), float(bottom))

    miss_left = max(0, src_left - left)
    miss_top = max(0, src_top - top)
    miss_right = max(0, right - src_right)
    miss_bottom = max(0, bottom - src_bottom)
    if miss_left or miss_top or miss_right or miss_bottom:
        bg = cv2.copyMakeBorder(
            bg,
            miss_top,
            miss_bottom,
            miss_left,
            miss_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    return bg, (float(left), float(top), float(right), float(bottom))


def _render_crop(
    image: np.ndarray,
    crop_xyxy: IntRect,
    *,
    imgsz: int | None = None,
    long_size: int | None = None,
    pad_type: PadType = "black",
) -> tuple[np.ndarray, CropRenderSpec] | None:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    if crop_w <= 0 or crop_h <= 0:
        return None

    if imgsz is not None:
        if pad_type == "background":
            bg_crop, source_xyxy = _background_square_crop(image, crop_xyxy)
            canvas = _resize_exact(bg_crop, imgsz, imgsz)
            side = max(1, bg_crop.shape[0])
            return canvas, CropRenderSpec(
                source_xyxy=source_xyxy,
                out_w=imgsz,
                out_h=imgsz,
                scale_x=float(imgsz) / float(side),
                scale_y=float(imgsz) / float(side),
                pad_x=0.0,
                pad_y=0.0,
            )

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    crop_h, crop_w = crop.shape[:2]
    if imgsz is not None:
        scale = float(imgsz) / float(max(crop_w, crop_h))
        resized_w = max(1, int(round(crop_w * scale)))
        resized_h = max(1, int(round(crop_h * scale)))
        resized = _resize_exact(crop, resized_w, resized_h)

        canvas = np.full((imgsz, imgsz, crop.shape[2]), PAD_COLORS[pad_type], dtype=crop.dtype)
        pad_x = max(0, (imgsz - resized_w) // 2)
        pad_y = max(0, (imgsz - resized_h) // 2)
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        return canvas, CropRenderSpec(
            source_xyxy=(float(crop_x1), float(crop_y1), float(crop_x2), float(crop_y2)),
            out_w=imgsz,
            out_h=imgsz,
            scale_x=float(resized_w) / float(crop_w),
            scale_y=float(resized_h) / float(crop_h),
            pad_x=float(pad_x),
            pad_y=float(pad_y),
        )

    if long_size is not None:
        scale = float(long_size) / float(max(crop_w, crop_h))
        resized_w = max(1, int(round(crop_w * scale)))
        resized_h = max(1, int(round(crop_h * scale)))
        resized = _resize_exact(crop, resized_w, resized_h)
        return resized, CropRenderSpec(
            source_xyxy=(float(crop_x1), float(crop_y1), float(crop_x2), float(crop_y2)),
            out_w=resized_w,
            out_h=resized_h,
            scale_x=float(resized_w) / float(crop_w),
            scale_y=float(resized_h) / float(crop_h),
            pad_x=0.0,
            pad_y=0.0,
        )

    return crop, CropRenderSpec(
        source_xyxy=(float(crop_x1), float(crop_y1), float(crop_x2), float(crop_y2)),
        out_w=crop_w,
        out_h=crop_h,
        scale_x=1.0,
        scale_y=1.0,
        pad_x=0.0,
        pad_y=0.0,
    )


def _transform_xyxy_to_canvas(xyxy: Rect, spec: CropRenderSpec) -> Rect | None:
    crop_x1, crop_y1, _crop_x2, _crop_y2 = spec.source_xyxy
    src_x1, src_y1, src_x2, src_y2 = xyxy

    out_x1 = spec.pad_x + (src_x1 - crop_x1) * spec.scale_x
    out_y1 = spec.pad_y + (src_y1 - crop_y1) * spec.scale_y
    out_x2 = spec.pad_x + (src_x2 - crop_x1) * spec.scale_x
    out_y2 = spec.pad_y + (src_y2 - crop_y1) * spec.scale_y

    out_x1 = min(float(spec.out_w), max(0.0, out_x1))
    out_y1 = min(float(spec.out_h), max(0.0, out_y1))
    out_x2 = min(float(spec.out_w), max(0.0, out_x2))
    out_y2 = min(float(spec.out_h), max(0.0, out_y2))
    if out_x2 <= out_x1 or out_y2 <= out_y1:
        return None
    return out_x1, out_y1, out_x2, out_y2


def _anchor_box(anchor: CropAnchor, classes: Sequence[str], spec: CropRenderSpec) -> BBox | None:
    out_xyxy = _transform_xyxy_to_canvas(anchor.source_xyxy, spec)
    if out_xyxy is None:
        return None

    out_x1, out_y1, out_x2, out_y2 = out_xyxy
    out_w = out_x2 - out_x1
    out_h = out_y2 - out_y1
    src = anchor.box
    label = annotation_label(anchor.cls, classes, anchor.label)
    return BBox(
        cx=((out_x1 + out_x2) / 2.0) / spec.out_w,
        cy=((out_y1 + out_y2) / 2.0) / spec.out_h,
        w=out_w / spec.out_w,
        h=out_h / spec.out_h,
        cls=anchor.cls,
        label=label,
        score=anchor.score,
        id=None if src is None else src.id,
        group_id=None if src is None else src.group_id,
        kind="det",
        text=None if src is None else src.text,
        model=None if src is None else src.model,
        prompt=None if src is None else src.prompt,
        attributes={} if src is None else dict(src.attributes or {}),
    )


def _transform_polygon(
    poly: Polygon,
    spec: CropRenderSpec,
    *,
    img_w: int,
    img_h: int,
) -> Polygon | None:
    crop_x1, crop_y1, crop_x2, crop_y2 = spec.source_xyxy
    if spec.out_w <= 0 or spec.out_h <= 0:
        return None

    abs_points = [(x * img_w, y * img_h) for x, y in poly.points]
    clipped = _clip_polygon_to_rect(
        abs_points,
        (
            int(np.floor(crop_x1)),
            int(np.floor(crop_y1)),
            int(np.ceil(crop_x2)),
            int(np.ceil(crop_y2)),
        ),
    )
    if len(clipped) < 3 or abs(_polygon_area(clipped)) <= 1e-6:
        return None

    points: List[Point] = []
    for x, y in clipped:
        out_x = spec.pad_x + (x - crop_x1) * spec.scale_x
        out_y = spec.pad_y + (y - crop_y1) * spec.scale_y
        norm_x = min(1.0, max(0.0, out_x / spec.out_w))
        norm_y = min(1.0, max(0.0, out_y / spec.out_h))
        points.append((norm_x, norm_y))

    if len(points) < 3 or abs(_polygon_area(points)) <= 1e-6:
        return None

    return Polygon(
        points=points,
        cls=poly.cls,
        label=poly.label,
        id=poly.id,
        score=poly.score,
        group_id=poly.group_id,
        model=poly.model,
        prompt=poly.prompt,
        attributes=dict(poly.attributes or {}),
    )


def crop_dets_dataset(
    ds: VisionDataset,
    *,
    imgsz: int | None = None,
    long_size: int | None = None,
    pad_type: PadType = "black",
    margin_frac: float = 0.0,
) -> VisionDataset:
    if imgsz is not None and imgsz <= 0:
        raise ValueError("--imgsz must be a positive integer.")
    if long_size is not None and long_size <= 0:
        raise ValueError("--long-size must be a positive integer.")
    if imgsz is not None and long_size is not None:
        raise ValueError("Use at most one of --imgsz and --long-size.")
    if margin_frac < 0:
        raise ValueError("--margin-frac must be non-negative.")
    if pad_type not in {"background", "black", "gray", "white"}:
        raise ValueError("--pad-type must be one of background, black, gray, or white.")

    tmp_root = Path(tempfile.mkdtemp(prefix="vislabel_crop_"))
    crops_root = tmp_root / "crops"
    out_records: List[Record] = []
    skipped_invalid = 0

    for rec in ds.records:
        split = rec.split or "train"
        img_path = resolve_image_path(rec.image.path, ds.root)
        anchors, skipped = _anchors_for_record(rec, ds.classes, margin_frac=margin_frac)
        skipped_invalid += skipped
        if not anchors:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            skipped_invalid += len(anchors)
            continue

        for anchor in anchors:
            rendered = _render_crop(
                image,
                anchor.crop_xyxy,
                imgsz=imgsz,
                long_size=long_size,
                pad_type=pad_type,
            )
            if rendered is None:
                skipped_invalid += 1
                continue
            crop, spec = rendered

            out_box = _anchor_box(anchor, ds.classes, spec)
            if out_box is None:
                skipped_invalid += 1
                continue

            matched_poly = _match_polygon(rec, anchor)
            out_poly = None if matched_poly is None else _transform_polygon(
                matched_poly,
                spec,
                img_w=rec.image.width,
                img_h=rec.image.height,
            )

            label = annotation_label(anchor.cls, ds.classes, anchor.label)
            crop_meta = _crop_metadata(
                rec=rec,
                root=ds.root,
                img_path=img_path,
                split=split,
                anchor=anchor,
                imgsz=imgsz,
                pad_type=pad_type,
                margin_frac=margin_frac,
            )
            _set_crop_metadata(out_box, crop_meta)
            dst = crops_root / split / sanitize_label(label) / f"{img_path.stem}__obj{anchor.source_box_index}.jpg"
            dst = next_output_path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dst), crop):
                skipped_invalid += 1
                continue

            out_records.append(
                Record(
                    image=ImageRecord(path=dst, width=spec.out_w, height=spec.out_h),
                    split=split,
                    task=rec.task or ds.task,
                    boxes=[out_box],
                    polys=[] if out_poly is None else [out_poly],
                    labelme_shapes=[],
                    attributes={},
                    classification=Classification(label=label, score=anchor.score),
                )
            )

    meta = dict(ds.meta)
    meta.update(
        {
            "transform": "crop-dets",
            "source_root": ds.root,
            "crop_count": len(out_records),
            "crop_imgsz": imgsz,
            "crop_long_size": long_size,
            "crop_pad_type": pad_type if imgsz is not None else None,
            "crop_margin_frac": margin_frac,
            "skipped_invalid": skipped_invalid,
            "crop_root": tmp_root,
        }
    )
    return VisionDataset(
        records=out_records,
        classes=list(ds.classes),
        task=ds.task,
        root=tmp_root,
        meta=meta,
        fm_request=ds.fm_request,
    )
