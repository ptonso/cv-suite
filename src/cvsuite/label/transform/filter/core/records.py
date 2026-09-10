from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Optional, Tuple
import numpy as np

from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset

from .utils import bbox_xyxy_from_norm, polygon_bbox, polygon_to_mask, box_iou, load_rgb_safe

Array = np.ndarray
SourceRef = Tuple[str, Optional[int], Optional[int]]  # (source_kind, box_idx, poly_idx)


def _label_name(cls_idx: int, label_field: Optional[str], classes: List[str]) -> str:
    if label_field:
        return str(label_field)
    if 0 <= cls_idx < len(classes):
        return str(classes[cls_idx])
    return str(cls_idx)


def _poly_entries(rec: Record, classes: List[str], img_w: int, img_h: int) -> List[dict]:
    entries = []
    for idx, p in enumerate(rec.polys):
        pts = [(x * img_w, y * img_h) for x, y in p.points]
        if len(pts) < 3:
            continue
        mask = polygon_to_mask(pts, img_w=img_w, img_h=img_h)
        x1, y1, x2, y2 = polygon_bbox(pts)
        entries.append(
            {
                "idx": idx,
                "mask": mask,
                "bbox": np.array([x1, y1, x2, y2], dtype=float),
                "label": _label_name(p.cls, p.label, classes),
                "group_id": p.group_id,
                "score": float(p.score) if p.score is not None else 1.0,
            }
        )
    return entries


def extract_detections(rec: Record, classes: List[str]) -> Tuple[Array, Array, Array, List[Optional[Array]], List[SourceRef]]:
    img_h, img_w = rec.image.height, rec.image.width
    poly_entries = _poly_entries(rec, classes=classes, img_w=img_w, img_h=img_h)
    poly_by_gid: Dict[int, List[dict]] = {}
    for pe in poly_entries:
        if pe["group_id"] is None:
            continue
        poly_by_gid.setdefault(int(pe["group_id"]), []).append(pe)

    boxes: List[List[float]] = []
    scores: List[float] = []
    labels: List[str] = []
    masks: List[Optional[np.ndarray]] = []
    sources: List[SourceRef] = []
    used_poly: set[int] = set()

    for i, b in enumerate(rec.boxes):
        if b.kind == "pose":
            continue
        x1, y1, x2, y2 = bbox_xyxy_from_norm(b.cx, b.cy, b.w, b.h, img_w=img_w, img_h=img_h)
        lb = _label_name(b.cls, b.label, classes)
        sc = float(b.score) if b.score is not None else 1.0

        mask = None
        poly_idx: Optional[int] = None
        if b.group_id is not None and int(b.group_id) in poly_by_gid:
            mask = poly_by_gid[int(b.group_id)][0]["mask"]
            poly_idx = poly_by_gid[int(b.group_id)][0]["idx"]
            used_poly.add(poly_idx)
        else:
            best_iou = 0.0
            best_poly = None
            for pe in poly_entries:
                if pe["idx"] in used_poly:
                    continue
                if pe["label"] != lb:
                    continue
                iou = box_iou(np.array([x1, y1, x2, y2], dtype=float), pe["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_poly = pe
            if best_poly is not None and best_iou >= 0.5:
                mask = best_poly["mask"]
                poly_idx = best_poly["idx"]
                used_poly.add(best_poly["idx"])

        boxes.append([x1, y1, x2, y2])
        scores.append(sc)
        labels.append(lb)
        masks.append(mask)
        sources.append(("box", i, poly_idx))

    for pe in poly_entries:
        if pe["idx"] in used_poly:
            continue
        boxes.append(pe["bbox"].tolist())
        scores.append(pe["score"])
        labels.append(pe["label"])
        masks.append(pe["mask"])
        sources.append(("poly", None, pe["idx"]))

    if not boxes:
        return (
            np.zeros((0, 4), dtype=float),
            np.zeros((0,), dtype=float),
            np.array([], dtype=object),
            [],
            [],
        )

    return (
        np.asarray(boxes, dtype=float),
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=object),
        masks,
        sources,
    )


def apply_sources_to_record(rec: Record, sources: List[SourceRef]) -> Record:
    kept_boxes_idx: set[int] = set()
    kept_polys_idx: set[int] = set()
    for src_kind, box_idx, poly_idx in sources:
        if src_kind == "box" and box_idx is not None:
            kept_boxes_idx.add(int(box_idx))
        if poly_idx is not None:
            kept_polys_idx.add(int(poly_idx))

    new_boxes = [b for i, b in enumerate(rec.boxes) if i in kept_boxes_idx]
    new_polys = [p for i, p in enumerate(rec.polys) if i in kept_polys_idx] if rec.polys else []
    return replace(rec, boxes=new_boxes, polys=new_polys)


def gt_boxes_labels_from_record(rec: Record, classes: List[str]) -> Tuple[Array, List[str]]:
    img_h, img_w = rec.image.height, rec.image.width
    boxes: List[List[float]] = []
    labels: List[str] = []

    added_gid: set[int] = set()
    for p in rec.polys:
        pts = [(x * img_w, y * img_h) for x, y in p.points]
        if len(pts) < 3:
            continue
        x1, y1, x2, y2 = polygon_bbox(pts)
        boxes.append([x1, y1, x2, y2])
        labels.append(_label_name(p.cls, p.label, classes))
        if p.group_id is not None:
            added_gid.add(int(p.group_id))

    for b in rec.boxes:
        if b.kind == "pose":
            continue
        if b.group_id is not None and int(b.group_id) in added_gid:
            continue
        x1, y1, x2, y2 = bbox_xyxy_from_norm(b.cx, b.cy, b.w, b.h, img_w=img_w, img_h=img_h)
        boxes.append([x1, y1, x2, y2])
        labels.append(_label_name(b.cls, b.label, classes))

    if not boxes:
        return np.zeros((0, 4), dtype=float), []
    return np.asarray(boxes, dtype=float), labels


def warn_on_stem_mismatch(tp_ds: VisionDataset, permissive_ds: VisionDataset):
    tp_map = _stem_map(tp_ds)
    perm_map = _stem_map(permissive_ds)
    tp_stems = set(tp_map.keys())
    perm_stems = set(perm_map.keys())
    common = sorted(tp_stems & perm_stems)
    only_tp = sorted(tp_stems - perm_stems)
    only_perm = sorted(perm_stems - tp_stems)

    if tp_ds.root and permissive_ds.root and tp_ds.root.resolve() != permissive_ds.root.resolve():
        print(f"[warn] GT root ({tp_ds.root}) differs from permissive root ({permissive_ds.root}); matching by stem only.")
    if only_tp or only_perm:
        def _sample(lst: List[str]) -> str:
            if not lst:
                return ""
            sample = ", ".join(lst[:5])
            more = "" if len(lst) <= 5 else f" ... (+{len(lst)-5})"
            return sample + more
        print(f"[warn] unmatched images -> gt_only={len(only_tp)} perm_only={len(only_perm)}")
        if only_tp:
            print(f"       gt-only stems: {_sample(only_tp)}")
        if only_perm:
            print(f"       perm-only stems: {_sample(only_perm)}")
    return common, tp_map, perm_map


def _stem_map(ds: VisionDataset) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    for rec in ds.records:
        stem = rec.image.path.stem
        if stem not in out:
            out[stem] = rec
    return out


def load_rgb_for_rec(rec: Record) -> Optional[np.ndarray]:
    try:
        return load_rgb_safe(rec.image.path)
    except Exception:
        return None


def build_batch(rec: Record, classes: List[str], rgb: Optional[np.ndarray] = None) -> Dict[str, object]:
    boxes, scores, labels, masks, sources = extract_detections(rec, classes=classes)
    return {
        "H": rec.image.height,
        "W": rec.image.width,
        "rgb": rgb,
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "masks": masks,
        "sources": sources,
    }
