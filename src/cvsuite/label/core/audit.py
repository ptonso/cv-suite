from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Tuple

from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset


def _resolution_summary(records: List[Record]) -> Dict[str, Any]:
    dims: List[Tuple[int, int]] = [(r.image.width, r.image.height) for r in records if r.image]
    if not dims:
        return {}
    widths = [w for w, _ in dims]
    heights = [h for _, h in dims]

    def _mean(vals: List[int]) -> float:
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    res: Dict[str, Any] = {
        "min": [min(widths), min(heights)],
        "max": [max(widths), max(heights)],
        "mean": [_mean(widths), _mean(heights)],
    }
    return res


def _summarize(records: List[Record]) -> Dict[str, Any]:
    total = len(records)
    det_anns = seg_anns = pose_boxes = pose_kpt_sets = 0
    imgs_with_det = imgs_with_seg = imgs_with_pose = 0
    empty = 0

    for rec in records:
        dets = [b for b in rec.boxes if b.kind != "pose"]
        pose = [b for b in rec.boxes if b.kind == "pose"]
        segs = rec.polys
        kpts = rec.kpts

        det_anns += len(dets)
        seg_anns += len(segs)
        pose_boxes += len(pose)
        pose_kpt_sets += len(kpts)

        has_det = bool(dets)
        has_seg = bool(segs)
        has_pose = bool(pose or kpts)

        if has_det:
            imgs_with_det += 1
        if has_seg:
            imgs_with_seg += 1
        if has_pose:
            imgs_with_pose += 1
        if not (rec.boxes or rec.polys or rec.kpts):
            empty += 1

    annotated = total - empty
    summary: Dict[str, Any] = {
        "images": total,
        "images_with_annotations": annotated,
        "images_without_annotations": empty,
        "images_with": {
            "detections": imgs_with_det,
            "segmentations": imgs_with_seg,
            "pose": imgs_with_pose,
        },
        "annotations": {
            "detections": det_anns,
            "segmentations": seg_anns,
            "pose_boxes": pose_boxes,
            "pose_keypoint_sets": pose_kpt_sets,
        },
        "resolutions": _resolution_summary(records),
    }
    return summary


def run_checks(ds: VisionDataset, thresholds: Dict[str, Any]) -> Dict[str, Any]:
    splits = ds.by_split()
    per_split = {split: _summarize(items) for split, items in splits.items()}

    overall = _summarize(ds.records)
    overall["images_per_split"] = {split: len(items) for split, items in splits.items()}
    overall["class_count"] = len(ds.classes) if ds.classes else 0

    return {"overall": overall, "per_split": per_split}
