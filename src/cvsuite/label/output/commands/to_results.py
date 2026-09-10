"""Export rich visual outputs (annotated overlays, crops, masks, keypoints) for a VisionDataset."""

from pathlib import Path
import argparse
from collections import Counter
import cv2

from cvsuite.common.core import VisionDataset
from cvsuite.common.stats import attach_with_stats_flag, build_label_stats, emit_stats_yaml, make_export_context
from cvsuite.label.core.annotate import render_overlay, detection_crops, class_masks, keypoint_overlay, select_indices
from cvsuite.label.output.common import assign_output_splits, attach_split_assignment_args


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument("dst", type=Path, help="Destination root for results.")
    p.add_argument("--names-mode", choices=["smart", "none", "det", "seg", "all"], default="smart",
                   help="Controls labels on annotated overlays.")
    p.add_argument("--imgsz", type=int, default=None, help="Optional square resize before exporting.")
    p.add_argument("--max", type=int, default=0, help="Maximum number of items to export (evenly spaced).")
    attach_split_assignment_args(p)
    p.add_argument("--skip-annotated", action="store_true", help="Do not write annotated overlays.")
    p.add_argument("--skip-crops", action="store_true", help="Do not write detection crops.")
    p.add_argument("--skip-masks", action="store_true", help="Do not write segmentation masks.")
    p.add_argument("--skip-keypoints", action="store_true", help="Do not write keypoint overlays.")
    attach_with_stats_flag(p)


def _write_image(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def run(ds, a: argparse.Namespace):
    if not bool(getattr(a, "preserve_splits", False)):
        ds = assign_output_splits(
            ds,
            val_frac=getattr(a, "val_frac", 0.0),
            test_frac=getattr(a, "test_frac", 0.0),
            seed=getattr(a, "seed", 0),
        )
    records = list(ds.records)
    idxs = select_indices(len(records), a.max) if records else []
    selected = [records[i] for i in idxs] if idxs else records
    crop_counts: Counter[str] = Counter()
    mask_counts: Counter[str] = Counter()
    annotated_images_written = 0
    keypoint_overlays_written = 0

    names = ds.classes or []
    root = ds.root
    any_kpts = any(rec.kpts for rec in selected)
    for rec in selected:
        stem = Path(rec.image.path).stem

        if not a.skip_annotated:
            vis = render_overlay(rec, names, names_mode=a.names_mode, imgsz=a.imgsz, root=root)
            if vis is not None:
                _write_image(a.dst / "annotated" / f"{stem}.jpg", vis)
                annotated_images_written += 1

        if not a.skip_crops:
            for idx, (cls_name, crop) in enumerate(detection_crops(rec, names, imgsz=a.imgsz, root=root)):
                _write_image(a.dst / "cropped" / cls_name / f"{stem}_{idx}.jpg", crop)
                crop_counts[cls_name] += 1

        if not a.skip_masks:
            for cls_name, mask in class_masks(rec, names, imgsz=a.imgsz, root=root):
                _write_image(a.dst / "masks" / cls_name / f"{stem}.png", mask)
                mask_counts[cls_name] += 1

        if not a.skip_keypoints and any_kpts:
            kvis = keypoint_overlay(rec, names, imgsz=a.imgsz, root=root)
            if kvis is not None:
                _write_image(a.dst / "keypoints" / f"{stem}.jpg", kvis)
                keypoint_overlays_written += 1

    export_ds = VisionDataset(
        records=list(selected),
        classes=list(ds.classes),
        task=ds.task,
        root=ds.root,
        meta=dict(ds.meta),
        fm_request=ds.fm_request,
    )
    if bool(getattr(a, "with_stats", False)):
        ctx = make_export_context(
            branch="label",
            command="to-results",
            dst=a.dst,
            writer_details={
                "selected_records": len(selected),
                "annotated_images_written": annotated_images_written,
                "crops_written_per_label": dict(sorted(crop_counts.items())),
                "masks_written_per_label": dict(sorted(mask_counts.items())),
                "keypoint_overlays_written": keypoint_overlays_written,
            },
        )
        emit_stats_yaml(build_label_stats(export_ds, ctx), ctx)
    return export_ds
