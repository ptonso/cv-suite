"""Generate background negatives and append them to the in-memory dataset.

Crops are written to a temporary `negatives/` folder under a fresh temp root.
New Records are added pointing to those crops (no annotations).
"""

from pathlib import Path
import argparse
import tempfile
from typing import Optional, Set

from ..core.negatives import plan_crops, write_crops
from cvsuite.common.core import ImageRecord, Record
from cvsuite.common.core import VisionDataset


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--splits",
        default="",
        help="Optional comma-separated splits to sample from (default: all in dataset).",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Square crop size in pixels.",
    )
    p.add_argument(
        "--per-image",
        type=int,
        default=3,
        help="Number of crops to draw from each source image.",
    )
    p.add_argument(
        "--min-gap",
        type=float,
        default=0.04,
        help="Reserved: target minimum fractional gap from labeled boxes (not enforced).",
    )
    p.add_argument(
        "--iou-thresh",
        type=float,
        default=0.0,
        help="Reserved: maximum IoU allowed with labeled boxes (not enforced).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic crop sampling.",
    )
    p.add_argument(
        "--preview",
        type=Path,
        help="Optional path for a collage preview of sampled crops.",
    )


def _rewrite_crop_paths(crops, new_root: Path, orig_root: Optional[Path]) -> None:
    for c in crops:
        rel = c.out_path
        if orig_root:
            try:
                rel = c.out_path.relative_to(orig_root)
            except Exception:
                pass
        c.out_path = new_root / rel


def run(ds: VisionDataset, a: argparse.Namespace) -> VisionDataset:
    tmp_root = Path(tempfile.mkdtemp(prefix="vislabel_neg_"))

    split_filter: Optional[Set[str]] = {s.strip() for s in a.splits.split(",") if s.strip()} or None
    if split_filter:
        recs = [r for r in ds.records if r.split in split_filter]
        ds = VisionDataset(records=recs, classes=ds.classes, task=ds.task, root=ds.root, meta=ds.meta)

    crops = plan_crops(
        ds,
        imgsz=a.imgsz,
        per_image=a.per_image,
        min_gap=a.min_gap,
        iou_thresh=a.iou_thresh,
        seed=a.seed,
    )
    _rewrite_crop_paths(crops, new_root=tmp_root, orig_root=ds.root)
    write_crops(crops, tmp_root, preview_sheet=a.preview)

    new_records = list(ds.records)
    for c in crops:
        x1, y1, x2, y2 = c.box_xyxy
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        split = c.out_path.parent.name  # negatives/<split>/
        if c.out_path.parent.parent.name == "negatives":
            split = c.out_path.parent.name
        new_records.append(
            Record(
                image=ImageRecord(path=c.out_path, width=w, height=h),
                split=split,
                task=ds.task,
                attributes={"negative": True, "source_image": str(c.src_image)},
            )
        )

    meta = dict(ds.meta)
    meta["negatives_root"] = tmp_root
    return VisionDataset(records=new_records, classes=ds.classes, task=ds.task, root=ds.root, meta=meta)
