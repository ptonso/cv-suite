"""Recompute pose bounding boxes from keypoints for an in-memory dataset."""

from typing import List, Optional, Set
import argparse

from ..core.rebox import rebox_item
from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mode",
        choices=["tight", "expand", "adaptive"],
        default="adaptive",
        help="tight uses the keypoint bounding box; expand/adaptive pad by margin.",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="Fractional padding added to width/height when mode is expand/adaptive.",
    )
    p.add_argument(
        "--edge-tol",
        type=float,
        default=0.004,
        help="Reserved tolerance for edge handling (currently unused).",
    )
    p.add_argument(
        "--splits",
        default="",
        help="Optional comma-separated splits to process (default: all).",
    )


def _should_process(rec: Record, split_filter: Optional[Set[str]]) -> bool:
    if not split_filter:
        return True
    return (rec.split or "train") in split_filter


def run(ds: VisionDataset, a: argparse.Namespace) -> VisionDataset:
    split_filter: Optional[Set[str]] = {s.strip() for s in a.splits.split(",") if s.strip()} or None
    fixed: List[Record] = []
    for rec in ds.records:
        if _should_process(rec, split_filter):
            fixed.append(rebox_item(rec, mode=a.mode, margin=a.margin, edge_tol=a.edge_tol))
        else:
            fixed.append(rec)
    return VisionDataset(records=fixed, classes=ds.classes, task=ds.task, root=ds.root, meta=ds.meta)
