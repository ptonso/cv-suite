"""Apply axis-aligned thresholds to a dataset of detections/segmentations."""

import argparse
from pathlib import Path
from typing import List

from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset

from ..core import (
    ThresholdFilter,
    load_filter_config,
    example_threshold_yaml,
    load_rgb_for_rec,
)


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="YAML with threshold config (global/per_class).")
    p.add_argument("--with-rgb", action="store_true", help="Load images to enable color-based thresholds.")


def _filter_records(ds: VisionDataset, flt: ThresholdFilter, load_rgb: bool) -> List[Record]:
    out: List[Record] = []
    for rec in ds.records:
        rgb = load_rgb_for_rec(rec) if load_rgb else None
        out.append(flt.filter_record(rec, classes=ds.classes, rgb=rgb))
    return out


def run(ds: VisionDataset, args: argparse.Namespace) -> VisionDataset:
    if not args.config:
        print("# threshold config example")
        print(example_threshold_yaml())
        return ds

    cfg = load_filter_config(args.config)
    flt = ThresholdFilter(cfg)
    filtered = _filter_records(ds, flt, load_rgb=bool(args.with_rgb))
    return VisionDataset(records=filtered, classes=ds.classes, task=ds.task, root=ds.root, meta=ds.meta)
