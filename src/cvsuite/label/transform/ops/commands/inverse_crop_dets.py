"""Paste modified crop-dets crops back into their original images."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvsuite.common.core import VisionDataset

from cvsuite.label.core import router
from cvsuite.label.transform.ops.core.inverse_crop_dets import inverse_crop_dets_dataset


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("modified_crops", type=Path, help="LabelMe crop dataset produced from crop-dets output.")


def run(ds: VisionDataset, args: argparse.Namespace) -> VisionDataset:
    crops = router.ingest(Path(args.modified_crops), from_hint="labelme")
    return inverse_crop_dets_dataset(ds, crops)
