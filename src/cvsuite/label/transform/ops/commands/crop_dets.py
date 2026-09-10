"""Crop each detected object into its own record and image."""

from __future__ import annotations

import argparse

from cvsuite.common.core import VisionDataset

from cvsuite.label.transform.ops.core.crop_dets import crop_dets_dataset


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Optional square size for each crop. Resizes the long side to this size, then pads the short side.",
    )
    parser.add_argument(
        "--long-size",
        type=int,
        default=None,
        help="Optional long-side size for each crop; preserves aspect ratio without padding.",
    )
    parser.add_argument(
        "--pad-type",
        choices=["background", "black", "gray", "white"],
        default="black",
        help="Padding fill used with --imgsz. 'background' fills from the original image; the others use a constant color.",
    )
    parser.add_argument(
        "--margin-frac",
        type=float,
        default=0.0,
        help="Extra margin, as a fraction of detection width/height, added on each crop edge.",
    )


def run(ds: VisionDataset, args: argparse.Namespace) -> VisionDataset:
    return crop_dets_dataset(
        ds,
        imgsz=getattr(args, "imgsz", None),
        long_size=getattr(args, "long_size", None),
        pad_type=getattr(args, "pad_type", "black"),
        margin_frac=float(getattr(args, "margin_frac", 0.0) or 0.0),
    )
