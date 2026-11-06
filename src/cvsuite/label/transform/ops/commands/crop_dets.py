"""Crop each detected object into its own record and image."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

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
        choices=["background", "black", "gray", "white", "shift"],
        default="black",
        help=(
            "Padding fill used with --imgsz. 'background' fills from the original image; the others use a constant color. "
            "'shift' cuts a square of real pixels around the box, slid inside the image, and never pads."
        ),
    )
    parser.add_argument(
        "--margin-frac",
        type=float,
        default=0.0,
        help="Extra margin, as a fraction of detection width/height, added on each crop edge.",
    )
    parser.add_argument(
        "--margin-px",
        type=float,
        default=0.0,
        help="Extra margin in source pixels added on each crop edge, on top of --margin-frac.",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Write the provenance of every crop to this JSON file.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing --manifest file.")


def run(ds: VisionDataset, args: argparse.Namespace) -> VisionDataset:
    manifest = getattr(args, "manifest", None)
    if manifest is not None and manifest.exists() and not getattr(args, "overwrite", False):
        raise FileExistsError(f"Manifest already exists: {manifest}. Pass --overwrite to replace it.")
    out = crop_dets_dataset(
        ds,
        imgsz=getattr(args, "imgsz", None),
        long_size=getattr(args, "long_size", None),
        pad_type=getattr(args, "pad_type", "black"),
        margin_frac=float(getattr(args, "margin_frac", 0.0) or 0.0),
        margin_px=float(getattr(args, "margin_px", 0.0) or 0.0),
    )
    if manifest is not None:
        entries = [
            {**record.attributes["crop"], "file": Path(record.image.path).name, "split": record.split, "label": record.classification.label}
            for record in out.records
        ]
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return replace(out, meta={**out.meta, "step_info": {"crops": len(out.records), "manifest": str(manifest) if manifest else None}})
