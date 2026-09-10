"""Preview images with all question-answer pairs rendered in a side panel."""

from __future__ import annotations

import argparse

from ...core import inspect
from ..common import ensure_unpartitioned, filter_dataset_by_splits


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", "--splits", dest="split", default="", help="Optional comma-separated splits to visualize.")
    parser.add_argument("--max", type=int, default=0, help="Maximum items to preview (0 means all).")


def run(dataset, args: argparse.Namespace):
    dataset = filter_dataset_by_splits(dataset, args.split)
    ensure_unpartitioned(dataset, command_name="inspect")
    splits = [item.strip() for item in args.split.split(",") if item.strip()] if args.split else None
    inspect.run_preview(dataset, splits=splits, max_items=args.max)
    return dataset
