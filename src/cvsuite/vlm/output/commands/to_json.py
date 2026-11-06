"""Write a flat directory of image + JSON sidecar pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvsuite.common.io import write_dataset
from cvsuite.common.io.attach import attach_link_mode

from cvsuite.common.stats import attach_with_stats_flag, build_vlm_stats, emit_stats_yaml, make_export_context

from ..common import attach_split_filter_arg, filter_dataset_by_splits, iter_answer_partitions


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dst", type=Path, help="Output directory for flat image+JSON pairs.")
    attach_link_mode(parser)
    parser.add_argument("--skip-images", action="store_true", help="Do not export image files; write only JSON sidecars.")
    attach_split_filter_arg(parser)
    attach_with_stats_flag(parser)


def run(dataset, args: argparse.Namespace):
    dataset = filter_dataset_by_splits(dataset, args.splits)
    partitions = iter_answer_partitions(dataset)
    if not partitions:
        write_dataset(
            dataset,
            args.dst,
            format="flat_vlm_json",
            link_mode=getattr(args, "link_mode", "copy"),
            skip_images=args.skip_images,
        )
        if bool(getattr(args, "with_stats", False)):
            ctx = make_export_context(branch="vlm", command="to-json", dst=args.dst)
            emit_stats_yaml(build_vlm_stats(dataset, ctx), ctx)
        return dataset

    for label, partition in partitions:
        write_dataset(
            partition,
            args.dst / label,
            format="flat_vlm_json",
            link_mode=getattr(args, "link_mode", "copy"),
            skip_images=args.skip_images,
        )
    if bool(getattr(args, "with_stats", False)):
        ctx = make_export_context(
            branch="vlm",
            command="to-json",
            dst=args.dst,
            writer_details={"partitions_written": len(partitions)},
        )
        emit_stats_yaml(build_vlm_stats(dataset, ctx), ctx)
    return dataset
