"""Write tar shards with one image + one JSON member per sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvsuite.common.io import write_dataset

from cvsuite.common.stats import attach_with_stats_flag, build_vlm_stats, emit_stats_yaml, make_export_context

from ..common import attach_split_filter_arg, filter_dataset_by_splits, iter_answer_partitions


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dst", type=Path, help="Output directory containing a shards/ folder.")
    parser.add_argument("--target-shard-size-mb", type=int, default=500, help="Approximate shard size target in MB.")
    parser.add_argument("--max-samples-per-shard", type=int, default=10_000, help="Maximum samples stored per shard.")
    attach_split_filter_arg(parser)
    attach_with_stats_flag(parser)


def run(dataset, args: argparse.Namespace):
    dataset = filter_dataset_by_splits(dataset, args.splits)
    partitions = iter_answer_partitions(dataset)
    if not partitions:
        write_dataset(
            dataset,
            args.dst,
            format="shards_vlm",
            target_shard_size_mb=args.target_shard_size_mb,
            max_samples_per_shard=args.max_samples_per_shard,
        )
        if bool(getattr(args, "with_stats", False)):
            shard_count = len(list((args.dst / "shards").glob("*.tar")))
            ctx = make_export_context(
                branch="vlm",
                command="to-shards",
                dst=args.dst,
                writer_details={"shards_written": shard_count},
            )
            emit_stats_yaml(build_vlm_stats(dataset, ctx), ctx)
        return dataset

    total_shards_written = 0
    for label, partition in partitions:
        write_dataset(
            partition,
            args.dst / label,
            format="shards_vlm",
            target_shard_size_mb=args.target_shard_size_mb,
            max_samples_per_shard=args.max_samples_per_shard,
        )
        total_shards_written += len(list(((args.dst / label) / "shards").glob("*.tar")))
    if bool(getattr(args, "with_stats", False)):
        ctx = make_export_context(
            branch="vlm",
            command="to-shards",
            dst=args.dst,
            writer_details={
                "partitions_written": len(partitions),
                "shards_written": total_shards_written,
            },
        )
        emit_stats_yaml(build_vlm_stats(dataset, ctx), ctx)
    return dataset
