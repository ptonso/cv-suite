"""Write images into class directories, optionally with train/val/test split folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvsuite.common.stats import (
    attach_with_stats_flag,
    build_classify_stats,
    emit_stats_yaml,
    make_export_context,
)

from ...core import io


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dst", type=Path, help="Destination directory for class-organized images.")
    parser.add_argument("--threshold", type=float, default=None, help="Route low-score predictions to the unlabeled bucket.")
    parser.add_argument(
        "--class-thresholds",
        type=Path,
        default=None,
        help="Optional YAML file containing per-class threshold overrides. Example: {cat: 0.2, dog: 0.5}.",
    )
    parser.add_argument(
        "--class-threshold",
        action="append",
        default=[],
        help="Repeatable per-class threshold override using class=value syntax. Example: --class-threshold dog=0.65",
    )
    parser.add_argument(
        "--multi-class",
        action="store_true",
        help=(
            "Write an image into every class folder whose probability meets the threshold; if none do, route it "
            "to the unlabeled bucket. This is enabled automatically for datasets ingested with `--from multi-class`."
        ),
    )
    parser.add_argument(
        "--exclude-class",
        action="append",
        default=[],
        dest="exclude_class",
        metavar="NAME",
        help="Repeatable. Drop every class with this name from the output; records belonging only to it are removed.",
    )
    parser.add_argument("--hardlink", action="store_true", help="Hard-link images instead of copying bytes.")
    parser.add_argument("--unlabeled-name", default="unlabeled", help="Folder name used when no class label is available.")
    parser.add_argument("--val-frac", type=float, default=0.0, help="Validation split fraction for output assignment.")
    parser.add_argument("--test-frac", type=float, default=0.0, help="Test split fraction for output assignment.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic split assignment.")
    parser.add_argument("--randomized", action="store_true", help="Use global random split assignment instead of stratified per-class assignment.")
    parser.add_argument("--preserve-splits", action="store_true", help="Keep existing record splits instead of assigning new output splits.")
    attach_with_stats_flag(parser)
    parser.epilog = (
        "Class-threshold YAML example:\n"
        "  partition_a: 0.10\n"
        "  partition_b: 0.50\n"
        "  partition_c: 0.25\n\n"
        "Routing behavior:\n"
        "  - Without --multi-class, thresholds only decide whether the image is unlabeled.\n"
        "    If any class meets its threshold, the winner is the class with the highest raw score.\n"
        "  - With --multi-class, the image is written to every class whose score meets its threshold.\n"
        "    If none do, it is written to the unlabeled bucket.\n"
        "  - Datasets ingested with `--from multi-class` automatically export their source labels as multi-class."
    )


def run(dataset, args: argparse.Namespace):
    class_thresholds = io.resolve_class_thresholds(
        yaml_path=getattr(args, "class_thresholds", None),
        cli_overrides=getattr(args, "class_threshold", None),
    )
    output_multi_class = io.resolve_output_multi_class(
        dataset,
        requested=bool(getattr(args, "multi_class", False)),
    )
    exported = io.write_class_dir(
        dataset,
        args.dst,
        threshold=args.threshold,
        class_thresholds=class_thresholds,
        hardlink=bool(getattr(args, "hardlink", False)),
        unlabeled_name=str(getattr(args, "unlabeled_name", "unlabeled")),
        val_frac=float(getattr(args, "val_frac", 0.0)),
        test_frac=float(getattr(args, "test_frac", 0.0)),
        seed=int(getattr(args, "seed", 0)),
        randomized=bool(getattr(args, "randomized", False)),
        multi_class=output_multi_class,
        preserve_splits=bool(getattr(args, "preserve_splits", False)),
        exclude_classes=getattr(args, "exclude_class", None),
    )
    if bool(getattr(args, "with_stats", False)):
        ctx = make_export_context(
            branch="classify",
            command="to-class-dir",
            dst=args.dst,
            run_meta={
                "threshold": args.threshold,
                "class_thresholds": class_thresholds,
                "unlabeled_name": str(getattr(args, "unlabeled_name", "unlabeled")),
                "multi_class": output_multi_class,
            },
        )
        emit_stats_yaml(build_classify_stats(exported, ctx), ctx)
    return exported
