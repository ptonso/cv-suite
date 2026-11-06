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

from cvsuite.common.io.attach import attach_link_mode, attach_overwrite_flag
from cvsuite.common.io.folds import fold_targets

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
    attach_link_mode(parser)
    attach_overwrite_flag(parser)
    parser.add_argument("--unlabeled-name", default="unlabeled", help="Folder name used when no class label is available.")
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
    exported = dataset
    for fold_dataset, fold_dst in fold_targets(dataset, args.dst):
        exported = io.write_class_dir(
            fold_dataset,
            fold_dst,
            threshold=args.threshold,
            class_thresholds=class_thresholds,
            link_mode=getattr(args, "link_mode", "copy"),
            unlabeled_name=str(getattr(args, "unlabeled_name", "unlabeled")),
            multi_class=output_multi_class,
            exclude_classes=getattr(args, "exclude_class", None),
            overwrite=bool(getattr(args, "overwrite", False)),
        )
        if bool(getattr(args, "with_stats", False)):
            ctx = make_export_context(
                branch="classify",
                command="to-class-dir",
                dst=fold_dst,
                run_meta={
                    "threshold": args.threshold,
                    "class_thresholds": class_thresholds,
                    "unlabeled_name": str(getattr(args, "unlabeled_name", "unlabeled")),
                    "multi_class": output_multi_class,
                },
            )
            emit_stats_yaml(build_classify_stats(exported, ctx), ctx)
    return exported
