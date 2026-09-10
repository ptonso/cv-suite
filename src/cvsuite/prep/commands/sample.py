from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ..core.config import SampleConfig
from ..core.sample.core import run_sample_core


def run_sample(args: argparse.Namespace) -> None:
    count: Optional[int] = args.count
    frac: Optional[float] = args.frac
    if count is not None and frac is not None:
        raise SystemExit("Use only one of --count or --frac.")
    if count is None and frac is None:
        raise SystemExit("One of --count or --frac is required.")

    if count is not None and count < 0:
        raise SystemExit("--count must be non-negative.")

    verbose = 1 if args.verbose else 0
    if args.dry_run:
        verbose = 1

    config = SampleConfig(
        srcs=tuple(args.srcs),
        dst=args.dst,
        count=count,
        frac=frac,
        hardlink=bool(args.hardlink),
        seed=args.seed,
        dry_run=bool(args.dry_run),
        verbose=verbose,
    )
    run_sample_core(config)


def register_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "sample",
        help="Sample a pooled subset from one or more flat image folders.",
        description=(
            "Pool images from one or more flat raw-image folders, sample a single subset, "
            "and write a flat destination folder. Use `cvsuite class sample` for class-organized "
            "datasets and `cvsuite label sample` for annotated datasets."
        ),
    )
    parser.set_defaults(func=run_sample)

    parser.add_argument("srcs", nargs="+", type=Path, help="One or more flat source folders containing images.")
    parser.add_argument("dst", type=Path, help="Destination folder for the sampled flat image set.")

    parser.add_argument(
        "--count",
        type=int,
        help="Number of pooled images to sample. Mutually exclusive with --frac.",
    )
    parser.add_argument(
        "--frac",
        type=float,
        help="Fraction of pooled images to sample. Mutually exclusive with --count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Hard-link sampled images instead of copying bytes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which images would be sampled without creating any files. Implies verbose.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print sampling details and statistics.",
    )
