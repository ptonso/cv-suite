from __future__ import annotations

import argparse
from pathlib import Path

from ..core.config import ArrangeConfig
from ..core.arrange.core import run_arrange_core


def run_arrange(args: argparse.Namespace) -> None:
    if args.flatten is None:
        flatten_mode = "none"
    elif args.flatten == "enc-prefix":
        flatten_mode = "enc-prefix"
    elif args.flatten == "enc-suffix":
        flatten_mode = "enc-suffix"
    else:
        flatten_mode = "plain"

    verbose = 1 if args.verbose else 0
    if args.dry_run:
        verbose = 1

    dst_subdir = args.dst_subdir
    if dst_subdir is not None:
        target = Path(dst_subdir)
        if target.is_absolute() or len(target.parts) != 1 or dst_subdir in {".", ".."}:
            raise SystemExit("--dst-subdir must be a single folder name.")

    if args.perceptual_dedup is not None:
        dedup_mode = "perceptual"
        perceptual_threshold = args.perceptual_dedup
    elif args.exact_dedup:
        dedup_mode = "exact"
        perceptual_threshold = None
    else:
        dedup_mode = "none"
        perceptual_threshold = None

    config = ArrangeConfig(
        src=args.src,
        dst=args.dst,
        unzip=bool(args.unzip),
        flatten=flatten_mode,
        dedup_mode=dedup_mode,
        perceptual_threshold=perceptual_threshold,
        rename_seq=bool(args.rename_seq),
        dst_subdir=dst_subdir,
        link_mode=args.link_mode,
        non_image=args.non_image,
        manifest=not bool(args.no_manifest),
        dry_run=bool(args.dry_run),
        verbose=verbose,
    )
    run_arrange_core(config)


def register_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "arrange",
        help="Structural arrange cleaning of a dataset tree.",
        description="Standardize directory layout, deduplicate files, control non-image handling.",
    )
    parser.set_defaults(func=run_arrange)

    parser.add_argument("src", type=Path, help="Source root directory.")
    parser.add_argument("dst", type=Path, help="Destination root directory.")

    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Expand supported archive files before other structural operations.",
    )

    parser.add_argument(
        "--flatten",
        nargs="?",
        const="plain",
        choices=("plain", "enc-prefix", "enc-suffix"),
        help="Flatten directory structure. If no value is given, lose structure. "
             "Use enc-prefix or enc-suffix to encode original folders in filenames.",
    )

    dedup_group = parser.add_mutually_exclusive_group()
    dedup_group.add_argument(
        "--exact-dedup",
        action="store_true",
        help="Remove byte-identical duplicate files by (size, sha1).",
    )
    dedup_group.add_argument(
        "--perceptual-dedup",
        nargs="?",
        const=5,
        type=int,
        metavar="THRESHOLD",
        help="Remove near-duplicate images by perceptual hash (pHash). "
             "Images within THRESHOLD Hamming distance are duplicates "
             "(default 5; lower is stricter). Non-images pass through.",
    )

    parser.add_argument(
        "--rename-seq",
        action="store_true",
        help="Rename image files to a consistent sequential scheme after other operations.",
    )

    parser.add_argument(
        "--dst-subdir",
        metavar="NAME",
        help="Write kept files under dst/NAME/ for incremental review.",
    )

    parser.add_argument(
        "--link-mode",
        choices=("copy", "move", "hard"),
        default="copy",
        help="How to materialize files into the destination tree.",
    )

    parser.add_argument(
        "--non-image",
        choices=("skip", "collect", "keep"),
        default="collect",
        help="How to handle non-image files: skip, collect into a separate folder, or keep with images.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without touching the filesystem. Implies verbose.",
    )

    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write dst/manifest.yaml after a successful run.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print actions and statistics.",
    )
