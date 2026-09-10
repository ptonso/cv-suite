from __future__ import annotations

import argparse
from pathlib import Path

from ..core.config import OrientConfig
from ..core.orient.core import run_orient_core


def run_orient(args: argparse.Namespace) -> None:
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1.")

    verbose = 1 if args.verbose else 0
    if args.dry_run:
        verbose = 1

    config = OrientConfig(
        src=args.src,
        dst=args.dst,
        try_hardlink=bool(args.try_hardlink),
        batch=args.batch,
        device=args.device,
        precision=args.precision,
        weights=args.weights,
        no_resume=bool(args.no_resume),
        dry_run=bool(args.dry_run),
        verbose=verbose,
    )
    run_orient_core(config)


def register_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "orient",
        help="Normalize image orientation with EXIF first and deep-model fallback.",
        description=(
            "Walk a folder of images, preserve the directory structure under dst, and write "
            "upright images. EXIF orientation is used when present; otherwise a 4-class "
            "orientation model predicts the corrective rotation for each image."
        ),
    )
    parser.set_defaults(func=run_orient)

    parser.add_argument("src", type=Path, help="Source root directory.")
    parser.add_argument("dst", type=Path, help="Destination root directory.")

    parser.add_argument(
        "--try-hardlink",
        action="store_true",
        help="Hard-link unchanged source files when possible; falls back to copy automatically.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Fallback model batch size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu/gpu hint for fallback inference (default: cpu; the deep-orientation venv is CPU-only).",
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        help="Numerical precision hint for fallback inference.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory for fallback-model weights.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any resumable FM run state and force a fresh model pass.",
    )
    parser.add_argument("--prompt", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without touching the filesystem. Implies verbose.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-file actions and summary statistics.",
    )
