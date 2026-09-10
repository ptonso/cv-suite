from __future__ import annotations

import argparse
from pathlib import Path

from ..core.config import ProcessConfig, ResizeMode
from ..core.process.core import run_process_core


def run_process(args: argparse.Namespace) -> None:
    if args.size is None and args.resize_mode is not None:
        raise SystemExit("--resize-mode requires --size.")
    if args.size is not None and args.resize_mode is None:
        resize_mode: ResizeMode = "long"
    elif args.resize_mode is None:
        resize_mode = "none"
    else:
        resize_mode = args.resize_mode  # type: ignore[assignment]

    if resize_mode != "square" and args.square_mode is not None:
        raise SystemExit("--square-mode is only valid with --resize-mode=square.")

    video_sample_mode = args.video_sample_mode
    video_sample_k = args.video_sample_k
    video_sample_min_gap = args.video_sample_min_gap
    if video_sample_mode == "skip":
        video_sample_k = None
        video_sample_min_gap = None

    verbose = 1 if args.verbose else 0
    if args.dry_run:
        verbose = 1

    config = ProcessConfig(
        src=args.src,
        dst=args.dst,
        orient=bool(args.orient),
        resize_mode=resize_mode,
        size=args.size,
        square_mode=args.square_mode,
        grayscale=bool(args.grayscale),
        to_format=args.to_format,
        backend=args.backend,
        jpeg_quality=args.jpeg_quality,
        transfer=args.transfer,
        non_image=args.non_image,
        video_sample_mode=video_sample_mode,
        video_sample_k=video_sample_k,
        video_sample_min_gap=video_sample_min_gap,
        manifest=not bool(args.no_manifest),
        dry_run=bool(args.dry_run),
        verbose=verbose,
    )
    run_process_core(config)


def register_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "process",
        help="Content normalization for image files.",
        description="Standardize image pixels and formats while preserving directory structure, with optional video frame sampling.",
    )
    parser.set_defaults(func=run_process)

    parser.add_argument("src", type=Path, help="Source root directory.")
    parser.add_argument("dst", type=Path, help="Destination root directory.")

    parser.add_argument(
        "--orient",
        action="store_true",
        help="Apply EXIF-based orientation correction when available.",
    )

    parser.add_argument(
        "--size",
        type=int,
        help="Target size parameter for resizing. Interpreted according to resize mode. Default long.",
    )

    parser.add_argument(
        "--resize-mode",
        choices=("long", "short", "square", "cap-long", "cap-short"),
        help="Resize mode: long/short force that side to size; square makes size x size. cap-long/cap-short only downscale if exceeding size.",
    )

    parser.add_argument(
        "--square-mode",
        choices=("pad", "crop", "distort"),
        help="How to make square images when using --resize-mode=square.",
    )

    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert images to grayscale.",
    )

    parser.add_argument(
        "--to-format",
        help="Convert images to the given format (e.g. jpg, png, webp).",
    )

    parser.add_argument(
        "--backend",
        choices=("pillow", "opencv"),
        default="opencv",
        help="Image decode/resize/encode backend. Default opencv.",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        choices=range(0, 101),
        default=95,
        metavar="0..100",
        help="JPEG encoder quality. Default 95.",
    )

    parser.add_argument(
        "--video-sample-mode",
        choices=("skip", "collect", "keep"),
        default="skip",
        help="How to treat sampled video frames as images: skip (no sampling), collect into a fixed folder, or keep alongside the video.",
    )

    parser.add_argument(
        "--video-sample-k",
        type=int,
        help="Maximum number of frames to sample per video. If not set, combined with min-gap rules only.",
    )

    parser.add_argument(
        "--video-sample-min-gap",
        type=float,
        help="Minimum temporal gap in seconds between sampled frames. "
             "If set without k, sample frames at this gap for the whole video. "
             "If both k and min-gap are set, sample up to k frames respecting the gap.",
    )

    parser.add_argument(
        "--transfer",
        choices=("copy", "move"),
        default="copy",
        help="How to write output images with respect to the source tree.",
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
