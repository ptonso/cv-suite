from __future__ import annotations

import argparse


def attach_with_stats_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--with-stats",
        action="store_true",
        help="Write a stats.yaml summary alongside the exported output.",
    )
