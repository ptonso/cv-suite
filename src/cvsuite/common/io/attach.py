"""CLI argument shared by the output commands that write through `write_dataset`."""

import argparse

from cvsuite.common.io.common import LINK_MODES


def attach_link_mode(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--link-mode",
        choices=LINK_MODES,
        default="copy",
        help="How images reach the destination: copy bytes (default), hard link, or absolute symbolic link. A failed link raises.",
    )


def attach_overwrite_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output trees this command writes when they already hold files.",
    )
