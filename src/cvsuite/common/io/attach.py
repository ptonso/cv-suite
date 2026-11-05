"""CLI argument shared by the output commands that write through `write_dataset`."""

import argparse


def attach_overwrite_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output trees this command writes when they already hold files.",
    )
