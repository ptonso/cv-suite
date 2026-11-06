"""Remove classes or records by name."""

from __future__ import annotations

import argparse

from cvsuite.common.transforms import select as common_select
from cvsuite.common.transforms.select import run


def attach(parser: argparse.ArgumentParser) -> None:
    common_select.attach(parser, pixel_flags=False)


__all__ = ["attach", "run"]
