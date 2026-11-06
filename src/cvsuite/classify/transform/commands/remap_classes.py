"""Merge, rename and drop classes with a YAML map."""

from __future__ import annotations

import argparse

from cvsuite.common.transforms import remap as common_remap
from cvsuite.common.transforms.remap import run


def attach(parser: argparse.ArgumentParser) -> None:
    common_remap.attach(parser, empty_flag=False)


__all__ = ["attach", "run"]
