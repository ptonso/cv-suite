"""Sample a class-organized dataset while preserving or balancing class distribution."""

from __future__ import annotations

import argparse

from ..core import sample as sample_core


def attach(parser: argparse.ArgumentParser) -> None:
    sample_core.attach(parser)


def run(dataset, args: argparse.Namespace):
    return sample_core.run(dataset, args)
