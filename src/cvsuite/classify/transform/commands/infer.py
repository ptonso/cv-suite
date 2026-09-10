"""Run FM-backed classification over the dataset."""

from __future__ import annotations

import argparse

from ..core import infer as infer_core


def attach(parser: argparse.ArgumentParser) -> None:
    infer_core.attach(parser)


def run(dataset, args: argparse.Namespace):
    return infer_core.run(dataset, args)
