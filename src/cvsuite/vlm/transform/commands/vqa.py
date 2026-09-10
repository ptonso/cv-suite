"""Run the VLM over existing record-local questions."""

from __future__ import annotations

import argparse

from ..core import runtime


def attach(parser: argparse.ArgumentParser) -> None:
    runtime.attach_runtime_args(parser)


def run(dataset, args: argparse.Namespace):
    if not runtime.has_pending_vqas(dataset):
        raise SystemExit("vqa requires at least one pending question in the input dataset.")
    return runtime.invoke_model(dataset, args)
