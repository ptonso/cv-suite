"""Generate one caption per image using a shared VLM backend."""

from __future__ import annotations

import argparse

from ..core import runtime


def attach(parser: argparse.ArgumentParser) -> None:
    runtime.attach_runtime_args(parser)


def run(dataset, args: argparse.Namespace):
    runtime.seed_caption(dataset)
    return runtime.invoke_model(dataset, args, prompt=runtime.io.CAPTION_PROMPT)
