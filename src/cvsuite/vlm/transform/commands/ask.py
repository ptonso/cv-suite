"""Seed one or more questions across all images, then run the VLM."""

from __future__ import annotations

import argparse

from ..core import runtime


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", required=True, help="Question string, JSON string/list/object, or a YAML/JSON file path.")
    runtime.attach_runtime_args(parser)


def run(dataset, args: argparse.Namespace):
    runtime.seed_ask(dataset, args.prompt)
    return runtime.invoke_model(dataset, args, prompt=args.prompt)
