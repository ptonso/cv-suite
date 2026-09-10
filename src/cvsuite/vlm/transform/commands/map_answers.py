"""Explode QA entries and map free-form answers into canonical labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core import map_answers


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("labels", nargs="*", help="Canonical mapped labels. Synonyms come from --labels-file.")
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Optional YAML/JSON file containing either a label list or a label -> synonym-list mapping.",
    )


def run(dataset, args: argparse.Namespace):
    registry = map_answers.load_label_registry(args.labels, labels_file=args.labels_file)
    return map_answers.apply_answer_mapping(dataset, registry)
