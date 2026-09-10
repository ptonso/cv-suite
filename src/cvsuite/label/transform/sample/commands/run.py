"""Sample a VisionDataset globally while preserving annotations."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import replace


def attach(parser: argparse.ArgumentParser) -> None:
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of records to sample globally.",
    )
    size_group.add_argument(
        "--frac",
        type=float,
        default=None,
        help="Fraction of records to sample globally.",
    )
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Request hard-linked image export when the output command supports it.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic sampling.")


def _sample_size(total: int, *, count: int | None, frac: float | None) -> int:
    if count is not None:
        if count < 0:
            raise SystemExit("--count must be non-negative.")
        if count > total:
            raise SystemExit(f"Requested --count {count} but only {total} records are available.")
        return int(count)

    assert frac is not None
    if frac < 0.0 or frac > 1.0:
        raise SystemExit("--frac must be between 0 and 1.")
    chosen = int(math.floor(total * frac))
    if frac > 0.0 and total > 0:
        chosen = max(1, chosen)
    return min(chosen, total)


def run(dataset, args: argparse.Namespace):
    total = len(dataset.records)
    target = _sample_size(
        total,
        count=getattr(args, "count", None),
        frac=getattr(args, "frac", None),
    )
    rng = random.Random(int(getattr(args, "seed", 0)))
    indices = rng.sample(range(total), target) if target else []
    records = [dataset.records[idx] for idx in sorted(indices)]

    meta = dict(dataset.meta)
    meta["sample"] = {
        "mode": "global",
        "seed": int(getattr(args, "seed", 0)),
        "count": getattr(args, "count", None),
        "frac": getattr(args, "frac", None),
        "hardlink": bool(getattr(args, "hardlink", False)),
        "sample_size": len(records),
        "input_size": total,
    }
    return replace(dataset, records=records, meta=meta)
