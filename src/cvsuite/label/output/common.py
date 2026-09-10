from __future__ import annotations

import argparse
import random
from dataclasses import replace

from cvsuite.common.core import VisionDataset


def attach_split_assignment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.0,
        help="Fraction to assign to the validation split before writing.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.0,
        help="Fraction to assign to the test split before writing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when assigning output splits.",
    )
    parser.add_argument(
        "--preserve-splits",
        action="store_true",
        help="Keep existing record splits instead of assigning new output splits.",
    )

def assign_output_splits(
    dataset: VisionDataset,
    *,
    val_frac: float = 0.0,
    test_frac: float = 0.0,
    seed: int = 0,
) -> VisionDataset:
    val_frac = float(val_frac or 0.0)
    test_frac = float(test_frac or 0.0)

    if val_frac < 0 or test_frac < 0:
        raise ValueError("Split fractions must be non-negative.")
    if val_frac > 1 or test_frac > 1 or (val_frac + test_frac) > 1:
        raise ValueError("Split fractions must satisfy val_frac + test_frac <= 1.0.")

    rng = random.Random(seed)
    records = []
    for rec in dataset.records:
        draw = rng.random()
        if draw < test_frac:
            split = "test"
        elif draw < test_frac + val_frac:
            split = "val"
        else:
            split = "train"
        records.append(replace(rec, split=split))

    meta = dict(dataset.meta)
    meta["output_split_assignment"] = {
        "val_frac": val_frac,
        "test_frac": test_frac,
        "seed": seed,
    }
    return replace(dataset, records=records, meta=meta)
