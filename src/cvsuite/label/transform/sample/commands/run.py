"""Sample a VisionDataset, globally or inside each stratum, while preserving annotations."""

from __future__ import annotations

import argparse
from dataclasses import replace

from cvsuite.common.keys import record_keys
from cvsuite.common.sampling import sample_size, strata, stratum_values, take


def attach(parser: argparse.ArgumentParser) -> None:
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--count", type=int, default=None, help="Number of records to sample (from every stratum with --stratify-by).")
    size_group.add_argument("--frac", type=float, default=None, help="Fraction of records to sample (of every stratum with --stratify-by).")
    parser.add_argument("--stratify-by", default=None, metavar="KEY[,KEY...]", help="Sample inside each group of records that share these keys: split, dir:N, regex:PATTERN, attr:NAME or file:PATH.csv.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic sampling.")


def run(dataset, args: argparse.Namespace):
    total = len(dataset.records)
    keys = record_keys(dataset)
    values = stratum_values(dataset, getattr(args, "stratify_by", None))
    groups = strata(values)
    chosen: list[int] = []
    for members in groups:
        count = getattr(args, "count", None)
        if values[members[0]] and count is not None and count > len(members):
            raise SystemExit(f"Stratum `{'|'.join(values[members[0]])}` holds {len(members)} records, fewer than --count {count}.")
        target = sample_size(len(members), count=getattr(args, "count", None), frac=getattr(args, "frac", None))
        chosen.extend(members[i] for i in take([keys[m] for m in members], int(getattr(args, "seed", 0)), target))
    records = [dataset.records[i] for i in sorted(chosen)]
    meta = dict(dataset.meta)
    meta["step_info"] = {"mode": "stratified" if getattr(args, "stratify_by", None) else "global", "sample_size": len(records), "input_size": total, "strata": len(groups)}
    return replace(dataset, records=records, meta=meta)
