"""Nested sampling: a seeded hash ranks items, so a smaller sample is a subset of a larger one. Contract: `specs/common/sampling.md`."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from cvsuite.common.core import VisionDataset
from cvsuite.common.keys import parse_key


def rank_order(keys: Sequence[str], seed: int) -> list[int]:
    if len(set(keys)) != len(keys):
        duplicates = sorted({k for k in keys if list(keys).count(k) > 1})
        raise ValueError(f"Sampling needs unique keys, found duplicates such as: {', '.join(duplicates[:5])}")
    ranks = [hashlib.sha1(f"{seed}:{key}".encode()).hexdigest() for key in keys]
    return sorted(range(len(keys)), key=ranks.__getitem__)


def take(keys: Sequence[str], seed: int, k: int) -> list[int]:
    return sorted(rank_order(keys, seed)[:k])


def sample_size(total: int, *, count: int | None, frac: float | None) -> int:
    """`--count N` takes N; `--frac F` takes floor(total * F), at least one for a positive fraction."""
    if count is not None:
        if count < 0:
            raise SystemExit("--count must be non-negative.")
        if count > total:
            raise SystemExit(f"Requested --count {count} but only {total} items are available.")
        return count
    assert frac is not None
    if not 0.0 <= frac <= 1.0:
        raise SystemExit("--frac must be between 0 and 1.")
    chosen = math.floor(total * frac)
    return min(total, max(1, chosen)) if frac > 0.0 and total > 0 else chosen


def strata(values: Sequence[Sequence[str]]) -> list[list[int]]:
    """Record indices of each stratum, strata in sorted value order."""
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, value in enumerate(values):
        groups.setdefault(tuple(value), []).append(index)
    return [groups[value] for value in sorted(groups)]


def stratum_values(dataset: VisionDataset, spec: str | None) -> list[tuple[str, ...]]:
    """The stratum value of every record for `--stratify-by KEY[,KEY...]`; one stratum without the option."""
    names = [name.strip() for name in (spec or "").split(",") if name.strip()]
    if "class" in names:
        raise SystemExit("--stratify-by takes keys (`split`, `dir:N`, `regex:`, `attr:`, `file:`). `class sample` already samples per class, and an annotated record holds several classes.")
    columns = [parse_key(name).values(dataset) for name in names]
    return [tuple(column[i] for column in columns) for i in range(len(dataset.records))]
