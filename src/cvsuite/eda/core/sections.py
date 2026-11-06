"""The section registry and the context every section runs in. Contract: `specs/eda/cli.md#which-sections-run`."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Callable

from cvsuite.common.core import VisionDataset
from cvsuite.common.keys import record_keys
from cvsuite.eda.core.table import ImageTable

NAMES = ("profile", "structure", "health", "split", "leak")


@dataclass
class Context:
    dataset: VisionDataset
    table: ImageTable
    top: int = 20
    seed: int = 0
    group_by: list[str] = field(default_factory=list)
    max_images: int = 20000
    measure: str = "hash"
    twin_threshold: float | None = None
    omitted: list[dict] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    notes: dict[str, dict] = field(default_factory=dict)

    def omit(self, file: str, reason: str) -> None:
        self.omitted.append({"file": file, "reason": reason})

    def subset(self, indices: list[int]) -> "Context":
        dataset = replace(self.dataset, records=[self.dataset.records[i] for i in indices])
        table = ImageTable([self.table.rows[i] for i in indices], [self.table.thumbs[i] for i in indices], self.table.unreadable)
        return replace(self, dataset=dataset, table=table, omitted=self.omitted, samples=self.samples)

    def has_geometry(self) -> bool:
        return any(r.boxes or r.polys for r in self.dataset.records)


def split_order(splits: set[str]) -> list[str]:
    return [s for s in ("train", "val", "test") if s in splits] + sorted(splits - {"train", "val", "test"})


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "group"


def sampling_keys(dataset: VisionDataset) -> list[str]:
    """Sampling keys that tolerate one file listed twice (for example in two splits), numbering the repeats."""
    seen: dict[str, int] = defaultdict(int)
    keys = []
    for key in record_keys(dataset):
        seen[key] += 1
        keys.append(key if seen[key] == 1 else f"{key}#{seen[key]}")
    return keys


def _distinct_labels(ctx: Context) -> int:
    labels: set[str] = set()
    for row in ctx.table.rows:
        labels.update(x for x in row["class"].split(";") if x and x != "(empty)")
    return len(labels)


def needs_split(ctx: Context) -> str | None:
    splits = {row["split"] for row in ctx.table.rows}
    if len(splits) < 2 or "train" not in splits:
        return f"needs at least two splits including train, found: {', '.join(sorted(splits))}"
    return None


def needs_leak(ctx: Context) -> str | None:
    return None if _distinct_labels(ctx) >= 2 else "needs at least two distinct labels"


def sections() -> dict[str, tuple[ModuleType, Callable[[Context], str | None]]]:
    """Section name to its module and its applicability rule."""
    from cvsuite.eda.core import health, leak, profile, split, structure

    return {
        "profile": (profile, profile.needs),
        "structure": (structure, structure.needs),
        "health": (health, health.needs),
        "split": (split, split.needs),
        "leak": (leak, leak.needs),
    }
