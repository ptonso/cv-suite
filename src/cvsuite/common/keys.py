"""Key specs that read one value per record and group records, as described in `specs/common/keys.md`."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import numpy as np

from cvsuite.common.core import VisionDataset, VisionRecord
from cvsuite.common.io.common import resolve_record_image_path

KINDS = ("regex", "dir", "attr", "file", "similar")
BARE = ("split",)
LAYOUT_FOLDERS = frozenset({"train", "val", "valid", "validation", "test", "eval", "images"})
_REPORTED = 5


def layout_path(record: VisionRecord, dataset: VisionDataset) -> str:
    if record.rel_image_path is not None:
        rel = PurePosixPath(Path(record.rel_image_path).as_posix())
    elif record.image is None:
        raise ValueError("A record without an image has no layout path.")
    else:
        path = Path(record.image.path)
        if path.is_absolute():
            under_root = dataset.root is not None and path.is_relative_to(dataset.root)
            path = path.relative_to(dataset.root) if under_root else Path(path.name)
        rel = PurePosixPath(path.as_posix())
    parts = list(rel.parts)
    while len(parts) > 1 and parts[0] in LAYOUT_FOLDERS:
        parts.pop(0)
    return "/".join(parts)


@dataclass(frozen=True)
class Key:
    spec: str
    kind: str
    resolve: Callable[[VisionDataset, list[str]], list[str | None]] | None

    def values(self, dataset: VisionDataset) -> list[str]:
        if self.resolve is None:
            raise ValueError(f"`{self.spec}` groups records and cannot give one value per record. Use it with --group-by only.")
        paths = [layout_path(record, dataset) for record in dataset.records]
        found = self.resolve(dataset, paths)
        missing = [path for path, value in zip(paths, found) if value is None]
        if missing:
            shown = ", ".join(missing[:_REPORTED])
            raise ValueError(f"Key `{self.spec}` does not resolve for {len(missing)} record(s), for example: {shown}.")
        return [str(value) for value in found]


def _regex(arg: str) -> Callable:
    try:
        pattern = re.compile(arg)
    except re.error as exc:
        raise ValueError(f"Invalid regex in key `regex:{arg}`: {exc}") from exc

    def resolve(_dataset, paths):
        out = []
        for path in paths:
            match = pattern.search(path)
            out.append(None if match is None else (match.group(1) if pattern.groups else match.group(0)))
        return out

    return resolve


def _dir(arg: str) -> Callable:
    try:
        index = int(arg)
    except ValueError as exc:
        raise ValueError(f"Key `dir:{arg}` needs an integer depth.") from exc

    def resolve(_dataset, paths):
        out = []
        for path in paths:
            folders = path.split("/")[:-1]
            out.append(folders[index] if -len(folders) <= index < len(folders) else None)
        return out

    return resolve


def _attr(arg: str) -> Callable:
    def resolve(dataset, _paths):
        return [str(r.attributes[arg]) if arg in r.attributes else None for r in dataset.records]

    return resolve


def _file(arg: str) -> Callable:
    table_path = Path(arg)
    if not table_path.is_file():
        raise ValueError(f"Key table not found: {table_path}")
    table: dict[str, str] = {}
    with table_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 2:
                raise ValueError(f"Key table {table_path} must have two columns, found a row with {len(row)}: {row}")
            if row[0] in table:
                raise ValueError(f"Key table {table_path} repeats `{row[0]}` in its first column.")
            table[row[0]] = row[1]

    def resolve(_dataset, paths):
        names = Counter(path.rsplit("/", 1)[-1] for path in paths)
        out = []
        for path in paths:
            name = path.rsplit("/", 1)[-1]
            out.append(table[path] if path in table else (table[name] if name in table and names[name] == 1 else None))
        return out

    return resolve


_BUILDERS = {"regex": _regex, "dir": _dir, "attr": _attr, "file": _file}


def parse_key(spec: str) -> Key:
    if spec == "split":
        return Key(spec, "split", lambda dataset, _paths: [record.split or "train" for record in dataset.records])
    kind, sep, arg = spec.partition(":")
    if not sep or kind not in KINDS:
        raise ValueError(f"Unknown key spec `{spec}`. Valid kinds: {', '.join([f'{k}:...' for k in KINDS] + list(BARE))}.")
    if kind == "similar":
        _similar_bits(arg)
        return Key(spec, kind, None)
    return Key(spec, kind, _BUILDERS[kind](arg))


def _similar_bits(arg: str) -> int:
    from cvsuite.common.similarity import check_bits

    try:
        return check_bits(int(arg))
    except ValueError as exc:
        raise ValueError(f"Key `similar:{arg}` needs a number of bits in [0, 64]: {exc}") from exc


def record_keys(dataset: VisionDataset) -> list[str]:
    """Stable per-record keys: layout paths, with the resolved image path appended where several sources repeat one."""
    keys = [layout_path(record, dataset) for record in dataset.records]
    counts = Counter(keys)
    return [
        f"{key}|{resolve_record_image_path(dataset, record).resolve()}" if counts[key] > 1 else key
        for key, record in zip(keys, dataset.records)
    ]


def group_records(dataset: VisionDataset, specs: list[str], *, bits: Optional[np.ndarray] = None) -> list[str]:
    """One group id per record: the smallest layout path in the component the specs connect.

    Records that share a layout path are one image and share a group id. `bits` are the records' hashes when the caller already holds them.
    """
    from cvsuite.common import similarity

    keys = [parse_key(spec) for spec in specs]
    names = [layout_path(record, dataset) for record in dataset.records]
    parent = list(range(len(names)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first_name: dict[str, int] = {}
    for index, name in enumerate(names):
        union(first_name.setdefault(name, index), index)

    for key in keys:
        if key.kind == "similar":
            hashes = bits if bits is not None else similarity.hash_bits([resolve_record_image_path(dataset, record) for record in dataset.records])
            for a, b in similarity.pairs_within(hashes, _similar_bits(key.spec.partition(":")[2])):
                union(a, b)
            continue
        first: dict[str, int] = {}
        for index, value in enumerate(key.values(dataset)):
            union(first.setdefault(value, index), index)

    smallest: dict[int, str] = {}
    for index, name in enumerate(names):
        root = find(index)
        smallest[root] = min(smallest.get(root, name), name)
    return [smallest[find(index)] for index in range(len(names))]
