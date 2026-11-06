"""Merge, rename and drop classes by a YAML map. Contract: `specs/common/select_remap.md#remap-classes`."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml

from cvsuite.common.core import VisionDataset
from cvsuite.common.transforms import vocab

SECTIONS = ("merge", "rename", "drop")


def attach(parser: argparse.ArgumentParser, *, empty_flag: bool = True) -> None:
    parser.add_argument("--map", type=Path, required=True, help="YAML file with merge, rename and drop sections.")
    parser.add_argument("--others", default=None, metavar="NAME", help="Class that receives every class the map does not mention.")
    if empty_flag:
        parser.add_argument("--empty", choices=vocab.EMPTY_CHOICES, default=None, help="Records left without annotations: drop them or keep them as images. Required when classes are dropped from annotated data.")


def load_map(path: Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or set(data) - set(SECTIONS):
        raise ValueError(f"{path} must be a mapping with only these sections: {', '.join(SECTIONS)}.")
    merge, rename, drop = data.get("merge") or {}, data.get("rename") or {}, data.get("drop") or []
    if not isinstance(merge, dict) or not isinstance(rename, dict) or not isinstance(drop, list):
        raise ValueError("merge and rename are mappings and drop is a list.")
    for new, sources in merge.items():
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"merge.{new} must list the classes it replaces.")
    return {"merge": {str(k): [str(x) for x in v] for k, v in merge.items()}, "rename": {str(k): str(v) for k, v in rename.items()}, "drop": [str(x) for x in drop]}


def plan(classes: list[str], mapping: dict, others: str | None) -> tuple[dict[str, str | None], list[str]]:
    """The old-name to new-name map (None drops) and the new class list."""
    name_map: dict[str, str | None] = {}
    mentioned: list[str] = []
    for new, sources in mapping["merge"].items():
        for source in sources:
            name_map[source] = new
            mentioned.append(source)
    for source, new in mapping["rename"].items():
        name_map[source] = new
        mentioned.append(source)
    for source in mapping["drop"]:
        name_map[source] = None
        mentioned.append(source)

    repeated = sorted({m for m in mentioned if mentioned.count(m) > 1})
    if repeated:
        raise ValueError(f"These classes appear more than once in the map: {', '.join(repeated)}.")
    absent = [m for m in mentioned if m not in classes]
    if absent:
        raise ValueError(f"The map names classes that the dataset does not have: {', '.join(absent)}. Known classes: {', '.join(classes)}.")

    created = [*mapping["merge"], *mapping["rename"].values()]
    if len(set(created)) != len(created):
        raise ValueError(f"The map creates the same class name twice: {created}.")
    for new in created:
        if new in classes and name_map.get(new) != new:
            raise ValueError(f"The map creates `{new}`, which is also an existing class that does not feed it. List `{new}` among its own sources.")
    unmapped = [c for c in classes if c not in name_map]
    if others is not None:
        if others in created:
            raise ValueError(f"--others `{others}` is also created by the map.")
        if others in classes and others in name_map:
            raise ValueError(f"--others `{others}` is an existing class that the map moves elsewhere.")
        name_map.update({c: others for c in unmapped})
        unmapped = []
    new_classes = [*mapping["merge"], *mapping["rename"].values(), *unmapped]
    if others is not None and any(v == others for v in name_map.values()):
        new_classes.append(others)
    return name_map, list(dict.fromkeys(new_classes))


def remap(dataset: VisionDataset, mapping: dict, *, others: str | None = None, empty: str | None = None) -> tuple[VisionDataset, dict]:
    vocab.reject_masks(dataset, "remap-classes")
    classes = list(dataset.classes)
    if not classes:
        raise ValueError("remap-classes needs a dataset with a class list.")
    name_map, new_classes = plan(classes, mapping, others)
    if mapping["drop"]:
        vocab.require_empty(dataset, empty, "remap-classes with a drop section")
    out, stats = vocab.reindex_classes(dataset, name_map, new_classes, empty=empty)
    info = {"old_to_new": {k: v for k, v in name_map.items() if v != k}, "classes": new_classes, **stats}
    return out, info


def run(dataset: VisionDataset, args: argparse.Namespace) -> VisionDataset:
    out, info = remap(dataset, load_map(args.map), others=args.others, empty=getattr(args, "empty", None))
    return replace(out, meta={**out.meta, "step_info": info})
