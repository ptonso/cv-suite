"""Write a VisionDataset to YOLO format.

Writes <split>/images and <split>/labels[_task]/ with YOLO txt labels and a
data.yaml describing the dataset.
"""

from pathlib import Path
import argparse

from cvsuite.common.core import VisionDataset
from cvsuite.common.io import write_dataset
from cvsuite.common.io.attach import attach_link_mode, attach_overwrite_flag
from cvsuite.common.io.folds import fold_targets

from cvsuite.common.stats import attach_with_stats_flag, build_label_stats, emit_stats_yaml, make_export_context
from cvsuite.label.core.schema import write_data_yaml


def _present_splits(dst: Path) -> list[str]:
    """Split trees on disk, whichever invocation wrote them: train, val, test, then the rest."""
    preferred = ("train", "val", "test")
    found = {p.parent.name for p in dst.glob("*/images") if any(p.iterdir())}
    return [name for name in preferred if name in found] + sorted(found - set(preferred))


def _yaml_split_paths(dst: Path) -> dict[str, str]:
    present = _present_splits(dst)
    split_paths = {name: f"{name}/images" for name in present}
    first_path = split_paths[present[0]] if present else "train/images"

    ordered: dict[str, str] = {
        "train": split_paths.get("train", first_path),
        "val": split_paths.get("val", first_path),
    }
    if "test" in split_paths:
        ordered["test"] = split_paths["test"]
    for name in present:
        ordered.setdefault(name, split_paths[name])
    return ordered


def _canonical_task(raw: str) -> str:
    value = str(raw or "auto").strip().lower()
    if value == "seg":
        return "inst-seg"
    return value


def _yaml_task_name(ds, requested: str, boxes_from_polygons: bool = False) -> str:
    if requested == "det":
        return "det"
    if requested == "seg":
        return "seg"
    if requested == "pose":
        return "pose"
    if any(box.kind not in {"pose", "seg"} for record in ds.records for box in record.boxes):
        return "det"
    if boxes_from_polygons and any(record.polys for record in ds.records):
        return "det"
    if any(record.kpts for record in ds.records):
        return "pose"
    if any(record.polys for record in ds.records):
        return "seg"
    return "det"


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument("dst", type=Path, help="Destination YOLO root to create.")
    p.add_argument(
        "--task",
        choices=["auto", "det", "seg", "pose"],
        default="auto",
        help="Task that should own plain labels/; auto exports all detected tasks and prefers det when multiple are present.",
    )
    p.add_argument("--boxes-from-polygons", action="store_true", help="Also write one detection box per polygon, taken from the polygon extent clipped to the image, so polygon-only data can be written as detection.")
    attach_link_mode(p)
    attach_overwrite_flag(p)
    attach_with_stats_flag(p)


def run(ds, a: argparse.Namespace):
    task_arg = getattr(a, "task", "auto")
    boxes_from_polygons = bool(getattr(a, "boxes_from_polygons", False))
    for fold_ds, dst in fold_targets(ds, a.dst):
        write_dataset(
            fold_ds,
            dst,
            format="yolo",
            task=_canonical_task(task_arg),
            link_mode=getattr(a, "link_mode", "copy"),
            boxes_from_polygons=boxes_from_polygons,
            overwrite=bool(getattr(a, "overwrite", False)),
        )
        write_data_yaml(
            dst / "data.yaml",
            root=dst,
            classes=fold_ds.classes,
            task=_yaml_task_name(fold_ds, task_arg, boxes_from_polygons),
            split_paths=_yaml_split_paths(dst),
        )
        if bool(getattr(a, "with_stats", False)):
            ctx = make_export_context(branch="label", command="to-yolo", dst=dst)
            emit_stats_yaml(build_label_stats(fold_ds, ctx), ctx)
    return ds
