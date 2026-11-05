"""Write a VisionDataset to YOLO format.

Writes <split>/images and <split>/labels[_task]/ with YOLO txt labels and a
data.yaml describing the dataset. Supports hard-linking images instead of
copying bytes.
"""

from pathlib import Path
import argparse

from cvsuite.common.core import VisionDataset
from cvsuite.common.io import write_dataset
from cvsuite.common.io.attach import attach_overwrite_flag

from cvsuite.common.stats import attach_with_stats_flag, build_label_stats, emit_stats_yaml, make_export_context
from cvsuite.label.core.schema import write_data_yaml
from cvsuite.label.output.common import assign_output_splits, attach_split_assignment_args


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


def _yaml_task_name(ds, requested: str) -> str:
    if requested == "det":
        return "det"
    if requested == "seg":
        return "seg"
    if requested == "pose":
        return "pose"
    if any(box.kind not in {"pose", "seg"} for record in ds.records for box in record.boxes):
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
    attach_split_assignment_args(p)
    p.add_argument("--hardlink", action="store_true", help="Hard-link images instead of copying bytes.")
    attach_overwrite_flag(p)
    attach_with_stats_flag(p)


def run(ds, a: argparse.Namespace):
    if not bool(getattr(a, "preserve_splits", False)):
        ds = assign_output_splits(
            ds,
            val_frac=getattr(a, "val_frac", 0.0),
            test_frac=getattr(a, "test_frac", 0.0),
            seed=getattr(a, "seed", 0),
        )
    task_arg = getattr(a, "task", "auto")
    write_dataset(
        ds,
        a.dst,
        format="yolo",
        task=_canonical_task(task_arg),
        hardlink=bool(getattr(a, "hardlink", False)),
        overwrite=bool(getattr(a, "overwrite", False)),
    )
    write_data_yaml(
        a.dst / "data.yaml",
        root=a.dst,
        classes=ds.classes,
        task=_yaml_task_name(ds, task_arg),
        split_paths=_yaml_split_paths(a.dst),
    )
    if bool(getattr(a, "with_stats", False)):
        ctx = make_export_context(branch="label", command="to-yolo", dst=a.dst)
        emit_stats_yaml(build_label_stats(ds, ctx), ctx)
    return ds
