"""Write a VisionDataset to a COCO dataset directory."""

from pathlib import Path
import argparse

from cvsuite.common.core import VisionDataset
from cvsuite.common.io import write_dataset
from cvsuite.common.io.attach import attach_overwrite_flag

from cvsuite.common.core.utils import write_yaml
from cvsuite.common.stats import attach_with_stats_flag, build_label_stats, emit_stats_yaml, make_export_context
from cvsuite.label.output.common import assign_output_splits, attach_split_assignment_args


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument("dst", type=Path, help="Output COCO dataset directory to create.")
    attach_split_assignment_args(p)
    p.add_argument("--coco-style", action="store_true", help="Write images/ + annotations/annotations_<split>.json layout.")
    p.add_argument("--no-images", action="store_true", help="Write annotations + data.yaml only; do not copy image files.")
    p.add_argument("--hardlink", action="store_true", help="Hard-link images instead of copying bytes.")
    attach_overwrite_flag(p)
    attach_with_stats_flag(p)


def _annotation_paths(dst: Path, coco_style: bool) -> dict[str, Path]:
    """Annotation files on disk, whichever invocation wrote them: train, val, test, then the rest."""
    preferred = ("train", "val", "test")
    if coco_style:
        found = {p.stem.removeprefix("annotations_"): p for p in (dst / "annotations").glob("annotations_*.json")}
    else:
        found = {p.parent.name: p for p in dst.glob("*/_annotations.coco.json")}
    return {name: found[name] for name in [*(n for n in preferred if n in found), *sorted(set(found) - set(preferred))]}


def run(ds, a: argparse.Namespace):
    dst: Path = a.dst
    if dst.suffix:
        raise SystemExit(f"to-coco writes a directory, not a file: {dst}")

    if not bool(getattr(a, "preserve_splits", False)):
        ds = assign_output_splits(
            ds,
            val_frac=getattr(a, "val_frac", 0.0),
            test_frac=getattr(a, "test_frac", 0.0),
            seed=getattr(a, "seed", 0),
        )
    splits = list(ds.by_split().keys()) or ["train"]

    dst.mkdir(parents=True, exist_ok=True)
    coco_style = bool(getattr(a, "coco_style", False))
    no_images = bool(getattr(a, "no_images", False))
    write_dataset(
        ds,
        dst,
        format="coco",
        splits=splits,
        overwrite=bool(getattr(a, "overwrite", False)),
        coco_style=coco_style,
        hardlink=bool(getattr(a, "hardlink", False)),
        no_images=no_images,
    )
    ann_paths = _annotation_paths(dst, coco_style)

    # Emit a helper data.yaml for organization (format: coco)
    data_obj = {
        "path": ".",
        "format": "coco",
        "task": getattr(ds.task, "value", ds.task) if ds.task else "det",
        "names": ds.classes,
    }
    for sp, pth in ann_paths.items():
        data_obj[sp] = str(pth.relative_to(dst))
    write_yaml(dst / "data.yaml", data_obj, overwrite=True)
    if bool(getattr(a, "with_stats", False)):
        ctx = make_export_context(branch="label", command="to-coco", dst=a.dst)
        emit_stats_yaml(build_label_stats(ds, ctx), ctx)
    return ds
