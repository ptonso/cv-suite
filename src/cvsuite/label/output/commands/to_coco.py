"""Write a VisionDataset to a COCO dataset directory."""

from pathlib import Path
import argparse

from cvsuite.common.core import VisionDataset
from cvsuite.common.io import write_dataset
from cvsuite.common.io.attach import attach_link_mode, attach_overwrite_flag
from cvsuite.common.io.folds import fold_targets

from cvsuite.common.core.utils import write_yaml
from cvsuite.common.stats import attach_with_stats_flag, build_label_stats, emit_stats_yaml, make_export_context


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument("dst", type=Path, help="Output COCO dataset directory to create.")
    p.add_argument("--coco-style", action="store_true", help="Write images/ + annotations/annotations_<split>.json layout.")
    p.add_argument("--no-images", action="store_true", help="Write annotations + data.yaml only; do not copy image files.")
    attach_link_mode(p)
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
    if a.dst.suffix:
        raise SystemExit(f"to-coco writes a directory, not a file: {a.dst}")
    coco_style = bool(getattr(a, "coco_style", False))
    for fold_ds, dst in fold_targets(ds, a.dst):
        dst.mkdir(parents=True, exist_ok=True)
        write_dataset(
            fold_ds,
            dst,
            format="coco",
            splits=list(fold_ds.by_split().keys()) or ["train"],
            overwrite=bool(getattr(a, "overwrite", False)),
            coco_style=coco_style,
            link_mode=getattr(a, "link_mode", "copy"),
            no_images=bool(getattr(a, "no_images", False)),
        )
        data_obj = {
            "path": ".",
            "format": "coco",
            "task": getattr(fold_ds.task, "value", fold_ds.task) if fold_ds.task else "det",
            "names": fold_ds.classes,
        }
        for split, path in _annotation_paths(dst, coco_style).items():
            data_obj[split] = str(path.relative_to(dst))
        write_yaml(dst / "data.yaml", data_obj, overwrite=True)
        if bool(getattr(a, "with_stats", False)):
            ctx = make_export_context(branch="label", command="to-coco", dst=dst)
            emit_stats_yaml(build_label_stats(fold_ds, ctx), ctx)
    return ds
