"""Write a VisionDataset to LabelMe JSON files.

Emits one JSON per image beside its image. Train-only exports write directly
under dst/; split exports write under dst/<split>/.
"""

from pathlib import Path
import argparse

from cvsuite.common.core import VisionDataset
from cvsuite.common.io import write_dataset

from cvsuite.common.stats import attach_with_stats_flag, build_label_stats, emit_stats_yaml, make_export_context
from cvsuite.label.output.common import assign_output_splits, attach_split_assignment_args


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument("dst", type=Path, help="Destination folder for LabelMe JSONs.")
    attach_split_assignment_args(p)
    p.add_argument("--embed-image", action="store_true", help="Store imageData inside JSON instead of copying images.")
    p.add_argument("--hardlink", action="store_true", help="Hard-link images instead of copying bytes.")
    attach_with_stats_flag(p)


def run(ds, a: argparse.Namespace):
    preserve_splits = bool(getattr(a, "preserve_splits", False))
    val_frac = float(getattr(a, "val_frac", 0.0) or 0.0)
    test_frac = float(getattr(a, "test_frac", 0.0) or 0.0)
    split_layout = preserve_splits or bool(val_frac or test_frac)
    if not preserve_splits:
        ds = assign_output_splits(
            ds,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=getattr(a, "seed", 0),
        )
    write_dataset(
        ds,
        a.dst,
        format="labelme",
        embed_image=bool(getattr(a, "embed_image", False)),
        splits=None,
        split_layout=split_layout,
        hardlink=bool(getattr(a, "hardlink", False)),
    )
    if bool(getattr(a, "with_stats", False)):
        ctx = make_export_context(branch="label", command="to-labelme", dst=a.dst)
        emit_stats_yaml(build_label_stats(ds, ctx), ctx)
    return ds
