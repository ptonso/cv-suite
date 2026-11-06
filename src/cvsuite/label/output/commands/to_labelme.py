"""Write a VisionDataset to LabelMe JSON files.

Emits one JSON per image beside its image. Train-only datasets write directly
under dst/; datasets with other splits write under dst/<split>/. After `split --folds`,
each fold is written under dst/fold<i>/.
"""

from pathlib import Path
import argparse

from cvsuite.common.core import VisionDataset
from cvsuite.common.io import write_dataset
from cvsuite.common.io.attach import attach_link_mode, attach_overwrite_flag
from cvsuite.common.io.folds import fold_targets

from cvsuite.common.stats import attach_with_stats_flag, build_label_stats, emit_stats_yaml, make_export_context


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument("dst", type=Path, help="Destination folder for LabelMe JSONs.")
    p.add_argument("--embed-image", action="store_true", help="Store imageData inside JSON instead of copying images.")
    attach_link_mode(p)
    attach_overwrite_flag(p)
    attach_with_stats_flag(p)


def run(ds, a: argparse.Namespace):
    for fold_ds, fold_dst in fold_targets(ds, a.dst):
        write_dataset(
            fold_ds,
            fold_dst,
            format="labelme",
            embed_image=bool(getattr(a, "embed_image", False)),
            splits=None,
            link_mode=getattr(a, "link_mode", "copy"),
            overwrite=bool(getattr(a, "overwrite", False)),
        )
        if bool(getattr(a, "with_stats", False)):
            ctx = make_export_context(branch="label", command="to-labelme", dst=fold_dst)
            emit_stats_yaml(build_label_stats(fold_ds, ctx), ctx)
    return ds
