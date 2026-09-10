"""Preview overlays for a VisionDataset in an interactive window (same visuals as results)."""

import argparse

from cvsuite.label.core.annotate import run_preview


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument("--split", default="", help="Optional comma-separated splits to visualize (default: all).")
    p.add_argument("--task", choices=["all", "det", "seg", "pose"], default="all", help="Annotation type to visualize.")
    p.add_argument("--imgsz", type=int, default=None, help="Square size to letterbox previews to; default keeps native size.")
    p.add_argument("--max", type=int, default=0, help="Maximum images to preview (0 means no limit).")
    p.add_argument(
        "--names-mode",
        choices=["smart", "none", "all", "det", "seg"],
        default="smart",
        help="Control where class names are drawn.",
    )
    p.add_argument("--show-confidence", action="store_true", help="Append per-annotation confidence to drawn labels.")
    p.add_argument("--show-prompt", action="store_true", help="Draw the winning prompt beneath each label when present.")


def _filter_records(ds, splits, task_filter):
    splits_set = set(splits) if splits else None
    out_records = []
    for rec in ds.records:
        if splits_set and rec.split not in splits_set:
            continue
        if task_filter == "all":
            out_records.append(rec)
        elif task_filter == "det" and any(b.kind != "pose" for b in rec.boxes):
            out_records.append(rec)
        elif task_filter == "seg" and rec.polys:
            out_records.append(rec)
        elif task_filter == "pose" and rec.kpts:
            out_records.append(rec)
    ds.records = out_records
    return ds


def run(ds, a: argparse.Namespace):
    splits = [s.strip() for s in a.split.split(",") if s.strip()] if a.split else None

    ds = _filter_records(ds, splits, a.task)

    run_preview(
        ds,
        imgsz=a.imgsz,
        max_items=a.max,
        names_mode=a.names_mode,
        show_confidence=bool(getattr(a, "show_confidence", False)),
        show_prompt=bool(getattr(a, "show_prompt", False)),
    )
    return ds
