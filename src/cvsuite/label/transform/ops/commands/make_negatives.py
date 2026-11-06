"""Generate background negatives and add them to the in-memory dataset.

With `--emit image` crops are written to a temporary `negatives/` folder under a fresh temp root and new Records
point to them (no annotations). With `--emit box` each window becomes a labeled box on its source record.
"""

from dataclasses import replace
from pathlib import Path
import argparse
from typing import Optional, Set

from ..core.negatives import plan_crops, write_crops
from cvsuite.common.core import BBox, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.io.common import build_in_temp_root


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument("--splits", default="", help="Optional comma-separated splits to take negatives from (default: all). Every record stays in the dataset.")
    p.add_argument("--imgsz", type=int, default=512, help="Square window size in pixels.")
    count = p.add_mutually_exclusive_group()
    count.add_argument("--per-image", type=int, default=None, help="Negatives to draw from each image (default 3 when --ratio is not given).")
    count.add_argument("--ratio", type=float, default=None, help="Negatives per annotated instance of the image: round(R * instances) per image.")
    p.add_argument("--size-from", choices=["fixed", "boxes"], default="fixed", help="`boxes` sizes each window like a random annotation of the same image; `fixed` uses --imgsz.")
    p.add_argument("--emit", choices=["image", "box"], default="image", help="`image` writes each window as a new image record; `box` adds it as a box labeled --label on its source record.")
    p.add_argument("--label", default=None, help="Class of the boxes added by --emit box.")
    p.add_argument("--within", choices=["image", "annotations"], default="image", help="`annotations` keeps windows inside the extent of the image's annotations.")
    p.add_argument("--on-shortfall", choices=["error", "warn"], default="error", help="What to do when an image cannot hold all its negatives without touching an annotation.")
    p.add_argument("--seed", type=int, default=0, help="Seed for deterministic placement.")
    p.add_argument("--preview", type=Path, help="Optional path for a collage preview of sampled crops.")


def run(ds: VisionDataset, a: argparse.Namespace) -> VisionDataset:
    emit = getattr(a, "emit", "image")
    if emit == "box" and not getattr(a, "label", None):
        raise SystemExit("--emit box needs --label.")
    if emit == "image" and getattr(a, "label", None):
        raise SystemExit("--label belongs to --emit box.")
    if emit == "box" and getattr(a, "preview", None):
        raise SystemExit("--preview needs --emit image.")
    if emit == "box":
        return _make_boxes(ds, a)
    return build_in_temp_root("vislabel_neg_", lambda tmp_root: _make_negatives_into(tmp_root, ds, a))


def _plan(ds: VisionDataset, a: argparse.Namespace, *, avoid_placed: bool):
    splits: Optional[Set[str]] = {s.strip() for s in getattr(a, "splits", "").split(",") if s.strip()} or None
    per_image, ratio = getattr(a, "per_image", None), getattr(a, "ratio", None)
    if per_image is None and ratio is None:
        per_image = 3
    if (per_image is not None and per_image < 0) or (ratio is not None and ratio < 0):
        raise SystemExit("--per-image and --ratio must be non-negative.")

    crops, shortfalls = plan_crops(
        ds,
        imgsz=a.imgsz,
        per_image=per_image,
        ratio=ratio,
        size_from=getattr(a, "size_from", "fixed"),
        seed=a.seed,
        splits=splits,
        within=getattr(a, "within", "image"),
        avoid_placed=avoid_placed,
    )
    if shortfalls:
        listing = "\n".join(f"  {path}: wanted {wanted}, placed {placed}" for path, wanted, placed in shortfalls)
        message = f"{len(shortfalls)} image(s) could not hold all their negatives without touching an annotation:\n{listing}"
        if getattr(a, "on_shortfall", "error") == "error":
            raise SystemExit(message + "\nUse --on-shortfall warn to continue with fewer, or a smaller --imgsz.")
        print("warning: " + message)
    return crops, shortfalls


def _make_boxes(ds: VisionDataset, a: argparse.Namespace) -> VisionDataset:
    crops, shortfalls = _plan(ds, a, avoid_placed=True)
    classes = list(ds.classes)
    if a.label not in classes:
        classes.append(a.label)
    cls = classes.index(a.label)
    added: dict[int, list[BBox]] = {}
    for crop in crops:
        record = ds.records[crop.index]
        x1, y1, x2, y2 = crop.box_xyxy
        width, height = record.image.width, record.image.height
        added.setdefault(crop.index, []).append(
            BBox((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height, cls, a.label, attributes={"negative": True})
        )
    records = [replace(r, boxes=[*r.boxes, *added.get(i, [])]) if i in added else r for i, r in enumerate(ds.records)]
    meta = dict(ds.meta)
    meta["step_info"] = {"negatives": len(crops), "images_short": len(shortfalls)}
    return replace(ds, records=records, classes=classes, meta=meta)


def _make_negatives_into(tmp_root: Path, ds: VisionDataset, a: argparse.Namespace) -> VisionDataset:
    crops, shortfalls = _plan(ds, a, avoid_placed=False)
    written = write_crops(crops, tmp_root, getattr(a, "preview", None))

    new_records = list(ds.records)
    for crop, path in zip(crops, written):
        x1, y1, x2, y2 = crop.box_xyxy
        new_records.append(
            Record(
                image=ImageRecord(path=path, width=x2 - x1, height=y2 - y1),
                split=crop.split,
                task=ds.task,
                attributes={"negative": True, "source_image": str(crop.src_image), "negative_window_px": [x1, y1, x2, y2]},
            )
        )

    meta = dict(ds.meta)
    meta["negatives_root"] = tmp_root
    meta["step_info"] = {"negatives": len(crops), "images_short": len(shortfalls)}
    return VisionDataset(records=new_records, classes=ds.classes, task=ds.task, root=ds.root, meta=meta)
