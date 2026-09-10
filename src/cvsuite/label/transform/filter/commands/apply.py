"""Apply threshold or logistic filter configs to a labeled dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from cvsuite.common.core import Record
from cvsuite.common.core import VisionDataset

from ..core import (
    FilterConfig,
    LogisticFilter,
    ThresholdFilter,
    filter_config_from_dict,
    load_filter_config,
    load_rgb_for_rec,
)


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="Filter config YAML. Supports threshold and logistic configs.")
    p.add_argument("--det-conf", type=float, default=None, help="Global confidence threshold for box detections.")
    p.add_argument("--seg-conf", type=float, default=None, help="Global confidence threshold for polygon detections.")
    p.add_argument("--nms-iou", type=float, default=None, help="Per-class NMS IoU threshold.")
    p.add_argument("--cross-class-iou", type=float, default=None, help="Cross-class suppression IoU threshold.")
    p.add_argument(
        "--class-det-conf",
        action="append",
        default=[],
        metavar="LABEL=VALUE",
        help="Per-class confidence threshold for box detections. Repeatable.",
    )
    p.add_argument(
        "--class-seg-conf",
        action="append",
        default=[],
        metavar="LABEL=VALUE",
        help="Per-class confidence threshold for polygon detections. Repeatable.",
    )
    p.add_argument("--with-rgb", action="store_true", help="Load images so color-aware configs can use RGB features.")


def _filter_records(ds: VisionDataset, flt, load_rgb: bool) -> List[Record]:
    out: List[Record] = []
    for rec in ds.records:
        rgb = load_rgb_for_rec(rec) if load_rgb else None
        out.append(flt.filter_record(rec, classes=ds.classes, rgb=rgb))
    return out


def _has_inline_thresholds(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.det_conf,
            args.seg_conf,
            args.nms_iou,
            args.cross_class_iou,
        )
    ) or bool(args.class_det_conf or args.class_seg_conf)


def _parse_class_threshold_pairs(values: List[str], *, flag_name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in values:
        label, sep, value = str(raw).partition("=")
        if not sep:
            raise SystemExit(f"{flag_name} entries must look like LABEL=VALUE.")
        label = label.strip()
        if not label:
            raise SystemExit(f"{flag_name} requires a non-empty label before '='.")
        try:
            out[label] = float(value)
        except ValueError as exc:
            raise SystemExit(f"{flag_name} value for '{label}' must be a float.") from exc
    return out


def _build_inline_config(args: argparse.Namespace) -> FilterConfig:
    per_class: dict[str, dict[str, float]] = {}
    for label, value in _parse_class_threshold_pairs(args.class_det_conf, flag_name="--class-det-conf").items():
        per_class.setdefault(label, {})["det_conf"] = value
    for label, value in _parse_class_threshold_pairs(args.class_seg_conf, flag_name="--class-seg-conf").items():
        per_class.setdefault(label, {})["seg_conf"] = value

    data = {
        "global": {
            "det_conf": args.det_conf,
            "seg_conf": args.seg_conf,
            "nms_iou": args.nms_iou,
            "cross_class_iou": args.cross_class_iou,
            "use_rgb": bool(args.with_rgb),
        },
        "per_class": per_class,
    }
    return filter_config_from_dict(data)


def _has_logistic_gates(cfg: FilterConfig) -> bool:
    return any(class_cfg.logistic_gate is not None for class_cfg in cfg.per_class.values())


def run(ds: VisionDataset, args: argparse.Namespace) -> VisionDataset:
    has_config = args.config is not None
    has_inline = _has_inline_thresholds(args)
    if has_config and has_inline:
        raise SystemExit("filter apply accepts either --config or inline threshold flags, not both.")
    if not has_config and not has_inline:
        raise SystemExit("filter apply requires --config or at least one inline threshold flag.")

    cfg = load_filter_config(args.config) if args.config else _build_inline_config(args)
    filter_cls = LogisticFilter if _has_logistic_gates(cfg) else ThresholdFilter
    flt = filter_cls(cfg)
    load_rgb = bool(args.with_rgb or cfg.global_cfg.use_rgb)
    filtered = _filter_records(ds, flt, load_rgb=load_rgb)
    return VisionDataset(records=filtered, classes=ds.classes, task=ds.task, root=ds.root, meta=ds.meta)
