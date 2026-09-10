from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
import yaml

from .types import (
    ClassThresholds,
    FilterConfig,
    GlobalConfig,
    LogisticGate,
)


def _parse_logistic_gate(obj: Dict[str, Any]) -> LogisticGate:
    if obj is None:
        return LogisticGate()
    return LogisticGate(
        features=list(obj.get("features", []) or []),
        mean=None if obj.get("mean", None) is None else list(obj.get("mean")),  # type: ignore[arg-type]
        std=None if obj.get("std", None) is None else list(obj.get("std")),    # type: ignore[arg-type]
        weights=list(obj.get("weights", []) or []),
        bias=float(obj.get("bias", 0.0) or 0.0),
        threshold_type=str(obj.get("threshold_type", "logit")),
        threshold_value=float(obj.get("threshold_value", 0.0) or 0.0),
    )


def _parse_class_thresholds(obj: Dict[str, Any]) -> ClassThresholds:
    if obj is None:
        obj = {}
    out = ClassThresholds()
    for k in (
        "det_conf",
        "seg_conf",
        "post_conf",
        "min_box_area_frac",
        "max_box_area_frac",
        "min_short_side_frac",
        "min_ar",
        "max_ar",
        "mask_min_per_box",
        "mask_min_circ",
        "mask_min_solidity",
        "mask_max_holes_frac",
        "mask_min_ar",
        "mask_max_ar",
        "white_min_frac",
        "black_max_frac",
        "yellow_min_frac",
        "red_min_frac",
        "green_min_frac",
        "blue_min_frac",
    ):
        if k in obj:
            setattr(out, k, None if obj[k] is None else float(obj[k]))
    if "require_mask" in obj:
        out.require_mask = bool(obj.get("require_mask", False))
    if obj.get("logistic_gate", None) is not None:
        out.logistic_gate = _parse_logistic_gate(obj["logistic_gate"])
    return out


def _parse_global(obj: Dict[str, Any]) -> GlobalConfig:
    if obj is None:
        obj = {}
    return GlobalConfig(
        nms_iou=float(obj.get("nms_iou", 0.6) or 0.6),
        topk_per_class=None if obj.get("topk_per_class", None) in (None, "") else int(obj.get("topk_per_class")),
        cross_class_iou=None if obj.get("cross_class_iou", None) in (None, "") else float(obj.get("cross_class_iou")),
        require_mask=bool(obj.get("require_mask", False)),
        det_conf=None if obj.get("det_conf", None) in (None, "") else float(obj.get("det_conf")),
        seg_conf=None if obj.get("seg_conf", None) in (None, "") else float(obj.get("seg_conf")),
        use_rgb=bool(obj.get("use_rgb", False)),
    )


def filter_config_from_dict(data: Dict[str, Any]) -> FilterConfig:
    g = _parse_global(data.get("global", {}) or data.get("global_cfg", {}))
    per_class_cfg: Dict[str, ClassThresholds] = {}
    for cls, cfg in (data.get("per_class", {}) or {}).items():
        per_class_cfg[str(cls)] = _parse_class_thresholds(cfg or {})
    return FilterConfig(global_cfg=g, per_class=per_class_cfg)


def load_filter_config(path: str | Path) -> FilterConfig:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return filter_config_from_dict(data)


def _example_dict() -> Dict[str, Any]:
    return {
        "global": {
            "nms_iou": 0.6,
            "topk_per_class": 50,
            "cross_class_iou": 0.25,
            "require_mask": False,
            "det_conf": 0.2,
            "seg_conf": 0.2,
            "use_rgb": False,
        },
        "per_class": {
            "class_name": {
                "det_conf": 0.25,
                "seg_conf": 0.25,
                "post_conf": 0.2,
                "min_box_area_frac": 0.0005,
                "max_box_area_frac": 0.5,
                "min_short_side_frac": 0.01,
                "min_ar": 1.0,
                "max_ar": 5.0,
                "mask_min_per_box": 0.4,
                "mask_min_circ": 0.2,
                "mask_min_solidity": 0.3,
                "mask_max_holes_frac": 0.2,
                "mask_min_ar": 1.0,
                "mask_max_ar": 4.0,
                "white_min_frac": 0.05,
                "black_max_frac": 0.6,
                "yellow_min_frac": 0.05,
                "red_min_frac": 0.05,
                "green_min_frac": 0.05,
                "blue_min_frac": 0.05,
                "require_mask": False,
                "logistic_gate": {
                    "features": ["score", "box_area_frac", "aspect_ratio"],
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                    "weights": [1.0, 0.5, -0.1],
                    "bias": 0.0,
                    "threshold_type": "logit",
                    "threshold_value": 0.0,
                },
            }
        },
    }


def example_threshold_yaml() -> str:
    """Example YAML covering all keys for threshold/logistic configs."""
    return yaml.safe_dump(_example_dict(), sort_keys=False, allow_unicode=True)
