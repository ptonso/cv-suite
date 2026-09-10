from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Features:
    """Per-detection feature columns used by gates/thresholds."""
    box_area_frac:   Array
    short_side_frac: Array
    aspect_ratio:    Array
    score:           Array
    labels:          Array
    boxes:           Array
    masks:       Optional[List[Array]] = None
    mask_per_box:      Optional[Array] = None
    mask_circularity:  Optional[Array] = None
    mask_solidity:     Optional[Array] = None
    mask_holes_frac:   Optional[Array] = None
    mask_ar:           Optional[Array] = None
    white_frac:        Optional[Array] = None
    black_frac:        Optional[Array] = None
    yellow_frac:       Optional[Array] = None
    red_frac:          Optional[Array] = None
    green_frac:        Optional[Array] = None
    blue_frac:         Optional[Array] = None


@dataclass
class LogisticGate:
    features: List[str] = field(default_factory=list)
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    threshold_type: str = "logit"
    threshold_value: float = 0.0


@dataclass
class ClassThresholds:
    det_conf: Optional[float] = None
    seg_conf: Optional[float] = None
    post_conf: Optional[float] = None
    min_box_area_frac: Optional[float] = None
    max_box_area_frac: Optional[float] = None
    min_short_side_frac: Optional[float] = None
    min_ar: Optional[float] = None
    max_ar: Optional[float] = None
    mask_min_per_box: Optional[float] = None
    mask_min_circ: Optional[float] = None
    mask_min_solidity: Optional[float] = None
    mask_max_holes_frac: Optional[float] = None
    mask_min_ar: Optional[float] = None
    mask_max_ar: Optional[float] = None
    white_min_frac: Optional[float] = None
    black_max_frac: Optional[float] = None
    yellow_min_frac: Optional[float] = None
    red_min_frac: Optional[float] = None
    green_min_frac: Optional[float] = None
    blue_min_frac: Optional[float] = None
    require_mask: bool = False
    logistic_gate: Optional[LogisticGate] = None


@dataclass
class GlobalConfig:
    nms_iou: float = 0.6
    topk_per_class: Optional[int] = None
    cross_class_iou: Optional[float] = None
    require_mask: bool = False
    det_conf: Optional[float] = None
    seg_conf: Optional[float] = None
    use_rgb: bool = False


@dataclass
class FilterConfig:
    global_cfg: GlobalConfig = field(default_factory=GlobalConfig)
    per_class: Dict[str, ClassThresholds] = field(default_factory=dict)

    def class_cfg(self, cls: str) -> ClassThresholds:
        return self.per_class.get(cls, ClassThresholds())
