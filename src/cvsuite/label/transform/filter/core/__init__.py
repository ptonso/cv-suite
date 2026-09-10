"""Core utilities for filter transforms (feature extraction, gates, optimizer)."""

from .config import (
    FilterConfig,
    GlobalConfig,
    ClassThresholds,
    LogisticGate,
    load_filter_config,
    example_threshold_yaml,
    filter_config_from_dict,
)
from .features import FeatureExtractor
from .filters import ThresholdFilter, LogisticFilter
from .types import Features
from .records import (
    extract_detections,
    apply_sources_to_record,
    gt_boxes_labels_from_record,
    warn_on_stem_mismatch,
    load_rgb_for_rec,
    build_batch,
)

__all__ = [
    "FilterConfig",
    "GlobalConfig",
    "ClassThresholds",
    "LogisticGate",
    "load_filter_config",
    "filter_config_from_dict",
    "example_threshold_yaml",
    "FeatureExtractor",
    "ThresholdFilter",
    "LogisticFilter",
    "Features",
    "extract_detections",
    "apply_sources_to_record",
    "gt_boxes_labels_from_record",
    "warn_on_stem_mismatch",
    "load_rgb_for_rec",
    "build_batch",
]
