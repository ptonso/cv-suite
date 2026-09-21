# Label Filter

Related: **Read first** [label transforms](transforms.md). **See also** [label output](output.md).

## Overview

The filter family post-processes permissive detections or segmentations, for example the output of a model run with a low threshold. It offers simple confidence filters, geometric and color constraints, and per-class logistic gates trained against a ground-truth dataset. It has two commands: `apply` runs a filter, `train-logistic` learns a filter config.

## Language

- **Threshold filter** (`ThresholdFilter`): confidence, geometric and color rules.
- **Logistic gate** (`LogisticGate`): per-class logistic model over the feature matrix that decides whether to keep a detection.
- **Feature matrix**: per-detection features listed below.

## Map

- `label/transform/filter/commands/`: `apply.py`, `threshold.py`, `logistic_opt.py`, `run.py`.
- `label/transform/filter/core/types.py`: configuration dataclasses.

## Contracts

### apply

Either `--config <yaml>` or at least one inline threshold: `--det-conf`, `--seg-conf`, `--nms-iou`, `--cross-class-iou`, repeatable `--class-det-conf label=value`, repeatable `--class-seg-conf label=value`; plus `--with-rgb`. Result is a filtered `VisionDataset`, with a `filter_results.yaml` summary written beside the target output area.

### train-logistic

`--gt-src`, `--config` (optional seed config), `--out-config`, `--match-iou`, `--epochs`, `--lr`, `--l2`, `--alpha`, `--max-prune-iters`, `--min-features`, `--always-keep`, `--seed`, `--device`, `--beta`, `--conf-metric {ap,fbeta}`, `--val-split`, `--tune-nms-iou`, `--nms-grid`, `--tune-cross-class-iou`, `--cross-grid`, `--with-rgb`, `--log-dir`. Writes a learned YAML config to `--out-config`, `filter_results.yaml`, and an optional tee log.

### Configuration and runtime types

`GlobalConfig`, `ClassThresholds`, `LogisticGate`, `FilterConfig` (`core/types.py`). Runtime: `ThresholdFilter`, `LogisticFilter`. Training: `LogisticOptimizer`.

### Feature matrix

`aspect_ratio`, `black_frac`, `blue_frac`, `box_area_frac`, `green_frac`, `mask_ar`, `mask_circularity`, `mask_holes_frac`, `mask_per_box`, `mask_solidity`, `red_frac`, `score`, `short_side_frac`, `white_frac`, `yellow_frac`.

### Behavior

- Threshold filter: per-class NMS uses `global.nms_iou`; optional `topk_per_class` trims each class after NMS; optional `cross_class_iou` suppresses overlaps across labels.
- Logistic optimizer: optimizes the threshold objective by micro-AP or micro-F-beta; may grid-search global NMS IoU and cross-class IoU; standardizes features before fitting; prunes iteratively by Wald p-value honoring `always_keep` and `min_features`; searches thresholds over logit percentiles and probability grids depending on gate configuration.

## Constraints

- `apply` MUST reject both `--config` plus inline thresholds, and neither.
- A config containing a logistic gate MUST instantiate `LogisticFilter`; otherwise `ThresholdFilter`.
- `require_mask`, global or per class, MUST decide whether polygon-derived features are required.
- Thresholding order MUST be: global task confidence keep; per-class task overrides; geometric and color thresholds; optional post-confidence threshold; NMS and top-k; optional cross-class suppression.
- The logistic filter MUST fall back to axis-aligned thresholds when it cannot build a usable feature matrix.
- Training MUST match detections to ground truth by image stem only, and MUST warn on unmatched stems or root mismatches.
- Validation split: explicit train-like and val/test-like splits MUST be used directly; otherwise a seeded random split.
- Threshold search ties MUST prefer higher thresholds. When detection scores are absent or degenerate, confidence grids MAY collapse to `[None]`.
