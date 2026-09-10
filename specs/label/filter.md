# Label Filter

## 1. General Description
The filter subsystem is a specialized label transform family for post-processing permissive detections or segmentations. It supports simple threshold filters, richer axis-aligned geometric/color constraints, and per-class logistic gates trained against a ground-truth dataset. This subsystem is large enough to deserve its own document rather than being compressed into `[[label/transforms]]`.

## 2. Inputs and Outputs (I/O)
`filter apply` inputs:
- either `--config <yaml>` or at least one inline threshold flag
- inline flags:
  - `--det-conf`
  - `--seg-conf`
  - `--nms-iou`
  - `--cross-class-iou`
  - repeatable `--class-det-conf label=value`
  - repeatable `--class-seg-conf label=value`
  - `--with-rgb`

`filter train-logistic` inputs:
- `--gt-src`
- `--config` optional seed config
- `--out-config`
- `--match-iou`
- `--epochs`
- `--lr`
- `--l2`
- `--alpha`
- `--max-prune-iters`
- `--min-features`
- `--always-keep`
- `--seed`
- `--device`
- `--beta`
- `--conf-metric {ap,fbeta}`
- `--val-split`
- `--tune-nms-iou`
- `--nms-grid`
- `--tune-cross-class-iou`
- `--cross-grid`
- `--with-rgb`
- `--log-dir`

Outputs:
- filtered `VisionDataset`
- learned YAML config for logistic optimization
- `filter_results.yaml` summary written beside the target output area
- optional tee log file for training

## 3. Interfaces
Configuration dataclasses in `label/transform/filter/core/types.py`:
- `GlobalConfig`
- `ClassThresholds`
- `LogisticGate`
- `FilterConfig`

Runtime filter implementations:
- `ThresholdFilter`
- `LogisticFilter`

Training implementation:
- `LogisticOptimizer`

Feature matrix fields:
- `aspect_ratio`
- `black_frac`
- `blue_frac`
- `box_area_frac`
- `green_frac`
- `mask_ar`
- `mask_circularity`
- `mask_holes_frac`
- `mask_per_box`
- `mask_solidity`
- `red_frac`
- `score`
- `short_side_frac`
- `white_frac`
- `yellow_frac`

## 4. Business Decisions and Strict Policies
- `filter apply` rejects mixed use of `--config` plus inline thresholds.
- `filter apply` also rejects the absence of both config and inline thresholds.
- Any config containing a logistic gate causes the runtime to instantiate `LogisticFilter`; otherwise it uses `ThresholdFilter`.
- `require_mask` can be specified globally or per class and changes whether polygon-derived features are required.
- Thresholding order is strict:
  - confidence keep via global task threshold,
  - per-class task overrides,
  - geometric and color thresholds,
  - optional post-conf threshold,
  - postprocess NMS and top-k,
  - optional cross-class suppression.
- Logistic filtering falls back to axis-aligned thresholds if it cannot build a usable feature matrix.
- Logistic training matches permissive detections to ground truth by image stem only and warns on unmatched stems or root mismatches.
- Validation split policy:
  - explicit train-like and val/test-like splits => use them directly,
  - otherwise use a seeded random split.
- Threshold search ties prefer higher thresholds.
- If detection scores are absent or degenerate, confidence-threshold grids may collapse to `[None]`.

## 5. Implementation Details
Threshold filter behavior:
- per-class NMS uses `global.nms_iou`
- optional `topk_per_class` trims each class after NMS
- optional `cross_class_iou` suppresses overlaps across labels

Logistic optimizer behavior:
- can optimize threshold objective by micro-AP or micro-F-beta
- can optionally grid-search global NMS IoU and cross-class IoU
- standardizes features before fitting gates
- performs iterative Wald p-value pruning with `always_keep` and `min_features`
- searches thresholds over both logit percentiles and probability grids depending on gate configuration

Artifacts:
- optimized config YAML is written to `--out-config`
- summary metrics are written to `filter_results.yaml`

Related docs:
- `[[label/transforms]]`
- `[[label/output]]`
