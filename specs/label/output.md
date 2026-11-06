# Label Output

Related: **Read first** [label cli](cli.md). **See also** [label adapters](adapters.md), [io](../common/io.md), [stats](../common/stats.md).

## Overview

Label outputs write the dataset as annotation formats (`to-yolo`, `to-labelme`, `to-coco`), or as review artifacts (`to-results`, `audit`, `inspect`). The three format outputs delegate to `cvsuite.common.io.write_dataset(...)`. The branch keeps CLI spellings, helper `data.yaml` emission and the review outputs. Outputs never assign splits: they write `record.split` as it stands, or one dataset per fold when `split --folds` ran ([split](../common/split.md#writing-folds)).

## Language

- **Fold folder**: `<dst>/fold<i>`, the destination of one fold.

## Map

- `label/output/commands/`: `to_yolo.py`, `to_labelme.py`, `to_coco.py`, `to_results.py`, `audit.py`, `inspect.py`.

## Contracts

- `to-yolo`: `dst`, `--task {auto,det,seg,pose}`, `--boxes-from-polygons` (the box is the polygon extent clipped to the image), `--link-mode`, `--overwrite`, `--with-stats`. Writes a split tree for every split present, `train` included.
- `to-labelme`: `dst`, `--embed-image`, `--link-mode`, `--overwrite`, `--with-stats`. Writes flat `dst/` sidecars when every record is in `train`, and split directories otherwise.
- `to-coco`: `dst` (a directory; a file path is rejected), `--coco-style`, `--no-images` (annotations and `data.yaml` only, with absolute `file_name` entries), `--link-mode`, `--overwrite`, `--with-stats`. Default layout `<dst>/<split>/_annotations.coco.json` with images under the split dir and helper `<dst>/data.yaml` (`format: coco`); `--coco-style` gives `images/` plus `annotations/annotations_<split>.json`.
- `to-results`: `dst`, `--names-mode`, `--imgsz`, `--max`, `--skip-annotated`, `--skip-crops`, `--skip-masks`, `--skip-keypoints`, `--with-stats`. It is a review output and takes a dataset with at most one fold.
- `audit`: optional `dst`, `--split`, `--thresholds`. It takes a dataset with at most one fold.
- `inspect`: `--split`, `--task`, `--imgsz`, `--max`, `--show-confidence`, `--show-prompt`. It takes a dataset with at most one fold.

The three format outputs call `fold_targets(dataset, dst)` and write each yielded dataset to its destination with `write_dataset` and the matching format id, passing `--overwrite` as its `overwrite` argument. Without folds, that is one dataset and `dst`. With folds, each `<dst>/fold<i>` gets its own `data.yaml` and, with `--with-stats`, its own `stats.yaml`. A destination that an earlier run filled raises unless `--overwrite` is given, and `--overwrite` replaces the trees this run writes ([output claims](../common/io.md#output-claims)). `--coco-style` writes every split of a fold in one `write_dataset` call.

The helper `data.yaml` of `to-yolo` and `to-coco` lists the split trees present under its destination after the write, whichever invocation wrote them. `to-results`, `audit` and `inspect` do not go through the io writer.

## Constraints

- `to-yolo` MUST emit helper `data.yaml` and keep the task spelling `seg` there, although canonical ids use `inst-seg`.
- Every output command that writes files MUST materialize a directory.
- `to-yolo`, `to-labelme` and `to-coco` MUST expose `--overwrite` and MUST NOT write into a dirty claimed path without it.
- The helper `data.yaml` MUST NOT name a split tree that is absent from disk, and MUST name every split tree that is present.
- Outputs MUST NOT expose split-assignment flags, and MUST write the splits the records carry.
- `to-yolo --task det` on a dataset that holds only polygons MUST raise unless `--boxes-from-polygons` is given.
- A review output given more than one fold MUST raise and name `split --only-fold`.
- `inspect` MAY mutate the working dataset when filtering, and is interactive.

## History

- [Idempotent outputs](../logs/2026-11-05_idempotent-outputs.md): why a rerun raises unless `--overwrite`, and why `data.yaml` follows the disk.
- [Chained transforms and split as a transform](../logs/2026-11-05_chained-transforms-and-split.md): why splitting left the outputs and why folds are written side by side.
