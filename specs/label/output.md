# Label Output

Related: **Read first** [label cli](cli.md). **See also** [label adapters](adapters.md), [io](../common/io.md), [stats](../common/stats.md).

## Overview

Label outputs write the dataset as annotation formats (`to-yolo`, `to-labelme`, `to-coco`), or as review artifacts (`to-results`, `audit`, `inspect`). The three format outputs delegate to `cvsuite.common.io.write_dataset(...)`. The branch keeps CLI spellings, split assignment, helper `data.yaml` emission and the review outputs.

## Language

- **Split assignment**: choosing `train`/`val`/`test` per record from `--val-frac`, `--test-frac`, `--seed`, `--preserve-splits`.

## Map

- `label/output/commands/`: `to_yolo.py`, `to_labelme.py`, `to_coco.py`, `to_results.py`, `audit.py`, `inspect.py`.
- `label/output/common.py`: shared split assignment.

## Contracts

- `to-yolo`: `dst`, `--task {auto,det,seg,pose}`, split-assignment args, `--hardlink`, `--overwrite`, `--with-stats`.
- `to-labelme`: `dst`, split-assignment args, `--embed-image`, `--hardlink`, `--overwrite`, `--with-stats`. Writes flat `dst/` sidecars for default train-only output, and split directories only when `--preserve-splits`, `--val-frac` or `--test-frac` asks for them.
- `to-coco`: `dst` (a directory; a file path is rejected), split-assignment args, `--coco-style`, `--no-images` (annotations and `data.yaml` only, with absolute `file_name` entries), `--hardlink`, `--overwrite`, `--with-stats`. Default layout `<dst>/<split>/_annotations.coco.json` with images under the split dir and helper `<dst>/data.yaml` (`format: coco`); `--coco-style` gives `images/` plus `annotations/annotations_<split>.json`.
- `to-results`: `dst`, `--names-mode`, `--imgsz`, `--max`, split-assignment args, `--skip-annotated`, `--skip-crops`, `--skip-masks`, `--skip-keypoints`, `--with-stats`.
- `audit`: optional `dst`, `--split`, `--thresholds`.
- `inspect`: `--split`, `--task`, `--imgsz`, `--max`, `--show-confidence`, `--show-prompt`.

Shared split-assignment arguments (`label/output/common.py`): `--val-frac`, `--test-frac`, `--seed`, `--preserve-splits`. The three format outputs build the split assignment, then call `write_dataset` with the matching format id and `--overwrite` as its `overwrite` argument. A destination that an earlier run filled raises unless `--overwrite` is given, and `--overwrite` replaces the trees this run writes ([output claims](../common/io.md#output-claims)). `--coco-style` writes every split in one `write_dataset` call.

The helper `data.yaml` of `to-yolo` and `to-coco` lists the split trees present under `dst` after the write, whichever invocation wrote them. `to-results`, `audit` and `inspect` do not go through the io writer.

## Constraints

- `to-yolo` MUST emit helper `data.yaml` and keep the task spelling `seg` there, although canonical ids use `inst-seg`.
- Every output command that writes files MUST materialize a directory.
- `to-yolo`, `to-labelme` and `to-coco` MUST expose `--overwrite` and MUST NOT write into a dirty claimed path without it.
- The helper `data.yaml` MUST NOT name a split tree that is absent from disk, and MUST name every split tree that is present.
- `inspect` MAY mutate the working dataset when filtering, and is interactive.

## History

- [Idempotent outputs](../logs/2026-11-05_idempotent-outputs.md): why a rerun raises unless `--overwrite`, and why `data.yaml` follows the disk.
