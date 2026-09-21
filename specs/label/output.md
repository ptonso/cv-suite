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

- `to-yolo`: `dst`, `--task {auto,det,seg,pose}`, split-assignment args, `--hardlink`, `--with-stats`.
- `to-labelme`: `dst`, split-assignment args, `--embed-image`, `--hardlink`, `--with-stats`. Writes flat `dst/` sidecars for default train-only output, and split directories only when `--preserve-splits`, `--val-frac` or `--test-frac` asks for them.
- `to-coco`: `dst` (a directory; a file path is rejected), split-assignment args, `--coco-style`, `--no-images` (annotations and `data.yaml` only, with absolute `file_name` entries), `--hardlink`, `--with-stats`. Default layout `<dst>/<split>/_annotations.coco.json` with images under the split dir and helper `<dst>/data.yaml` (`format: coco`); `--coco-style` gives `images/` plus `annotations/annotations_<split>.json`.
- `to-results`: `dst`, `--names-mode`, `--imgsz`, `--max`, split-assignment args, `--skip-annotated`, `--skip-crops`, `--skip-masks`, `--skip-keypoints`, `--with-stats`.
- `audit`: optional `dst`, `--split`, `--thresholds`.
- `inspect`: `--split`, `--task`, `--imgsz`, `--max`, `--show-confidence`, `--show-prompt`.

Shared split-assignment arguments (`label/output/common.py`): `--val-frac`, `--test-frac`, `--seed`, `--preserve-splits`. The three format outputs build the split assignment, then call `write_dataset` with the matching format id. `to-results`, `audit` and `inspect` do not go through the io writer.

## Constraints

- `to-yolo` MUST emit helper `data.yaml` and keep the task spelling `seg` there, although canonical ids use `inst-seg`.
- Every output command that writes files MUST materialize a directory.
- `inspect` MAY mutate the working dataset when filtering, and is interactive.
