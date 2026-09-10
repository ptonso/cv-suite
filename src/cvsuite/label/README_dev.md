# cvsuite label — Developer Notes

Deep dive into how `cvsuite label` ingests, transforms, and writes datasets.

## Architecture
- **Router** picks a format (`yolo`, `coco`, `labelme`, `images`, `unstructured`) via `detect_format`, then ingests to a unified `RecordsDataset`.
- **RecordsDataset** holds image path/size, boxes, polygons, keypoints, split, classes, metadata, plus optional FM predictions.
- Pipeline shape is fixed: `ingest → [single transform] → output command`. Transforms and output commands are dynamically discovered under `cvsuite/label/transform/*/commands` and `cvsuite/label/output/commands/`.

## Format detection
- YAML with `format: coco` → COCO; `format: labelme` → LabelMe; otherwise YOLO.
- JSON with `images`+`annotations` → COCO; JSON with `shapes` → LabelMe.
- Directories: presence of `data.yaml` applies the same `format` rule; else we look for split folders with `images/` + `labels*/`.
- Fallback: any image file → `unstructured`; otherwise error.

## YOLO ingest heuristics
- Reads `data.yaml` (or a folder containing it). `path` may be relative; if nonexistent, fall back to the YAML’s parent.
- Task resolution when `task=auto`:
  - Prefer explicit subdirs: `labels_det/` → det, `labels_seg/` → seg, `labels_pose/` → pose.
  - When a plain `labels/` directory exists and `data.yaml` declares `task`, that task owns `labels/`.
  - Sample label files to infer: 5 tokens → det; even tokens ≥7 → seg; triplets pattern → pose.
  - If `kpt_shape`/`keypoints` present in YAML and no labels found, default to pose.
- Segmentation polygons are accepted from either `labels_seg/` **or** plain `labels/` (e.g., Roboflow YOLOv8 exports).

## COCO ingest heuristics
- Accepts direct annotation JSON or a `data.yaml` with `format: coco`.
- In `data.yaml` we resolve `train/val/test` entries relative to both the YAML dir and `path`. `path` is allowed to be `.` (as written by `to-coco`).
- Split name inference for bare JSON files uses path parts and filename tokens (`train`, `val`, `test`, etc.).
- Tasks: `task=auto` uses presence of polygons to choose `seg` else `det`.

## Writes
- `to-coco`: emits COCO JSON plus `data.yaml` with `path: .`, `format: coco`, split keys → annotation files (relative to export root). Images are copied or hardlinked based on layout/flag.
- `to-yolo`: writes split-first layout and preserves every detected task. One task owns plain `labels/`: if only one task is present, it uses `labels/`; when multiple tasks are present, `--task` chooses the plain-label task and `auto` falls back to `det` when available. The remaining tasks go to typed folders such as `labels_det/`, `labels_seg/`, and `labels_pose/`. For Ultralytics compatibility, emitted `data.yaml` always contains `train` and `val`, aliasing one to the other when only one real split was exported.
- `to-labelme`: writes train-only default output directly under `dst/`; split folders are used only for `--preserve-splits` or nonzero `--val-frac`/`--test-frac`.
- `ops crop-dets`: crop transform that fans out one object per record/image before a normal exporter runs. `--margin-frac` expands the source box first. `--imgsz` makes square crops with `--pad-type {background,black,gray,white}`; `--long-size` is the aspect-preserving alternative.
- dataset-writing `to-*` commands assign output splits inline with `--val-frac`, `--test-frac`, and `--seed` before writing; default `0.0/0.0` means train-only output.
- `to-labelme`, `to-results`, `audit`, `inspect`: read-only for inputs, write to fresh destinations.

## Contracts & guardrails
- Never mutate source datasets; outputs go to caller-specified folders/files.
- Avoid overwriting annotation files unless explicitly allowed by the helper (e.g., YOLO writer raises if label exists).
- All paths in emitted YAML are relative to the export root when possible, to keep datasets movable.
- Transforms are pure: they return a new `RecordsDataset`; only output commands touch disk.
- FM-capable transforms such as `ground` do not own model wrappers; they populate `dataset.fm_request`, seed any branch-local prompt state on records, and call `cvsuite.common.fm`.

## Extending
- New transform: add under `cvsuite/label/transform/<project>/commands/`; `run.py` is exposed as `<project>`, any other module becomes `<project>-<command>`.
- New writer: drop a module in `cvsuite/label/output/commands/` and it becomes available as `to-<name>`.
- Keep ingest heuristics conservative—prefer explicit config, but sample labels when needed to stay robust to common YOLO/COCO variants.
