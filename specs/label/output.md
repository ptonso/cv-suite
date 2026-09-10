# Label Output

## 1. General Description
The label output layer materializes normalized datasets into annotation formats, audits, previews, and results folders. Canonical dataset writing is delegated to `cvsuite.common.io.write_dataset(...)`.

cvsuite still owns:
- CLI spellings
- split-assignment helpers
- helper `data.yaml` emission
- results/audit/inspect outputs

## 2. Inputs and Outputs (I/O)
Output commands and major args:
- `to-yolo`
  - `dst`
  - `--task {auto,det,seg,pose}`
  - split-assignment args
  - `--hardlink`
  - `--with-stats`
- `to-labelme`
  - `dst`
  - split-assignment args
  - `--embed-image`
  - `--hardlink`
  - `--with-stats`
- `to-coco`
  - `dst` (directory; a file path is rejected)
  - split-assignment args
  - `--coco-style`
  - `--no-images` (annotations + `data.yaml` only; `file_name` entries are absolute)
  - `--hardlink`
  - `--with-stats`
- `to-results`
  - `dst`
  - `--names-mode`
  - `--imgsz`
  - `--max`
  - split-assignment args
  - `--skip-annotated`
  - `--skip-crops`
  - `--skip-masks`
  - `--skip-keypoints`
  - `--with-stats`
- `audit`
  - optional `dst`
  - `--split`
  - `--thresholds`
- `inspect`
  - `--split`
  - `--task`
  - `--imgsz`
  - `--max`
  - `--show-confidence`
  - `--show-prompt`

## 3. Interfaces
Shared split-assignment interface:
- `--val-frac`
- `--test-frac`
- `--seed`
- `--preserve-splits`
- implemented by `label/output/common.py`

Delegated writer interfaces:
- `to-yolo`, `to-labelme`, and `to-coco` call `cvsuite.common.io.write_dataset(...)`
- `to-labelme` writes flat `dst/` sidecars for default train-only output and split directories only when `--preserve-splits`, `--val-frac`, or `--test-frac` requests split materialization.

cvsuite-owned outputs:
- `to-results`
- `audit`
- `inspect`

## 4. Business Decisions and Strict Policies
- `to-yolo` still emits helper `data.yaml` and preserves legacy task spelling `seg` there even though canonical internal task ids use `inst-seg`.
- Every output command materializes a **directory**. `to-coco` writes `<dst>/<split>/_annotations.coco.json`, images under the split dir (unless `--no-images`), and a helper `<dst>/data.yaml` with `format: coco`; `--coco-style` switches to the `images/` + `annotations/annotations_<split>.json` layout.
- `inspect` remains interactive and intentionally mutates the working dataset when filtering.

## 5. Implementation Details
- `to-yolo` / `to-labelme` / `to-coco` build split-assignment from `label/output/common.py`, then hand the dataset to `cvsuite.common.io.write_dataset(...)` with the matching format id.
- `to-results`, `audit`, and `inspect` are cvsuite-owned and do not go through the io writer.

Related docs:
- `[[label/cli]]`
- `[[label/adapters]]`
- `[[common/io]]`
- `[[common/stats]]`
