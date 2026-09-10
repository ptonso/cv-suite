# Classify Output

## 1. General Description
The classify branch currently has one output writer: `to-class-dir`. It materializes a shared dataset as class-organized image folders, optionally creates train/val/test split subfolders, optionally emits stats, and contains the branch’s thresholding and multi-class routing rules. Canonical file writing is delegated to `cvsuite.common.io` class-dir logic, while cvsuite keeps branch-specific split assignment, threshold policy, and multi-class ingest behavior.

## 2. Inputs and Outputs (I/O)
Command:
- `cvsuite class <src> ... to-class-dir <dst> [options]`

Arguments:
- `dst`
- `--threshold`
- `--class-thresholds <yaml path>`
- repeatable `--class-threshold label=value`
- `--multi-class`
- repeatable `--exclude-class NAME`
- `--hardlink`
- `--unlabeled-name` default `unlabeled`
- `--val-frac` default `0.0`
- `--test-frac` default `0.0`
- `--seed` default `0`
- `--randomized`
- `--preserve-splits`
- `--with-stats`

Outputs:
- class-directory tree under `dst`
- optional split-rooted tree under `dst/train`, `dst/val`, `dst/test`
- optional `stats.yaml`

## 3. Interfaces
Writer interface:
- consumes `record.classification`, `classification.probs`, dataset meta, and record splits
- may call classify split-assignment helpers before writing files
- may hardlink or copy image bytes
- delegates canonical class-dir materialization to `cvsuite.common.io.write_dataset(..., format="class-dir", ...)`
- may emit stats through `[[common/stats]]`

Path conventions:
- flat output when both split fractions are zero and `--preserve-splits` is false
- split output when:
  - `--preserve-splits` is true, or
  - `val_frac > 0`, or
  - `test_frac > 0`

## 4. Business Decisions and Strict Policies
- Multi-class output auto-enables when the dataset came from multi-class ingest, even if `--multi-class` was not passed.
- When multi-class output is active, a single record may be exported into multiple class folders.
- Low-score predictions route to the `unlabeled` bucket rather than being dropped.
- `--exclude-class NAME` removes the named class from the dataset before writing: its folder never appears, it is stripped from multi-class label sets, and records that belonged only to it are dropped (not routed to `unlabeled`). Exclusion is applied before split assignment, so split fractions are computed over the surviving records.
- Nested class labels such as `lights/canopy_light` are preserved as nested directories; they are not flattened.
- Path components are sanitized to prevent unsafe folder names such as `.` or `..`, and folder names also sanitize spaces and slash-separated components as needed.
- Filename collisions are resolved by suffixing `_1`, `_2`, and so on.
- Split fractions must be non-negative and their sum must be at most 1.
- Randomized split assignment uses global counts; default split assignment is stratified per effective class label.
- `--preserve-splits` skips reassignment and respects existing `record.split` values.

Threshold semantics:
- if probabilities exist, the output resolves labels from `classification.probs`
- if no probabilities exist, the output falls back to `classification.label` and `classification.score`
- per-class threshold overrides can come from YAML and repeatable CLI flags
- a classification meta threshold may also serve as the effective default threshold when explicit output thresholds are absent
- threshold precedence is:
  1. repeatable `--class-threshold label=value`
  2. `--class-thresholds <yaml>`
  3. global `--threshold`

## 5. Implementation Details
Split assignment details:
- stratified mode groups by effective class label
- val/test counts are floored
- split-assignment provenance is written into `dataset.meta["output_split_assignment"]`

File writing details:
- hardlink mode uses `os.link`
- copy mode copies raw bytes
- destination folder structure is either:
  - `<dst>/<label>/<filename>` or
  - `<dst>/<split>/<label>/<filename>`
- multi-class ingest metadata is mirrored into the canonical class-dir multi-label attribute so delegated writing preserves all source labels
- `write_classified(...)` still exists as a compatibility wrapper and simply delegates to `write_class_dir(...)`

Stats details:
- `build_classify_stats()` reports:
  - `assignments_by_output_label`
  - `unlabeled_records`
  - confidence summary
  - `multi_class_enabled`
  - `multi_class_records` when applicable
  - threshold metadata used by the run

Related docs:
- `[[classify/cli]]`
- `[[classify/transforms]]`
- `[[common/stats]]`
