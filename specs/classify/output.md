# Classify Output

Related: **Read first** [classify cli](cli.md). **See also** [classify transforms](transforms.md), [io](../common/io.md), [stats](../common/stats.md).

## Overview

`to-class-dir` is the branch's only output. It writes the dataset as class-organized image folders, optionally split into `train`/`val`/`test`, optionally with `stats.yaml`. The branch owns thresholding, split assignment and multi-class routing; the file writing itself is delegated to the `class-dir` adapter through `cvsuite.common.io.write_dataset(..., format="class-dir")`.

## Language

- **Effective label**: the label(s) a record is written under after thresholds are applied.
- **Unlabeled bucket**: folder (default `unlabeled`) receiving records below threshold.

## Map

- `classify/output/commands/to_class_dir.py`.
- Adapter: [io](../common/io.md). Stats: [stats](../common/stats.md).

## Contracts

`cvsuite class <src> ... to-class-dir <dst> [options]`

Arguments: `dst`; `--threshold`; `--class-thresholds <yaml>`; repeatable `--class-threshold label=value`; `--multi-class`; repeatable `--exclude-class NAME`; `--hardlink`; `--overwrite`; `--unlabeled-name` (default `unlabeled`); `--val-frac` (0.0); `--test-frac` (0.0); `--seed` (0); `--randomized`; `--preserve-splits`; `--with-stats`.

Layout:

- Flat `<dst>/<label>/<filename>` when both fractions are zero and `--preserve-splits` is off.
- `<dst>/<split>/<label>/<filename>` when `--preserve-splits` is set or either fraction is above zero.

A class folder (`<label>` or `<split>/<label>`) that already holds files raises unless `--overwrite`, which replaces that folder ([output claims](../common/io.md#output-claims)). Class folders that this run does not write stay on disk.

Inputs consumed: `record.classification`, `classification.probs`, dataset meta, record splits. Copy mode copies bytes; `--hardlink` uses `os.link`.

Threshold precedence: repeatable `--class-threshold label=value`, then `--class-thresholds <yaml>`, then global `--threshold`. A classification meta threshold serves as the default when no explicit threshold is given. With probabilities, labels resolve from `classification.probs`; otherwise from `classification.label` and `classification.score`.

Split assignment is written to `dataset.meta["output_split_assignment"]`. Stratified mode (default) groups by effective class label and floors val/test counts. `--randomized` uses global counts.

Stats (`build_classify_stats`): `assignments_by_output_label`, `unlabeled_records`, confidence summary, `multi_class_enabled`, `multi_class_records`, thresholds used.

## Constraints

- Multi-class output MUST auto-enable when the dataset came from multi-class ingest. When active, a record MAY be written into several class folders.
- Below-threshold predictions MUST go to the unlabeled bucket and MUST NOT be dropped.
- `--exclude-class` MUST remove the class before split assignment: its folder never appears, it is stripped from multi-class label sets, and records that belonged only to it are dropped and not sent to unlabeled.
- Nested class labels such as `lights/canopy_light` MUST stay nested.
- Path components MUST be sanitized against unsafe names such as `.` and `..`; spaces and slash-separated components are sanitized.
- Filename collisions between records of one run MUST be resolved with `_1`, `_2`, and so on.
- A dirty class folder MUST raise `FileExistsError` unless `--overwrite` is given.
- Split fractions MUST be non-negative with a sum of at most 1.
- `--preserve-splits` MUST skip reassignment and keep `record.split`.
- Multi-class ingest metadata MUST be mirrored into the class-dir multi-label attribute so delegated writing keeps every label.

## History

- [Idempotent outputs](../logs/2026-11-05_idempotent-outputs.md): why a rerun raises unless `--overwrite`.
