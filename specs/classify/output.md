# Classify Output

Related: **Read first** [classify cli](cli.md). **See also** [classify transforms](transforms.md), [io](../common/io.md), [stats](../common/stats.md).

## Overview

`to-class-dir` is the branch's only output. It writes the dataset as class-organized image folders, split into `train`/`val`/`test` as the records carry them (or into fold folders when `split --folds` ran), optionally with `stats.yaml`. The branch owns thresholding and multi-class routing; it never assigns splits ([split](../common/split.md)). The file writing itself is delegated to the `class-dir` adapter through `cvsuite.common.io.write_dataset(..., format="class-dir")`.

## Language

- **Effective label**: the label(s) a record is written under after thresholds are applied.
- **Unlabeled bucket**: folder (default `unlabeled`) receiving records below threshold.

## Map

- `classify/output/commands/to_class_dir.py`.
- Adapter: [io](../common/io.md). Stats: [stats](../common/stats.md).

## Contracts

`cvsuite class <src> ... to-class-dir <dst> [options]`

Arguments: `dst`; `--threshold`; `--class-thresholds <yaml>`; repeatable `--class-threshold label=value`; `--multi-class`; repeatable `--exclude-class NAME`; `--link-mode {copy,hard,sym}` ([pipeline](../pipeline.md#link-mode)); `--overwrite`; `--unlabeled-name` (default `unlabeled`); `--with-stats`.

Layout:

- Flat `<dst>/<label>/<filename>` when every record is in `train`.
- `<dst>/<split>/<label>/<filename>` otherwise.
- With folds, the same layouts under `<dst>/fold<i>/`, one per present fold ([split](../common/split.md#writing-folds)).

A class folder (`<label>` or `<split>/<label>`) that already holds files raises unless `--overwrite`, which replaces that folder ([output claims](../common/io.md#output-claims)). Class folders that this run does not write stay on disk.

Inputs consumed: `record.classification`, `classification.probs`, dataset meta, record splits. Copy mode copies bytes; `--link-mode hard` uses `os.link` and `sym` writes absolute symbolic links.

Threshold precedence: repeatable `--class-threshold label=value`, then `--class-thresholds <yaml>`, then global `--threshold`. A classification meta threshold serves as the default when no explicit threshold is given. With probabilities, labels resolve from `classification.probs`; otherwise from `classification.label` and `classification.score`.

Stats (`build_classify_stats`): `assignments_by_output_label`, `unlabeled_records`, confidence summary, `multi_class_enabled`, `multi_class_records`, thresholds used.

## Constraints

- Multi-class output MUST auto-enable when the dataset came from multi-class ingest. When active, a record MAY be written into several class folders.
- Below-threshold predictions MUST go to the unlabeled bucket and MUST NOT be dropped.
- `--exclude-class` MUST remove the class at output: its folder never appears, it is stripped from multi-class label sets, and records that belonged only to it are dropped and not sent to unlabeled. To keep a class out of a split's balance, drop it with `select --drop-class` before `split`.
- Nested class labels such as `lights/canopy_light` MUST stay nested.
- Path components MUST be sanitized against unsafe names such as `.` and `..`; spaces and slash-separated components are sanitized.
- Filename collisions between records of one run MUST be resolved with `_1`, `_2`, and so on.
- A dirty class folder MUST raise `FileExistsError` unless `--overwrite` is given.
- `to-class-dir` MUST NOT expose split-assignment flags, and MUST write the splits the records carry.
- With folds, `to-class-dir` MUST write every present fold into its own `fold<i>` folder.
- Multi-class ingest metadata MUST be mirrored into the class-dir multi-label attribute so delegated writing keeps every label.

## History

- [Idempotent outputs](../logs/2026-11-05_idempotent-outputs.md): why a rerun raises unless `--overwrite`.
- [Chained transforms and split as a transform](../logs/2026-11-05_chained-transforms-and-split.md): why splitting left this output and why `--link-mode` replaced `--hardlink`.
