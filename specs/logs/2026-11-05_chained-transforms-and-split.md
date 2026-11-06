# Chained transforms and split as a transform

Affects: [architecture](../architecture.md), [pipeline](../pipeline.md), [split](../common/split.md), [records_dataset](../common/records_dataset.md), [io](../common/io.md), [label output](../label/output.md), [classify output](../classify/output.md), [label transforms](../label/transforms.md), [classify transforms](../classify/transforms.md), [vlm output](../vlm/output.md), [gen output](../gen/output.md), [stats](../common/stats.md), [label cli](../label/cli.md), [classify cli](../classify/cli.md), [eda cli](../eda/cli.md).

Recorded from the maintainer's account and a survey of the existing code, 2026-11-05.

## Situation

A run applied at most one transform. [The branch note](2026-09-20_branch-per-domain.md) records no reason for the limit, and the guide states it as a fact. Stacking steps meant chaining runs through intermediate directories.

Split assignment lived in the output writers, in three copies. `label/output/common.py` draws a random number per record, so ratios are only approximate. `classify/core/io.py` and `common/io/class_dir.py` stratify by class and floor the counts. None of them knows groups, folds or pinned splits. Without `--preserve-splits`, an output reassigns every record, so converting a YOLO dataset with train and val folders flattened it to train.

pcb-mc-ml builds its crop, graph and detection datasets from five deterministic steps in a row: exclude boards, filter by box size, drop cross-split duplicates, resplit by whole physical board, balance classes. synth-label builds cross-validation folds with blocks of frame ids and keeps the official test split while carving validation out of train. Outputs also had three problems: temporary roots of `crop-dets`, `make-negatives` and `inverse-crop-dets` were never cleaned up in the label and class branches, `--hardlink` fell back to copying silently on any `OSError`, and nothing could link by symlink.

## Alternatives considered

- **Keep splitting as output flags and add group, stratum and fold options to every output.** Five outputs would carry the same options, and a split could not be inspected or followed by another transform.
- **A single `--fold i/K` flag, one dataset per run.** A reader who asks for K folds expects K folds. The looping version is still available through `--only-fold`.
- **Folds as a separate output command.** Folds would then exist only at write time, invisible to `eda` and to later transforms.
- **A recipe YAML as the only way to chain.** It records a run well, but it adds a second way to say what a command line already says. The resolved chain is recorded in `meta["steps"]` and in `stats.yaml` instead.
- **Keep chaining through intermediate directories.** Hardlinks make this cheap on disk, but each step needs a destination, a re-read and its own cleanup.

## Decision

Transforms chain left to right, with the contract in [pipeline](../pipeline.md#chaining). `split` becomes a transform shared by `label` and `class`, with the contract in [split](../common/split.md). Outputs write `record.split` as it is, and lose the five split flags. With `--folds K` the dataset carries all K assignments and the outputs write `fold0` to `fold<K-1>` side by side.

Links are controlled by one `--link-mode {copy,hard,sym}` flag on every output that places images. A failed link raises. The `sample --hardlink` hint that copied a transform flag onto the output command disappears with it.

`prep` keeps its own link flags, because it works on files and has its own contract.

## What tipped it

The pcb-mc-ml recipe. Five sequential deterministic steps, each of which would otherwise write a complete dataset, make chaining a requirement of the work. The three diverging copies of split logic made the case for a single owner.

## Trade-offs accepted

- Outputs preserve splits by default. A dataset that came with train and val folders keeps them, where it used to be flattened.
- `--hardlink` is gone, so scripts that use it, including petro-cv's, must move to `--link-mode hard`.
- `meta["sample"]` is replaced by an ordered `meta["steps"]`. Nothing in cvsuite reads the old key, and only tests did.
- At most one model-backed transform per run, and it must come first, because the model runtime serializes the dataset, runs a subprocess and reloads a new dataset.
- `class` split stratifies by the record's label, so threshold routing to `unlabeled` at output can place records differently from the old stratification by effective label.
- Finding where a stage ends probes the transform's own parser at each later command name, so a stage is parsed more than once. Two stages can share a flag name, and a command name used as an option value stays an argument.
