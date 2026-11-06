# Split

Related: **Read first** [records_dataset](records_dataset.md), [keys](keys.md). **See also** [pipeline](../pipeline.md#chaining), [similarity](similarity.md), [label output](../label/output.md), [classify output](../classify/output.md), [eda report](../eda/report.md).

## Overview

`split` is a transform that assigns every record to `train`, `val` or `test`, or to K cross-validation folds. It is shared by the `label` and `class` branches, and `eda` runs it as part of a chain. Outputs never assign splits: they write `record.split` as it stands, or the folds when the dataset carries them.

Two ideas shape the assignment. A **group** is a set of records that must land on the same side, such as all photos of one physical board ([keys](keys.md)). A **stratum column** is a count the splits should share in proportion, such as the number of instances of each class. A group is an indivisible lump with fixed counts, so the shares can only be approached. The command packs whole groups into splits, reports the shares it reached against the targets, and fails when they are too far apart.

Example: `cvsuite label ./boards split --group-by 'regex:^(\w+?)(?:_top|_bottom)?\.' --stratify-by class --val-frac 0.2 --test-frac 0.2 to-yolo ./out`.

With `--folds 5` the same command writes `./out/fold0` to `./out/fold4`. Each fold tests on one fifth of the groups, validates on a share of the remaining groups, and trains on the rest. `--only-fold 2` writes only `fold2`, and it is the same `fold2` the full run holds.

## Language

- **Group**: records that stay together ([keys](keys.md#grouping)). Without `--group-by`, each record is a group.
- **Column**: one count per record that the splits balance. With `--stratify-by class`, the columns are the classes plus `(empty)` for records without annotations.
- **Share**: the part of a column's total that a split holds.
- **Target share**: the share a split should hold, from the fractions.
- **Pool**: the records that are not pinned.
- **Pinned**: kept in their current split by `--keep`.
- **Sibling**: an unpinned record that shares a group with a pinned record.
- **Fold**: one of K disjoint test sets, with its own train and val assignment.

## Map

- `common/splitting.py`: the packing, free of I/O.
- `common/transforms/split.py`: the flags, validation and weights, shared by both branches.
- `label/transform/split/commands/run.py` and `classify/transform/commands/split.py`: thin command modules over it.
- `common/io/folds.py`: `fold_targets`, which outputs use to write folds, and `require_single_fold` for review outputs.

## Contracts

### Command

```
split [--group-by KEY]... [--stratify-by {class,class-size:PX[,PX...],none,KEY}]
      [--val-frac F] [--test-frac F]
      [--folds K [--only-fold I[,I...]]]
      [--keep SPLIT[,SPLIT...] [--on-siblings {join,drop}]]
      [--tolerance T] [--restarts N] [--seed S]
```

Defaults: `--stratify-by class`, fractions `0`, `--tolerance 0.05`, `--restarts 200`, `--seed 0`.

- Ratio mode: a positive `--val-frac` and/or `--test-frac`, with train taking the remainder.
- Fold mode: `--folds K` with `K >= 2`. `--val-frac` is the share of each fold's non-test groups used for validation, and `0` gives folds with train and test only. `--test-frac` is rejected. `--only-fold` takes indices in `[0, K)`.
- `--keep` pins the records whose current split is listed. The fractions then apply to the pool, and a kept split takes no fraction: `--keep test` with `--test-frac` raises. A group that holds a pinned record is pinned whole when `--on-siblings join` (the default), and with `--on-siblings drop` the records of that group that are not pinned themselves are removed from the dataset. The report states how many were removed. `--on-siblings` without `--keep` raises. A group that holds records pinned to different splits raises, naming the group.
- `--keep` and `--folds` are exclusive.

### Assignment

Groups are visited largest first, each with its size multiplied by a seeded random factor in `[0.7, 1.3]`. A group goes to the split where it reduces the squared relative deviation from the target shares the most. The best of `--restarts` runs by total squared relative deviation is kept. Groups are sorted by group id before packing, so the result is the same for any record order.

Columns for `--stratify-by class`: one count per annotation of a record, where a polygon and the box derived from it count once, and one count of the record's label for classification data. A multi-label record counts once per label. `--stratify-by class-size:20,64` splits every class column by the size of the annotation, measured as `sqrt(w * h)` in source pixels of its box or the extent of its polygon: one column per class and bucket (`<20`, `20-64`, `>=64`), named `class|bucket`, with `(empty)` for records without annotations. It needs boxes or polygons and raises on classification data. `--stratify-by KEY` makes one column per value of a [key](keys.md) and counts images. `none` has a single column that counts images.

Fold mode packs the groups into K equal test sets. For each fold i, the groups outside its test set are packed into train and val with shares `1 - val` and `val`, using seed `S + i + 1`.

### Representation

Ratio mode sets `record.split`. Fold mode sets `record.attributes["fold_splits"]`, a list of K split names that holds the record's split in every fold, and `dataset.meta["folds"] = {"k": K, "present": [i, ...]}`. `present` is every fold, or the ones named by `--only-fold`. `record.split` holds the split of the lowest present fold, so a transform or output that knows nothing about folds still sees a valid dataset.

### Report

The command prints a table of target and achieved shares for each split, for the 20 largest columns plus an aggregate row, and the group count per split. The same data goes into the step's `info` in `meta["steps"]` ([pipeline](../pipeline.md#chaining)).

A column is checked when its total is at least ten times the number of splits. Every checked column in every split MUST be within `--tolerance` of its target share, measured in share points, or the command exits with the table and the largest deviation. Unchecked columns are listed.

### Writing folds

`fold_targets(dataset, dst)` yields `(dataset_i, dst / f"fold{i}")` for each present fold, with `record.split` set from `fold_splits[i]`. For a dataset without folds it yields `(dataset, dst)`. Outputs that write datasets call it ([label output](../label/output.md), [classify output](../classify/output.md)).

## Constraints

- A run without a positive fraction and without `--folds` MUST raise.
- The same dataset, options and seed MUST give the same assignment for any record order.
- No group MAY span two splits in ratio mode, and no group MAY span the test set and another split in any fold.
- With `--only-fold I`, the assignment of fold I MUST equal fold I of the run without `--only-fold`.
- Every fold's test sets MUST partition the records: each record is in the test split of exactly one fold.
- A record's split in each fold MUST be one of `train`, `val`, `test`.
- A deviation beyond `--tolerance` MUST fail the command and MUST NOT write a split silently.
- `similar:` MUST be rejected by `--stratify-by`.
- With `--on-siblings drop`, no removed record MAY be pinned, and every group that holds a pinned record MUST end with only pinned records.
- `class-size` buckets MUST partition the annotations: thresholds are positive and strictly increasing, and each annotation falls in exactly one bucket.

## History

- [Chained transforms and split as a transform](../logs/2026-11-05_chained-transforms-and-split.md): why splitting left the outputs and why folds are written together.
- [What belongs in cvsuite](../logs/2026-11-05_what-belongs-in-cvsuite.md): why groups enter as data.
- [The eda branch](../logs/2026-11-05_eda-branch.md): why the split section of a report checks groups and twins against the same keys.
- [Gaps found by migrating the projects](../logs/2026-11-06_migration-gaps.md): why `--on-siblings` and `class-size` exist.
