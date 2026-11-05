# Eda CLI

Related: **Read first** [pipeline](../pipeline.md). **See also** [eda report](report.md), [keys](../common/keys.md), [split](../common/split.md), [select and remap](../common/select_remap.md), [label cli](../label/cli.md).

## Overview

`cvsuite eda` explores a dataset before you run experiments on it: what it looks like, how its classes are distributed, whether the split can be trusted, and whether labels can be guessed from file metadata. It writes figures and tables into a report folder. It follows the [pipeline](../pipeline.md) shape, with the report as its output command, so a dataset can be narrowed or split in the same command and the analysis runs on the result. A split can be checked before it is ever written.

Example: `cvsuite eda ./visa select --drop-class Background --empty drop split --group-by dir:0 --val-frac 0.15 --test-frac 0.15 to-report ./report --group-by dir:0`.

The analysis never changes its source. It reads the dataset like the `label` branch does, with the same ingest options, and writes only under the report folder.

## Language

- **Section**: one analysis, written to its own folder: `profile`, `structure`, `health`, `split`, `leak`. Their contents are in [eda report](report.md).
- **Top N**: the cap on how many classes or groups a per-class or per-group view draws.
- **Baseline**: the reference drawn in a figure so a number can be judged: chance, a shuffled-label floor, a random-split curve.

## Map

- `eda/cli.py`: parsing and dispatch.
- `eda/core/`: the image table, the shared plotting style, one module per section.
- `eda/transform/`: none of its own. It registers the deterministic transforms of the `label` branch.
- `eda/output/commands/to_report.py`: `to-report`.

## Contracts

Usage: `cvsuite eda <src> [<src> ...] [ingest opts] [<transform> <transform args>]... to-report <out> [options]`.

Ingest options are the `label` branch's: `--from {auto,yolo,labelme,coco,images,class-dir,semseg-mask}`, `--keypoints-file`, `--max-depth`. Class-folder datasets need `--from class-dir`.

Transforms: the `label` registry without model-backed transforms, so `select`, `remap-classes`, `split`, `sample`, `filter` and `ops`. Chaining follows [pipeline](../pipeline.md#chaining).

### to-report

```
to-report <out> [--group-by KEY]... [--top N] [--only SECTION[,SECTION...]]
                [--measure {hash,ssim}] [--twin-threshold T] [--max-images N]
                [--seed S] [--overwrite]
```

- `--group-by` takes [key specs](../common/keys.md) and has the meaning it has in `split`: records that belong together. Each section adds per-group panels for the `--top` largest groups, the leak section also reports accuracy within each group, and the split section reports groups that straddle splits.
- `--top N` (default 20) caps every per-class and per-group view. The remainder is drawn as one `other (k)` entry. Tables always list every class and group.
- `--only` runs the listed sections. By default every section that applies runs.
- `--measure` and `--twin-threshold` set the similarity for the split section: `hash` with a threshold in bits (default 6 of 64) or `ssim` with an index (default 0.6) ([similarity](../common/similarity.md)). With `ssim`, the best score is taken over the 25 nearest train images by hash distance, so SSIM refines the hash candidates and does not scan every pair.
- `--max-images N` (default 20000) bounds the quadratic work. When a section would compare more records than N, it samples N of them with [nested sampling](../common/sampling.md) and the report says so.
- `--seed` (default 0) seeds sampling and the leak baselines.
- A report folder that is not empty raises unless `--overwrite`, which replaces only the section folders and files the run writes ([output claims](../common/io.md#output-claims)).

### Which sections run

| section | needs |
|---|---|
| `profile`, `structure`, `health` | any dataset |
| `split` | at least two distinct splits, one of them `train` |
| `leak` | at least two distinct labels, from classification or annotations |

A section whose need is missing is skipped. `report.yaml` lists every skipped section with its reason, and the command prints the same. A name given to `--only` that cannot run raises.

A dataset with folds ([split](../common/split.md#representation)) MUST have exactly one present fold, which `split --only-fold` provides.

## Constraints

- `eda` MUST NOT modify its source or write outside `<out>`.
- A model-backed transform MUST be rejected before ingest.
- Every per-class and per-group view MUST respect `--top` and MUST be accompanied by a view of the whole distribution.
- Every figure that shows a rate MUST draw its baseline.
- A missing `matplotlib` MUST raise before ingest with the install command `pip install "cv-suite[eda]"`.
- The same dataset, options and seed MUST give the same tables.
- Help MUST stay offline and MUST NOT import matplotlib.

## History

- [What belongs in cvsuite](../logs/2026-11-05_what-belongs-in-cvsuite.md): why exploratory analysis passed the test for joining the tool.
- [The eda branch](../logs/2026-11-05_eda-branch.md): why a branch, why sections are folders, and how it scales to 1000 classes.
- [Chained transforms and split as a transform](../logs/2026-11-05_chained-transforms-and-split.md): why a report can follow transforms.
