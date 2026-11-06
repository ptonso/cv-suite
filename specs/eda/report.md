# Eda Report

Related: **Read first** [eda cli](cli.md). **See also** [similarity](../common/similarity.md), [keys](../common/keys.md), [split](../common/split.md), [stats](../common/stats.md).

## Overview

The report is a folder with one subfolder per section and one shared table of images. Each section answers one question about the dataset and writes numbered figures and the tables behind them, so the number a note quotes can be traced to a file. The sections are ordered by how much a reader needs them: look at the data, count it, check it, test the split, test for shortcuts.

Every figure names its question in its title, prints counts on its marks and draws a baseline. Per-class and per-group views stay readable at any class count through one rule: they draw the `--top` largest entries and one `other (k)` entry, and a companion figure shows the whole distribution. The tables keep every entry.

## Language

- **Image table**: `images.csv`, one row per record, read by every section.
- **Cue**: a cheap file-level fact that could predict the label: resolution, file size, brightness, sharpness, container details, file-name digits, file index, file time.
- **Likelihood rule**: predict the class with the largest share of its records showing the cue's value (a quartile bin for numeric cues), fitted on one set and scored on another. Shares make classes of unequal size comparable, as balanced accuracy does.
- **Balanced accuracy**: the mean over classes of the share of each class's records predicted correctly. Chance is `1/k` for k classes.
- **Shuffled floor**: the likelihood rule fitted on shuffled training labels.
- **Independent unit**: a group of records linked by similarity, counted once because its members are not independent test samples.

## Map

- `eda/core/table.py`: the image table and thumbnails. `eda/core/style.py`: the shared plotting style.
- `eda/core/profile.py`, `structure.py`, `health.py`, `split.py`, `leak.py`: one module per section.
- Reuse: overlay drawing in `label/core/annotate/` (`render_overlay`, `detection_crops`, `select_indices`), [similarity](../common/similarity.md), [splitting](../common/split.md) for the grouped fallback in `leak`.

## Contracts

### Layout

```
<out>/
  report.yaml            run summary: sources, resolved steps, options, sections run or skipped with reasons, samples taken
  images.csv             one row per record
  profile/figures/       01_contact_sheet.jpg  02_class_examples.jpg  03_extremes.jpg
  structure/figures/     01_class_counts.png  02_class_by_split.png  03_long_tail.png  04_box_size.png
                         05_instances_per_image.png  06_box_centers.png  07_resolutions.png
  structure/tables/      structure.csv  box_sizes.csv
  health/tables/         health.csv
  split/figures/         01_twin_rank.png  02_twin_examples.jpg  03_class_balance.png
                         04_independent_units.png  05_missing_classes.png  06_group_straddle.png
  split/tables/          twins.csv  class_balance.csv  units.csv  missing_classes.csv  straddle.csv
  split/by_group/<group>/figures/   01_twin_rank.png  (with --group-by)
  leak/figures/          01_metadata_cues.png  02_cue_bins.png  03_cue_contribution.png
  leak/tables/           guess_accuracy.csv  cue_bins.csv  cue_contribution.csv
```

With `--group-by`, `split` also writes `06_group_straddle.png`, `straddle.csv` and the twin curve of each of the `--top` largest groups. `profile` and `structure` also write `<section>/by_group/<group>/` with the same figures and tables for each of the `--top` largest groups. The pooled files stay in place. Figures that need a type of annotation the dataset lacks, such as box size on a classification dataset, are not written.

### images.csv

Columns: `path` (layout path), `split`, `group` (when `--group-by` is given: the key value for one spec, the smallest member path when specs are joined), `class` (the label, or labels joined by `;`, or `(empty)`), `n_instances`, `width`, `height`, `file_bytes`, `ext`, `brightness` (mean gray level), `sharpness` (variance of the Laplacian of the gray image), `phash` (16 hex digits), `mtime`.

### Sections

**profile** (what the data looks like)
- `01_contact_sheet.jpg`: random annotated images with their overlays, in a fixed grid.
- `02_class_examples.jpg`: crops of the `--top` largest and the `--top` rarest classes.
- `03_extremes.jpg`: the images with the largest and smallest boxes, and images without annotations.

**structure** (how much of what)
- Class counts with values on the bars, class by split on a log axis, and the rank-frequency curve of all classes on log axes.
- Box size per class as a share of the image and in pixels, box centers as a heatmap, instances per image, resolutions and aspect ratios.
- `structure.csv`: images and instances per class and split. `box_sizes.csv`: size quantiles per class.

**health** (is anything broken)
- `health.csv` has one row per check with its count and up to ten example paths: unreadable or zero-size images, boxes or polygons outside the image by more than one pixel, zero-area boxes, duplicate boxes of one class (IoU at least 0.99), class ids out of range, images without annotations, file names shared by different folders, exact duplicate files (same size and SHA-1).

**split** (can the split be trusted)
- `01_twin_rank.png`: for each val and test image, the best similarity to any train image, ranked, one curve per split, with the threshold line and the **random-split baseline**: the same measurement for a random split of the same sizes. The title says what share is above the threshold.
- `02_twin_examples.jpg`: a test image beside its closest train image at five similarity levels, scores printed.
- `03_class_balance.png`: each class's share per split against the target, for the `--top` classes.
- `04_independent_units.png` and `units.csv`: per class, the test images, the independent units among them, and the half-width of a 95% interval for an accuracy measured on that many units. Images are linked by hash distance at the twin threshold, or at 6 bits when `--measure ssim` sets the threshold as an index.
- `05_missing_classes.png` and `missing_classes.csv`: the number of classes with zero, fewer than 5 and fewer than 20 samples in val and in test, and the class names.
- `06_group_straddle.png` and `straddle.csv`: groups that hold records of more than one split, per split pair and the `--top` largest. `straddle.csv` has `pair` rows (groups and rows per split pair) and `group` rows (rows per split). Written only with `--group-by`.
- `03_class_balance.png` and `04_independent_units.png` carry the whole class distribution in a second panel, a histogram over every class, beside the `--top` entries.
- `01_twin_rank.png` draws the random-split baseline once per eval split, from one shuffle of the pooled rows with the run seed. `twins.csv` lists every sampled eval row with its closest train row, the score (matching bits of 64 for `hash`, the index for `ssim`) and whether it reaches the threshold.

**leak** (can the label be guessed without the object)
- The cues are the resolution (width and height), aspect ratio, file size, container (extension, and for JPEG the quality estimated from the quantization table and whether EXIF is present), brightness, sharpness, the last digit run of the file stem as an integer (`none` when the stem has no digit), the rank of the stem in sorted order and the modification time truncated to the minute. Numeric cues use quartile bins fitted on the fitting rows, the others exact values, and a value the fitting rows never showed predicts the fitting majority class. The combined cue is the tuple of the file size, brightness and sharpness bins.
- `01_metadata_cues.png`: balanced accuracy of the likelihood rule for each cue, fitted on train and scored on test, with the chance line and the shuffled floor. The target is the record's label for classification data. For annotated data it is the pooled one-versus-rest accuracy for the `--top` classes. With `--group-by`, the figure shows each cue's accuracy pooled and within each of the `--top` largest groups plus one `other (k)` entry. The fit and the score are restricted to the group, so a group that sits wholly in one split, which a grouped split produces, has no points.
- `02_cue_bins.png`: for the most predictive cues, the class share per bin of the cue. It shows which values carry the signal.
- `03_cue_contribution.png`: the combined cue (file size, brightness, sharpness) against each cue alone, and the gain from adding one cue at a time.
- The target is the record's label for classification data (a multi-label record counts for each of its labels) and the one-versus-rest presence of each of the `--top` classes by image count for annotated data. `guess_accuracy.csv` has one row per cue and target, a `(pooled)` row for one-versus-rest data, and the within-group rows. `cue_contribution.csv` lists the forward selection steps. A dataset that has a `train` split and another split is scored by fitting on `train` and scoring on `test` (else `val`).
- A dataset with no splits is scored by grouped five-fold cross-validation through [split](../common/split.md), with `--group-by` as the groups.

## Constraints

- Every section MUST read the dataset and `images.csv` and MUST NOT recompute hashes or thumbnails that the table or its cache already holds.
- Figure and table names MUST match the layout above. A figure that cannot be drawn is omitted and listed in `report.yaml`.
- Every per-class or per-group figure MUST draw at most `--top` entries plus `other (k)`, and MUST have a whole-distribution figure in the same section.
- A rate MUST be drawn with its baseline: the chance line and shuffled floor for the leak cues, the random-split curve for twins.
- The likelihood rule MUST be fitted only on the fitting split and scored only on the other.
- `report.yaml` MUST record every sample taken because of `--max-images`, and under `notes` the measure, threshold and twin shares of the split section and the fit and score plan of the leak section.
- Counts MUST count a polygon and the box derived from it once.

## History

- [The eda branch](../logs/2026-11-05_eda-branch.md): why these sections, the cap rule, and what was left out.
