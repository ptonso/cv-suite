# Explore a dataset

`cvsuite eda` looks at a dataset before you train on it: what it looks like, how the classes are distributed, whether anything is broken, whether the split can be trusted and whether the label can be guessed without looking at the object. It writes figures and tables into a report folder and never touches the source.

```bash
pip install "cv-suite[eda]"
cvsuite eda ./my-dataset to-report ./report
```

The report folder holds one folder per section and one table of images:

```
report/
  report.yaml            what ran, what was skipped and why, the transforms applied
  images.csv             one row per image: path, split, class, size, brightness, sharpness, perceptual hash
  profile/figures/       contact sheet, examples of the largest and rarest classes, extremes
  structure/figures/     class counts, class by split, long tail, box sizes, centers, resolutions
  structure/tables/      structure.csv, box_sizes.csv
  health/tables/         health.csv
  split/figures/         twin curve, twin examples, class balance, independent units, missing classes, group straddle
  split/tables/          twins.csv, class_balance.csv, units.csv, missing_classes.csv, straddle.csv
  leak/figures/          metadata cues, cue bins, cue contribution
  leak/tables/           guess_accuracy.csv, cue_bins.csv, cue_contribution.csv
```

A section whose requirements are missing is skipped and the command says why. A dataset with a single split has no `split` section, for example, and a dataset with one class has no `leak` section.

## Narrow the question first

`eda` follows the same shape as the other branches, so the report describes the result of any transforms in front of it:

```bash
cvsuite eda ./visa select --drop-class Background --empty drop \
    split --group-by dir:0 --val-frac 0.15 --test-frac 0.15 \
    to-report ./report --group-by dir:0
```

Class folders need `--from class-dir`. Model-backed transforms (`ground`, `ocr`) are not available: write the labels with `cvsuite label` first.

## A thousand classes

Every per-class view draws the `--top` largest classes (default 20) plus one `other (k)` bar, and a companion figure shows the whole distribution. The tables list every class. A dataset with 1000 classes therefore gives the same readable figures as one with 10.

## One panel per group

`--group-by` takes the same key specs as `split`. With it, `images.csv` gets a `group` column, and `profile` and `structure` repeat for the `--top` largest groups under `<section>/by_group/<group>/`. The pooled figures stay in place.

## Checks in `health.csv`

Unreadable or zero-size images, boxes or polygons outside the image, zero-area boxes, duplicate boxes (IoU of at least 0.99), class ids outside the class list, images without annotations, file names shared by different folders, and exact duplicate files. Each row has a count and up to ten example paths.

## Can the split be trusted? (`split`)

The section measures how close each val and test image is to the train set. `01_twin_rank.png` ranks every eval image by its best similarity to any train image, most similar first, one curve per split. A dotted line marks the twin threshold (`--twin-threshold`, 6 bits of 64 by default), and the dashed curves are the baseline: the same measurement after the images are reassigned to train, val and test at random with the same sizes. The title states the share of eval images at or above the threshold.

Read the curve against its baseline. A random split of a collection with bursts or re-exports already shows some twins, so a share of 5% means little when the baseline says 5%. A share of 60% against a baseline of 5% means the split puts copies of the train images in the eval sets, and test accuracy measures memory. `02_twin_examples.jpg` shows an eval image beside its closest train image at the 0%, 1%, 5%, 25% and 75% marks of the ranked list, with the score printed, so you can judge what the threshold lets through.

`--measure ssim` replaces the hash by the structural similarity index on 160 pixel gray thumbnails (threshold `--twin-threshold 0.6` by default). SSIM refines the hash: the 25 nearest train images by hash distance are scored and the best one counts. It costs about 17 ms per eval image, so a 20000 image eval set takes minutes. `--max-images` (default 20000) bounds the work: a split that holds more images is sampled with the nested sampler, and `report.yaml` lists every sample taken.

The other figures: `03_class_balance.png` (each class's share per split against the splits' shares of all instances), `04_independent_units.png` (images linked by the hash threshold count once: a class with 40 test images in 5 bursts has 5 independent units, and the figure prints the worst-case 95% half-width of an accuracy measured on them), `05_missing_classes.png` (classes with no, fewer than 5 and fewer than 20 images in each eval split, with the names in `missing_classes.csv`), and with `--group-by`, `06_group_straddle.png` (groups that hold images of more than one split) and a twin curve per large group under `split/by_group/<group>/`.

## Can the label be guessed without the object? (`leak`)

The section fits the cheapest predictor on file facts and shows how far it gets. The cues are the resolution, the aspect ratio, the file size, the container (extension, JPEG quality estimate and EXIF), the brightness, the sharpness, the digits of the file name, the rank of the name in sorted order and the modification minute. For each cue value the likelihood rule predicts the class with the largest share of its fitting records showing it, and the figure reports the balanced accuracy on the scoring set: fit on `train`, score on `test` (on `val` when there is no `test`). A dataset without a usable split is scored by grouped five-fold cross-validation, with `--group-by` as the groups.

Classification datasets are scored on the record's label. Annotated datasets are scored one-versus-rest on the presence of each of the `--top` classes by image count, and the figure shows the pooled mean.

`01_metadata_cues.png` draws every cue with two baselines: the dashed chance line (`1/k`, or 0.5 one-versus-rest) and a black diamond for the shuffled-label floor, the same rule fitted on shuffled training labels, averaged over 20 shuffles. A cue at its floor carries no label. A cue far above its floor, such as file size at 0.75 against a floor of 0.37, means the label can be read off the file: files of one class were exported by a different tool, from a different camera or on a different day. A model trained on such a dataset can learn that shortcut and fail on data from another source. `02_cue_bins.png` shows which values carry the signal, and `03_cue_contribution.png` compares the combined cue (file size, brightness and sharpness) with each cue alone and adds one cue at a time, largest gain first. With `--group-by`, the cue figure also marks the accuracy within each large group, which shows whether the cue survives once the group is fixed. Groups that sit wholly in one split cannot be fitted and scored on their own and are left out.

## Rerunning

A report folder that already holds output raises. `--overwrite` replaces only the section folders and files the run writes, so `--only structure --overwrite` refreshes one section and leaves the others.
