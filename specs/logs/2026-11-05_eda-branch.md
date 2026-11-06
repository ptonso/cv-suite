# The eda branch

Affects: [eda cli](../eda/cli.md), [eda report](../eda/report.md), [architecture](../architecture.md), [keys](../common/keys.md), [similarity](../common/similarity.md), [split](../common/split.md), [select and remap](../common/select_remap.md).

Recorded from the maintainer's account and a review of the synth-label analysis figures, 2026-11-05.

## Situation

Before running experiments on a dataset, the maintainer checks what it contains, whether the split is trustworthy, and whether the label can be guessed without the object. synth-label does this with about forty tools, each writing figures and tables per dataset: contact sheets, counts per class and split, train-test twin curves with example pairs, independent test units, and balanced accuracy of guessing the label from file metadata. The tools copy the same plotting style and `guess_accuracy` code from one another. pcb-mc-ml found the same need by hand: some boards were re-exported under two names, so the same pixels landed in train and test.

cvsuite has pieces: `stats.yaml` (counts, no per-split classes or sizes), `audit` (counts of images and annotations), and `inspect` (interactive). Nothing answers the split and leak questions.

## Alternatives considered

- **Extend `audit` and `inspect`.** `audit` is a YAML summary and `inspect` is interactive. Neither produces figures.
- **Put it in `prep`.** `prep` is file-first data preparation, and analysis there would mix two jobs.
- **A separate package.** The analysis depends on the shared dataset, the reader, the overlay drawing and the splitter, so a package would depend on cvsuite for almost everything.
- **Levels as required subcommands** (`eda structure`, `eda leak`). The usual run is all of them, so the sections are output folders, with `--only` to pick some.
- **`--target`, `--versus` and `--control` flags.** The target was a way to focus on one question, and the chain already narrows a dataset with `select` and `remap-classes`. The control was the same idea as `--group-by`, which already carries the shared key grammar.
- **Per-class plots everywhere.** A dataset can have 1000 classes. Every per-class view has a cap and a whole-distribution twin instead.

## Decision

`eda` is a pipeline branch. Its command line is the label branch's, with the output command `to-report`: any number of deterministic transforms, then the analysis, so a dataset can be analyzed as it will be used, including a split that was never written. The contract is in [eda cli](../eda/cli.md) and the sections in [eda report](../eda/report.md).

`--group-by` has one meaning across `split` and `eda`: images that belong together. In `eda` each group gets its own panels and its own leak numbers, and the report lists groups that straddle splits.

Every per-class or per-group view shows at most `--top N` entries plus an `other (k)` entry, next to a plot of the whole distribution, and the tables keep every entry. Every figure carries a baseline: chance, a shuffled-label floor, a random-split curve.

## What tipped it

The 1000-class concern. It ruled out per-class panels as a default and led to the caps, and it made `--target` unnecessary, because a question is narrowed by remapping the vocabulary to a handful of classes before the report.

## Trade-offs accepted

- matplotlib is an optional extra, `cvsuite[eda]`. Running a report without it fails with an install hint.
- Twin analysis compares every val or test image with every train image. `--max-images` bounds the work by sampling, and the report says when it did.
- The default similarity measure is a perceptual hash, which is fast and coarse. SSIM is available with `--measure ssim`.
- The logistic background probe of synth-label is deferred, because it needs scikit-learn and a project-specific idea of background.
- Class-folder datasets need `--from class-dir`, as in the label branch.
