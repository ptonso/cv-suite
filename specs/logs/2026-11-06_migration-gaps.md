# Gaps found by migrating the sibling projects

Affects: [sampling](../common/sampling.md), [keys](../common/keys.md), [split](../common/split.md), [io](../common/io.md), [label transforms](../label/transforms.md), [classify transforms](../classify/transforms.md), [label cli](../label/cli.md), [eda cli](../eda/cli.md).

Recorded from the migration of the four sibling projects and the maintainer's decisions, 2026-11-06.

## Situation

Migrating pcb-mc-ml, synth-label, petro-cv and teia-lab to the chained commands ([what belongs in cvsuite](2026-11-05_what-belongs-in-cvsuite.md)) retired some project tools and kept others. Each kept tool did one thing the commands cannot say yet:

- teia-lab samples a fraction inside every split and class. `sample` was global, and `class sample` stratified by class only.
- pcb-mc-ml keeps the paper's test photos exactly and deletes every other photo of a board that holds a test photo. `split --keep` pins such a group whole, so the test set grows.
- pcb-mc-ml balances small and large boxes as separate columns. `split --stratify-by` counted classes or one key.
- pcb-mc-ml adds synthetic background boxes to photos, and cuts them with the same context geometry as real boxes. `make-negatives` wrote new image windows that `crop-dets` could not cut.
- SDNET and the raw PCB-MC folds keep their images outside the COCO folder, and the COCO reader looked next to the JSON only.

## Alternatives considered

- **Assigning or renaming splits** (`set-split`, `split --map`, `--split-map` at ingest) so that the paper's `valid` photos become `test`. `set-split` collides with `split` in meaning. `split --map` makes `split` sometimes re-split and sometimes rename. An ingest option fits best but serves one project that can stage a folder named `test` itself, as synth-label already does for SDNET. Nothing ships.
- **Sampling per split** as a flag (`--per-split`). It solves one key. A `--stratify-by` that takes any key, with `split` among them, solves it and every other fraction-per-group case, and it matches the option `split` already has.
- **Keeping `class sample --mode preserve` and `label sample` apart.** Both now accept the same strata, and the per-class rule is a stratum of its own. The wider merge of the `class` and `label` branches is a separate debate, taken last.
- **Name of the pinned-group option** (`--on-pinned-group`, `--groupmates`, `--keep-exact`). The first describes the mechanism, the second sits apart from `--keep`, and the third hides that two behaviors exist. `--on-siblings {join,drop}` reads as a sentence with `--keep` and names the records it acts on.
- **COCO image folders in `data.yaml`.** A raw folder stays untouched, and a source can have two COCO files with two image folders. A repeatable `--images-dir` covers both.
- **A separate transform for background boxes.** The placement rules of `make-negatives` (no overlap with annotations, sizes drawn from real boxes, a ratio per image, a shortfall error) are exactly what the boxes need. Only the product differs, so the product becomes an option.

## Decision

`sample` takes `--stratify-by` with the key grammar plus `split`. `split` gains `--on-siblings {join,drop}` and the `class-size` stratification. The COCO reader takes `--images-dir`, repeatable. `make-negatives` gains `--emit {image,box}`, `--label` and `--within`. Delta 7 carried the work.

## What tipped it

The board identity of pcb-mc-ml is one regex over file names, which `--group-by` already accepts. The remaining differences were the three behaviors above and the image folder, each needed by at least two of the four projects or by a general pattern: fraction sweeps that keep the evaluation splits' proportions, test sets that must stay exact, background samples for classifiers.

## Trade-offs accepted

- `class-size` reads image sizes, which opens every image that has not been measured. The `eda` table already holds them, and a split in a chain pays the cost once.
- `--on-siblings drop` deletes records. The report lists how many, and a run that drops nothing is silent.
- pcb-mc-ml's paper folds still need a staged folder, because renaming a split is not shipped.
