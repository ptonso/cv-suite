# Changelog

Versions follow [semver](https://semver.org). While the major version is 0, a minor release may break the command line.

## 0.1.0 - 2026-11-06

### Breaking

- `--hardlink` is gone from `label to-coco`, `to-labelme`, `to-yolo`, `class sample`, `class to-class-dir`, `vlm to-json` and `vlm to-vqa-style`. Use `--link-mode {copy,hard,sym}`. A failed link raises. `prep sample` takes `--link-mode {copy,hard}` in place of `--hardlink`.
- The split flags `--val-frac`, `--test-frac`, `--seed`, `--preserve-splits` and `--randomized` are gone from the output commands. Assign splits with the `split` transform before the output command.
- An output refuses a destination that an earlier run filled, with `FileExistsError`. Pass `--overwrite` to replace it. Reruns no longer write `_1`, `_2` copies beside the first output.
- `to-yolo --task det` raises on a dataset that holds only polygons unless `--boxes-from-polygons` is given.
- The dataset metadata key `meta["sample"]` is replaced by the ordered list `meta["steps"]`.

### Changed results

- Outputs keep the split each record carries. A dataset that arrives with train and val folders keeps them, where an output used to flatten and reassign them.
- The COCO reader keeps images that carry no annotations.
- `gen to-dst` copies the original source images by default instead of hardlinking them. Pass `--link-mode hard` for the old behavior.

### Added

- Chained transforms: a run applies any number of transforms left to right before one output command.
- Transforms `split` (groups, strata, cross-validation folds, pinned splits), `select` and `remap-classes`, shared by `label` and `class`.
- `--stratify-by` for `sample`, `--folds` for `split`, `--images-dir` for COCO sources, `--boxes-from-polygons` for `to-yolo`, `--overwrite` and `--link-mode` on the outputs that write images.
- The `eda` branch, with image table, structure, health, split and leak sections, written as figures and tables by `eda to-report`. It needs `pip install "cv-suite[eda]"`.
- Layered `--help`: the branches, the commands of one branch with ingest options and examples, and the flags of one command.

## 0.0.1 - 2026-09-10

Initial public release: the `label`, `class`, `vlm`, `gen`, `prep` and `common` branches with their shared dataset, format adapters and isolated foundation-model runtime.
