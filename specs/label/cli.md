# Label CLI

Related: **Read first** [pipeline](../pipeline.md). **See also** [label adapters](adapters.md), [label transforms](transforms.md), [label output](output.md), [io](../common/io.md).

## Overview

`cvsuite label` works on annotated datasets (boxes, polygons, keypoints, masks) and on plain image folders that need annotations. It reads one or more sources, applies any number of transformations in order (sampling, selecting, remapping classes, splitting, model grounding, OCR, geometry operations, filtering), and writes an output. It follows the [pipeline](../pipeline.md) shape.

Shared parsing and writing of formats is delegated to [io](../common/io.md). The branch keeps public spellings, source-detection policy, multi-source merging and label-specific outputs.

Example: `cvsuite label ./raw ground --provider gdino --prompt "car,person" split --val-frac 0.2 to-yolo ./out`.

## Language

- **Multi-source merge**: ingesting several sources into one dataset, remapping class ids by label text.

## Map

- `label/cli.py`; `label/core/router.py` ([adapters](adapters.md)); `label/core/schema.py`.
- Transforms: `label/transform/<project>/commands/*.py` ([transforms](transforms.md), [filter](filter.md)).
- Outputs: `label/output/commands/*.py` ([output](output.md)).

## Contracts

Usage: `cvsuite label <src> [<src> ...] [ingest opts] [<transform> <transform args>]... <command> [args]`.

Base arguments: one or more `src`; `--from {auto,yolo,labelme,coco,images,class-dir,semseg-mask}`; `--keypoints-file <path>` (LabelMe only); `--max-depth <n>` (recursion limit for flat and class-dir readers, `-1` unlimited); `--images-dir <dir>` (COCO only, repeatable); `-h/--help`.

Transforms: `filter`, `ground`, `ocr`, `ops`, `sample`, `select`, `remap-classes`, `split`. Outputs: `to-yolo`, `to-labelme`, `to-coco`, `to-results`, `audit`, `inspect`. The model-backed transforms are `ground` and `ocr`. Stage splitting, the chain rules and the steps record follow [pipeline](../pipeline.md#chaining).

Public source spellings map to canonical format ids: `images` and detected unstructured trees to `flat`; `yolo`, `labelme`, `coco`, `class-dir` to themselves; `semseg-mask` to `semseg_mask`.

Transform command names come from `label/transform/<proj>/commands/*.py`: module `run` gives the project name, any other module gives `<proj>-<module>`. Outputs come from `label/output/commands/*.py`.

Multi-source merge: records are deep-copied, image paths absolutized against each source root, class ids remapped by label text, so COCO sources whose category ids differ merge by class name. Metadata: `meta["multisrc"] = True`, `meta["sources"]` (path strings), `meta["source_formats"]`; `root` becomes the common path when possible.

Auto-detection order is in [adapters](adapters.md#format-detection).

## Constraints

- The chain rules of [pipeline](../pipeline.md#constraints) apply: any number of transforms, at most one model-backed and first, temporary folders removed after the output.
- The base parser MUST use `allow_abbrev=False`, so a flag such as `--max` is never taken as a prefix of `--max-depth`.
- `--help` MUST go to the most specific parser named before it ([pipeline](../pipeline.md#help)).
- Multi-source merging MUST be label-centric: class ids are reallocated by class name.
- The merged dataset task MUST be kept only when all non-null source tasks agree; otherwise it is downgraded.
- YAML semantic-segmentation manifests MUST be detected before the generic YOLO fallback when they hold split-level `images` and `masks` mappings.
- Older spellings remain at output boundaries, such as `to-yolo --task seg` ([output](output.md)).

## History

- [Branch per domain](../logs/2026-09-20_branch-per-domain.md).
- [Chained transforms and split as a transform](../logs/2026-11-05_chained-transforms-and-split.md): why a run takes several transforms.
- [Gaps found by migrating the projects](../logs/2026-11-06_migration-gaps.md): why COCO takes `--images-dir`.
