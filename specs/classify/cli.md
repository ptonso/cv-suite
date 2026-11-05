# Classify CLI

Related: **Read first** [pipeline](../pipeline.md). **See also** [classify transforms](transforms.md), [classify output](output.md), [io](../common/io.md).

## Overview

`cvsuite class` works on class-folder datasets: images sorted into one folder per class. It follows the [pipeline](../pipeline.md) shape: read a source, apply any number of transforms (infer classes with a model, sample per class, select, remap classes, split), and write a class-folder tree with `to-class-dir`.

The branch is public as `class`, not `classify`, for brevity ([the branch note](../logs/2026-09-20_branch-per-domain.md)). It owns source detection for class layouts and a special *multi-class* ingest mode.

Example: `cvsuite class ./photos infer --prompt "cat,dog" to-class-dir ./sorted --threshold 0.6` classifies each photo with a model and writes `./sorted/cat/...`, `./sorted/dog/...`, and `./sorted/unlabeled/...` for low-confidence photos.

## Language

- **Unstructured ingest**: class-folder ingest through canonical `class-dir`, where the label comes from the parent folder or the folder before `images`.
- **Multi-class ingest**: an ingest mode where one filename may appear under several class folders and so carries several labels.

## Map

- `classify/cli.py`: parsing and dispatch.
- `classify/core/io.py`: `ingest(...)`.
- `classify/transform/commands/*.py`: `infer`, `sample` ([transforms](transforms.md)).
- `classify/output/commands/*.py`: `to-class-dir` ([output](output.md)).

## Contracts

Usage: `cvsuite class <src> [ingest opts] [<transform> <transform args>]... <command> [args]`.

Base arguments: positional `src`; `--from {auto,images,unstructured,multi-class}`; `-h/--help`.

Ingest routing:

- `images` goes through canonical `flat`.
- `unstructured` goes through canonical `class-dir`.
- `multi-class` is branch-owned logic that collapses duplicate filenames across class folders into one record.

Transform commands: `infer` (model-backed), `sample`, `select`, `remap-classes`, `split`. Output commands: `to-class-dir`. Stage splitting, chaining, help routing and exit codes follow [pipeline](../pipeline.md). Registries are discovered from `classify/transform/commands/*.py` and `classify/output/commands/*.py`.

Execution: parse base args, split and parse every stage, `classify.core.io.ingest(...)`, each transform `run(dataset, args)` in order, output `run(dataset, args)`, then removal of temporary folders.

### Multi-class ingest

- Layout: `<src>/<class>/<filename>` or nested class paths such as `<src>/lights/canopy_light/<filename>`.
- The filename is the deduplication key. Duplicate filenames across class folders become one shared record holding all labels.
- Ingest mode is stored in dataset metadata (`classify_ingest_mode`) and labels in the record attribute `classify_multi_class_labels`, mirrored to the class-dir multi-label attribute (`class_dir_multi_labels`, `CLASS_DIR_MULTI_LABELS_ATTR` in [io](../common/io.md)).

## Constraints

- `cvsuite classify` MUST be rejected; only `cvsuite class` is valid ([architecture](../architecture.md#top-level-dispatch)).
- The chain rules of [pipeline](../pipeline.md#constraints) apply: any number of transforms, and `infer`, the only model-backed one, first.
- `sample` MUST refuse annotated datasets and point to `cvsuite label sample`.
- Multi-class ingest MUST reject split-prefixed trees such as `train/` or `val/`; users MUST use `--from unstructured`.
- Multi-class ingest MUST fail if two files with the same filename differ in image bytes or dimensions.
- `to-class-dir` MUST export multi-class source labels as multi-class even without `--multi-class`.
- Help handling MUST be command-aware and not subparser-based, so prompt strings that look like command names are accepted as arguments.
- Missing `<src>` MUST return exit code `1`; an unrecognized output token MUST print usage and raise `SystemExit` naming the token.

## History

- [Branch per domain](../logs/2026-09-20_branch-per-domain.md): why `class` is its own branch and why it is named `class`.
- [Chained transforms and split as a transform](../logs/2026-11-05_chained-transforms-and-split.md): why a run takes several transforms.
