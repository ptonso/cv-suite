# Architecture

Related: **Read first** [README](README.md). **See also** [pipeline](pipeline.md), [records_dataset](common/records_dataset.md), [fm_runtime](common/fm_runtime.md).

## Overview

`cvsuite` is a command-line tool for computer-vision dataset work. You give it a source folder, optionally apply transformations, and it writes a destination folder. It is exposed as `cvsuite` through the console entry point in `pyproject.toml`.

The tool is split by domain. Each domain of computer vision gets one top-level command (a *branch*), and the commands inside a branch are what you can do in that domain. There is no "convert anything to anything" command: each branch owns how it recognizes its input, what its transformations mean, and how its output is laid out. The reasons are in [the branch-per-domain note](logs/2026-09-20_branch-per-domain.md).

The branches:

- `prep` cleans up raw files before any dataset format applies: unpacking archives, removing duplicates, resizing, fixing rotation, sampling.
- `label` works on annotated datasets (boxes, polygons, keypoints, masks), including model-assisted grounding and OCR.
- `class` works on class-folder datasets, including model-based classification.
- `vlm` works on image and question-answer datasets: captioning, asking questions, answer mapping.
- `gen` generates or edits images from text prompts.
- `eda` explores a dataset before experiments and writes figures and tables: what it contains, how it is distributed, whether its split can be trusted, whether its labels can be guessed from file metadata.
- `common` holds shared operations. Its public surface is model-cache management.

Three layers sit under the branches:

1. **Branch policy**: source detection, transformations and output layout, owned by each branch.
2. **The shared dataset**: one in-memory model of images and annotations, plus one adapter per on-disk format ([records_dataset](common/records_dataset.md), [io](common/io.md)). Every branch except `prep` reads sources into it and writes outputs from it.
3. **Model execution**: foundation models run as separate processes ([fm_runtime](common/fm_runtime.md), [fm_provider](common/fm_provider.md), [fm_cache](common/fm_cache.md)). This layer never decides output layout or source detection.

Walk-through of a typical run, `cvsuite label ./data ground ... split --val-frac 0.2 to-yolo ./out`: the top-level command hands the arguments to the `label` branch; the branch reads `./data` into the shared dataset; the `ground` transformation attaches a model request and the model runtime runs the model in its own process; the model adds boxes to the dataset; `split` assigns each record to `train` or `val`; `to-yolo` writes the dataset as a YOLO tree. The shared shape of this flow is in [pipeline](pipeline.md#chaining).

The tool aims at reproducible results. Sampling and split assignment use fixed default seeds, and format outputs refuse a destination that an earlier run filled unless `--overwrite` is given ([output claims](common/io.md#output-claims)). Filename collisions between records of one run get a suffix instead of overwriting.

For install steps and worked examples see [`../docs/`](../docs/). Specs constrain; guides teach.

## Language

Terms used across several specs. Terms specific to one spec are defined there.

- **Branch**: a top-level command (`prep`, `label`, `class`, `vlm`, `gen`, `eda`, `common`).
- **Pipeline branch**: `class`, `label`, `vlm`, `gen`, `eda`. They follow the [pipeline](pipeline.md) shape.
- **Ingest**: reading a source folder into the shared dataset.
- **Transform**: a step that changes the dataset before output. In `gen` it is called an *action*. A run may chain several ([pipeline](pipeline.md#chaining)).
- **Step**: one transform of a run as recorded in `dataset.meta["steps"]`, with its arguments and what it reported.
- **Group**, **key spec**: records that belong together and the string that says how to find them ([keys](common/keys.md)). **Stratum column**, **fold**: see [split](common/split.md).
- **Output command**: writes the dataset to a branch-owned on-disk form.
- **Shared dataset**: `VisionDataset`, made of `VisionRecord`s. See [records_dataset](common/records_dataset.md).
- **Canonical format id**: the name of an adapter in [io](common/io.md), such as `yolo` or `flat_vlm_json`. Branch spellings map onto these.
- **FM**: foundation model. **Provider**: the wrapper around one FM, such as `clip`, `qwen`, `sam3`, `openrouter`. **Family**: what a provider does: `classify`, `create`, `edit`, `ground`, `ocr`, `vlm`. **Local provider** runs in its own managed virtual environment, an **API provider** runs in the current interpreter. See [fm_provider](common/fm_provider.md).
- **`fm_request`**: the model request attached to a dataset before a model runs ([records_dataset](common/records_dataset.md)).
- **Prompt** means different things per branch: for `class`, the class vocabulary or a label-to-prompts mapping; for grounding, the phrases to find or a label-to-prompts mapping; for `vlm`, questions; for `gen`, text prompts.

## Map

- `src/cvsuite/cli.py`: top-level dispatch. `src/cvsuite/branch_cli.py`: shared command-line helpers.
- `src/cvsuite/common/`: shared code.
  - `core/`: the shared dataset ([records_dataset](common/records_dataset.md)).
  - `io/`: format router and adapters ([io](common/io.md)).
  - `keys.py`, `splitting.py`, `similarity.py`, `sampling.py`, `transforms/`: record keys, the split packing, perceptual similarity, nested sampling, and the `select` and `remap-classes` logic ([keys](common/keys.md), [split](common/split.md), [similarity](common/similarity.md), [sampling](common/sampling.md), [select and remap](common/select_remap.md)).
  - `fm/`: model runtime, provider registry, providers ([fm_runtime](common/fm_runtime.md), [fm_provider](common/fm_provider.md), [fm_utils](common/fm_utils.md)).
  - `cache/`: cache commands ([fm_cache](common/fm_cache.md), [common cli](common/cli.md)).
  - `stats/`: `stats.yaml` builders ([stats](common/stats.md)).
- `src/cvsuite/prep/`: [prep cli](prep/cli.md), [transforms](prep/transforms.md), [output](prep/output.md).
- `src/cvsuite/label/`: [cli](label/cli.md), [adapters](label/adapters.md), [transforms](label/transforms.md), [filter](label/filter.md), [output](label/output.md).
- `src/cvsuite/classify/`: [cli](classify/cli.md), [transforms](classify/transforms.md), [output](classify/output.md).
- `src/cvsuite/vlm/`: [cli](vlm/cli.md), [adapters](vlm/adapters.md), [transforms](vlm/transforms.md), [output](vlm/output.md).
- `src/cvsuite/gen/`: [cli](gen/cli.md), [transforms](gen/transforms.md), [output](gen/output.md).
- `src/cvsuite/eda/`: [cli](eda/cli.md), [report](eda/report.md).

Each branch package holds `cli.py`, `core/`, `transform/` (or `commands/` for `prep`) and `output/`. `eda` has no `transform/` of its own, because it registers the deterministic transforms of `label`.

## Contracts

### Top-level dispatch

`cvsuite.cli:main` is the console entry point.

```
cvsuite                      # usage
cvsuite -h | --help          # usage
cvsuite <branch> [args]      # runs <branch>'s main()
```

- Public branches are the child packages of `src/cvsuite/` that contain a `cli.py`, found by scanning the filesystem. Nested packages are never public.
- Public names come from package names: `classify` is aliased to `class`; other underscores become hyphens. The requested token is normalized the same way, then mapped back to the package and imported as `cvsuite.<package>.cli`.
- An unknown branch prints usage, then raises `SystemExit("Unknown branch: ...")`.
- Top-level help only scans the filesystem and reads the first docstring line of each branch's `cli.py` as that branch's summary; it does not import branch modules. It prints the command shape, one line per branch, how the help levels nest and a few examples.

### Command discovery below the top level

- `common` and `prep` discover commands from `commands/` folders. `cvsuite.branch_cli.discover_commands()` walks `commands/` folders recursively, turns path parts into hyphenated command names, imports `attach(parser)` (optional) and `run(args)`, and skips `venv` and `__pycache__`. `prep` uses a close local variant ([prep cli](prep/cli.md)).
- The pipeline branches use custom parsers. See [pipeline](pipeline.md).

### Help

Help has three levels, from general to specific: `cvsuite --help` (the branches), `cvsuite <branch> --help` (the commands of one branch) and `cvsuite <branch> <src> <command> --help` (the flags of one command). Each level names the next. Help is offline and deterministic. Provider and model catalogs shown in `--help` come from source metadata read through the provider registry, never from live lookups ([fm_provider](common/fm_provider.md)).

## Constraints

- Only packages with a `cli.py` MUST become public branches. `cvsuite classify` MUST be rejected and `cvsuite class` accepted (`src/cvsuite/tests/test_cli_structure.py`).
- The shared dataset MUST be defined once, in `cvsuite.common.core`. No branch MAY define a competing dataset model.
- Branches MUST read and write canonical formats through `cvsuite.common.io.read_dataset` / `write_dataset` and MUST NOT keep duplicate readers or writers.
- `common` MUST NOT choose folder layouts or dataset schemas. The model layer MUST NOT decide output layout or source detection.
- A transform that several branches share MUST keep its logic in `cvsuite.common` and MUST be exposed by a thin command module in each branch that offers it. Branches MUST NOT keep competing copies of it.
- Top-level help MUST NOT import every branch.
- Every command module MUST have a docstring, because its first line is the command's summary in the branch help, and every parameter of every command MUST have help text (`src/cvsuite/tests/test_cli_structure.py`).
- Guardrails belong to the spec that owns the command: for example [`class sample`](classify/transforms.md#sample), [`prep sample`](prep/transforms.md#sample), [`split`](common/split.md#constraints), [`--link-mode`](pipeline.md#link-mode), [`sam3` on CPU](label/transforms.md#ground), and [`--max-gpu-memory`](vlm/transforms.md#constraints).

## History

- [Branch per domain](logs/2026-09-20_branch-per-domain.md): why the CLI is split by computer-vision domain and why `prep`, `class` and the pipeline branches differ.
- [Shared dataset contract](logs/2026-09-20_shared-dataset-contract.md): why one in-memory model sits between formats and models.
- [Idempotent outputs](logs/2026-11-05_idempotent-outputs.md): why a rerun into a dirty destination raises and `--overwrite` replaces what a writer owns.
- [What belongs in cvsuite](logs/2026-11-05_what-belongs-in-cvsuite.md): the test a routine passes to join the tool, and what stays in projects.
- [Chained transforms and split as a transform](logs/2026-11-05_chained-transforms-and-split.md): why a run takes several transforms and why splitting left the outputs.
- [The eda branch](logs/2026-11-05_eda-branch.md): why dataset analysis is a branch.
