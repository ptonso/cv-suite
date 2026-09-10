# Classify CLI

## 1. General Description
The classify branch is exposed publicly as `cvsuite class`, not `cvsuite classify`. Its CLI implements a single-ingest, optional-single-transform, single-output pipeline specialized for class-directory workflows. The branch is responsible for source detection, classification-specific ingest heuristics, optional FM inference or class-preserving sampling, and class-directory export through `to-class-dir`.

The shared dataset containers used during this branch are from `cvsuite.common.core` (`VisionDataset`, `VisionRecord`, `ImageInfo`), even though cvsuite still exposes compatibility aliases such as `Record` and `ImageRecord`.

## 2. Inputs and Outputs (I/O)
Canonical usage:
- `cvsuite class <src> [ingest opts] [<transform> <transform args>] <command> [args]`

Base CLI inputs:
- positional `src`
- `--from {auto,images,unstructured,multi-class}`
- `-h/--help`

Registered transform commands:
- `infer`
- `sample`

Registered output commands:
- `to-class-dir`

Ingest modes:
- `auto`
- `images`
- `unstructured`
- `multi-class`

Output behavior:
- prints branch usage when help is requested or no source is given
- routes to one output command
- returns `0` for normal dataset or writer returns through `result_to_exit_code`

## 3. Interfaces
CLI splitting rules:
- source parsing happens first through a small base parser
- remaining tokens are split into:
  - optional transform name and transform args
  - required output command and output args
- only one optional transform is allowed before the output command

Help interfaces:
- `cvsuite class --help` prints pipeline usage
- `cvsuite class infer --help` and `cvsuite class sample --help` print transform help
- `cvsuite class to-class-dir --help` prints output help
- a leading `<src>` does not change this: `--help` is dispatched from the raw
  argv to the most specific transform/command parser named before it

Ingest routing interfaces:
- `images` routes through canonical format `flat`
- `unstructured` routes through canonical format `class-dir`
- `multi-class` is still branch-owned special ingest logic because it collapses duplicate basenames across class folders into one shared record

Transform/output wiring:
- if the chosen transform is `sample` and the user passed `--hardlink`, the CLI copies that hint into the output command, but only if the output command actually exposes a `hardlink` argument

## 4. Business Decisions and Strict Policies
- `cvsuite classify` is rejected by the top-level CLI; only `cvsuite class` is valid.
- The branch refuses `sample` when the source looks like an annotated dataset and explicitly tells the user to use `cvsuite label sample` instead.
- Multi-class ingest is a first-class ingest mode and is preserved in dataset metadata so downstream logic can behave differently.
- If the CLI cannot find a recognized output command in the remaining argv, it prints usage and raises `SystemExit` with the unknown token name.
- Missing source (no `<src>` provided) returns exit code `1`, not `0`.
- Help handling is command-aware rather than argparse-subparser-based so prompt strings or output-command-like strings can still be accepted as ordinary arguments in other positions.

`--from multi-class` specific policies:
- expected layout is `<src>/<class>/<filename>` or nested class paths such as `<src>/lights/canopy_light/<filename>`
- the filename is the deduplication key
- duplicate filenames across class folders collapse into one shared record
- split-prefixed trees such as `train/` or `val/` are rejected in this mode; users must use `--from unstructured` instead
- if two files share a filename but have different image bytes or dimensions, ingest fails fast
- downstream `to-class-dir` automatically exports those source labels as multi-class even without `--multi-class`

## 5. Implementation Details
Shared helpers from `cvsuite.branch_cli`:
- `build_parser`, `parse_help`, `split_rest`, `apply_transform_output_hints` — see `[[architecture]]` § "Shared pipeline CLI helpers"

Command registry is dynamic but shallow:
- outputs are discovered from `classify/output/commands/*.py`
- transforms are discovered from `classify/transform/commands/*.py`
- CLI names are module names with underscores converted to hyphens

Execution flow:
1. Parse base args and usage/help.
2. Split remaining tokens into transform and output stages.
3. Ingest via `classify.core.io.ingest(...)`.
4. Optionally run the transform module’s `run(dataset, args)`.
5. Run the output module’s `run(dataset, args)`.

Related docs:
- `[[classify/transforms]]`
- `[[classify/output]]`
- `[[common/io]]`
- `[[common/records_dataset]]`
