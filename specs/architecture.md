# Architecture

## 1. General Description
The repository implements a branch-oriented CLI architecture where each top-level subpackage under `src/cvsuite` corresponds either to a public branch or to shared infrastructure. The relationship between CLI shape and repository shape is direct:
- top-level `src/cvsuite/cli.py` owns `cvsuite <branch>` dispatch
- branch packages such as `src/cvsuite/classify`, `src/cvsuite/label`, `src/cvsuite/vlm`, `src/cvsuite/gen`, `src/cvsuite/prep`, and `src/cvsuite/common` own branch-specific or shared CLI surfaces
- `src/cvsuite/common` owns shared compatibility types, runtime-only helpers, FM runtime, stats, cache utilities, and the public `cvsuite common` branch

The architecture deliberately separates three layers:
- branch ingest and output policy
- the shared in-memory dataset contract and its format adapters in `cvsuite.common.core` and `cvsuite.common.io`
- shared FM execution in `cvsuite.common.fm`

## 2. Inputs and Outputs (I/O)
Architecture-level inputs:
- repository layout under `src/cvsuite`
- Python package metadata in `pyproject.toml`
- runtime-discovered `commands/` packages and static command registries
- provider modules and setup scripts under `src/cvsuite/common/fm/providers`

Architecture-level outputs:
- the `cvsuite` entry point
- branch CLIs and generic branch command discovery
- provider subprocess launches under `[[common/fm_runtime]]`
- on-disk exports written by branch-owned output commands

## 3. Interfaces
Top-level interfaces:
- `cvsuite` is declared as `cvsuite.cli:main`
- `cvsuite.cli._discover_branches()` scans child directories that contain `cli.py`
- `classify` is publicly aliased to `class`; underscore names are rendered as hyphenated names

Repository structure and ownership:
- `src/cvsuite/common`
  - `core`: the canonical dataset contract (`VisionDataset`, `VisionRecord`, `ImageInfo`, geometry types), the runtime dataclasses (`FMRequest`, `Embedding`, `LabelMeShape`), the `Task` enum, and JSON helpers
  - `io`: the dataset format router and one adapter per on-disk format (`[[common/io]]`)
  - `fm`: provider runtime, cache/hub root resolution (`fm/core/paths.py`), provider registry, provider implementations
  - `stats`: reusable stats emitters used by file-writing outputs
  - `cache`: CLI-accessible FM cache inspection, root reporting, and purge commands
- `src/cvsuite/classify`
  - explicit CLI registry for one optional transform plus one output command
  - owns the branch-specific multi-class ingest mode and output split policy
- `src/cvsuite/label`
  - explicit CLI registry with nested transform packages (`filter`, `ground`, `ocr`, `ops`, `sample`) and output commands
  - owns multi-source merge semantics, CLI spellings, helper YAML emission, and preview flows
- `src/cvsuite/vlm`
  - explicit CLI registry for one optional transform plus one output command
  - owns answer partitioning, prompt seeding, and cleanup of extracted temp resources
- `src/cvsuite/gen`
  - explicit action registry followed by explicit output registry
  - reuses label-style ingest for edit-source workflows
- `src/cvsuite/prep`
  - generic subcommand discovery across `commands/` directories using subparsers, not the record-pipeline pattern used by other branches
- `src/cvsuite/common/io`
  - the canonical `read_dataset(...)` / `write_dataset(...)` entrypoints and registry
  - one adapter per format: YOLO, COCO, LabelMe, semantic-mask, flat images, flat VLM JSON, VQA-style, shards, and class-dir

Generic command discovery:
- `[[cli]]` covers the user-facing side
- internally, `cvsuite.branch_cli.discover_commands()` recursively finds `commands/` folders, converts relative path parts into hyphenated CLI names, imports `attach(parser)` plus `run(args)` when present, and skips `venv` and `__pycache__`
- `common` and most of `prep` use this generic discovery pattern or a close prep-specific variant; other branches implement custom parsing because they need pipeline-style argument splitting

Shared pipeline CLI helpers (`cvsuite.branch_cli`):
- pipeline branches (`classify`, `label`, `vlm`, `gen`) share four helpers defined in `branch_cli`:
  - `build_parser(module, prog)` — builds an `ArgumentParser` from a command module's `attach()` and docstring
  - `parse_help(parser, argv)` — triggers the parser's `--help` path and returns `0`; re-raises non-help exits
  - `split_rest(rest, commands, transform_commands)` — splits the argv tail into `(transform_name, transform_args, command_name, command_args)`; returns `(None, [], None, rest)` when no recognized output command is present
  - `apply_transform_output_hints(prog, transform_name, parsed_transform, command_name, parsed_command)` — currently propagates `--hardlink` from sampling transforms to outputs that support it
- branch-specific split logic stays local when the semantics differ from the shared pattern

## 4. Business Decisions and Strict Policies
- CLI design follows branch semantics, not one shared parser abstraction. This is why `label`, `classify`, `vlm`, and `gen` each keep custom parsers even though `common` and most of `prep` can reuse generic discovery ideas.
- Only packages with `cli.py` become public top-level branches. Nested packages are not public branches.
- The shared dataset contract is owned once, in `cvsuite.common.core`. No branch defines a competing dataset model.
- Active branch routers call `cvsuite.common.io.read_dataset(...)` / `write_dataset(...)` for canonical formats rather than maintaining duplicate branch-owned readers and writers.
- The FM layer is intentionally prohibited from deciding export layouts or source detection.

## 5. Implementation Details
Concrete data flow by branch:
1. `[[cli]]` dispatches to a branch CLI.
2. The branch CLI splits argv into source tokens, optional transform/action tokens, and output tokens.
3. Source paths are ingested into the shared dataset contract via `[[common/io]]`, or processed directly by prep.
4. Optional transforms mutate dataset contents or attach `fm_request`.
5. If FM-backed, the branch uses `[[common/fm_runtime]]`, which spawns a provider subprocess under `cvsuite.common.fm.providers.*`.
6. The branch runs a writer/output command and may call `[[common/stats]]`.

Important cross-links:
- `[[overview]]`
- `[[cli]]`
- `[[common/cli]]`
- `[[dataclasses]]`
- `[[common/io]]`
- `[[common/fm_runtime]]`
- `[[common/fm_provider]]`
- `[[common/fm_utils]]`
- `[[common/fm_cache]]`
