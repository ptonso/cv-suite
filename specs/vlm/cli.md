# VLM CLI

## 1. General Description
The VLM branch is exposed as `cvsuite vlm` and implements a source-ingest, optional-single-transform, single-output pipeline specialized for captioning, VQA answering, answer mapping, and VLM dataset reformatting. It mirrors the classify CLI shape but adds provider-catalog help and cleanup semantics for temporary extracted resources. All supported source formats ultimately normalize into `record.vqas`; `[[vlm/adapters]]` documents the format-specific details.

The active structured-format routing is delegated to `cvsuite.common.io`, while the branch keeps ownership of legacy CLI spellings, prompt seeding, answer partitioning, provider help, and cleanup behavior.

## 2. Inputs and Outputs (I/O)
Canonical usage:
- `cvsuite vlm <src> [ingest opts] [<transform> <transform args>] <command> [args]`

Base inputs:
- positional `src`
- `--from {auto,images,json,shards,vqa-style}`
- `-h/--help`

Transform commands:
- `caption`
- `ask`
- `map-answers`
- `vqa`

Output commands:
- `to-json`
- `to-shards`
- `to-vqa-style`
- `inspect`

## 3. Interfaces
Help interfaces:
- branch usage help via `cvsuite vlm --help`
- provider catalog help via `cvsuite vlm --provider <provider> -h`
- transform help via `cvsuite vlm <transform> --help`
- output help via `cvsuite vlm <output> --help`
- all of the above also work with a leading `<src>` present (e.g.
  `cvsuite vlm ./ds caption --help`); `--help` is dispatched from the raw argv
  to the most specific parser named before it

Provider-catalog interface:
- when transform help is requested and a provider/model token is already present, the CLI appends that provider’s catalog to the transform parser epilog
- provider/model names are normalized before catalog lookup

Execution interface:
- source ingest occurs through `vlm.core.router.ingest(Path(src), from_hint=...)`
- transform modules run as `run(dataset, parsed_transform)`
- output modules run as `run(dataset, parsed_output)`
- `io.cleanup_dataset_resources(dataset)` always runs in `finally`

Format mapping interface:
- `images` -> canonical `flat`
- `json` -> canonical `flat_vlm_json`
- `shards` -> canonical `shards_vlm`
- `vqa-style` -> canonical `vqa_style`

## 4. Business Decisions and Strict Policies
- Format detection is branch-owned and can fail on ambiguous directory layouts rather than guessing.
- Temporary resources created during shard extraction are tracked in dataset metadata and must be cleaned even on errors or early exits.
- Provider help is explicit because VLM providers have model catalogs and alias normalization that users need before runtime.
- Like classify, only one transform stage is supported before the output command.
- Help must stay deterministic and offline. Catalog text comes from the shared provider registry rather than live network queries.
- cvsuite preserves legacy VLM source spellings and some legacy metadata values even though the canonical adapter ids in `cvsuite.common.io` use underscore-based format names.

## 5. Implementation Details
Shared helpers from `cvsuite.branch_cli`:
- `build_parser`, `parse_help`, `split_rest` — see `[[architecture]]` § "Shared pipeline CLI helpers"
- `apply_transform_output_hints` is not used by this branch
- missing source returns exit code `1`, not `0`

Router detection precedence:
- directory:
  - flat JSON
  - shards
  - VQA-style
  - fallback images
- file:
  - shards
  - VQA-style
  - images

Ambiguity policy:
- if a directory matches multiple structured VLM formats, ingest fails instead of choosing one silently

Related docs:
- `[[vlm/adapters]]`
- `[[vlm/transforms]]`
- `[[vlm/output]]`
- `[[common/fm_runtime]]`
