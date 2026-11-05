# VLM CLI

Related: **Read first** [pipeline](../pipeline.md). **See also** [vlm adapters](adapters.md), [vlm transforms](transforms.md), [vlm output](output.md), [fm_runtime](../common/fm_runtime.md).

## Overview

`cvsuite vlm` works on image and question-answer datasets for vision-language models: captioning, answering questions, mapping free-text answers to fixed labels, and reformatting the datasets. It follows the [pipeline](../pipeline.md) shape and adds provider-catalog help and cleanup of temporary extracted resources. Every supported source format ends up as `record.vqas` ([adapters](adapters.md)).

Format parsing is delegated to [io](../common/io.md). The branch keeps its public spellings, prompt seeding, answer partitioning, provider help and cleanup.

Example: `cvsuite vlm ./photos caption --provider qwen to-json ./captions`.

## Map

- `vlm/cli.py`; `vlm/core/router.py`, `io.py`, `adapters/`, `inspect.py`.
- `vlm/transform/commands/`: `caption.py`, `ask.py`, `vqa.py`, `map_answers.py` ([transforms](transforms.md)).
- `vlm/output/commands/`: `to_json.py`, `to_shards.py`, `to_vqa_style.py`, `inspect.py` ([output](output.md)).

## Contracts

Usage: `cvsuite vlm <src> [ingest opts] [<transform> <transform args>] <command> [args]`.

Base arguments: positional `src`; `--from {auto,images,json,shards,vqa-style}`; `-h/--help`.

Transforms: `caption`, `ask`, `map-answers`, `vqa`. Outputs: `to-json`, `to-shards`, `to-vqa-style`, `inspect`.

Format mapping: `images` to `flat`, `json` to `flat_vlm_json`, `shards` to `shards_vlm`, `vqa-style` to `vqa_style`.

Help forms, all also working with a leading `<src>` (for example `cvsuite vlm ./ds caption --help`): `cvsuite vlm --help`, `cvsuite vlm --provider <provider> -h`, `cvsuite vlm <transform> --help`, `cvsuite vlm <output> --help`. When transform help is requested and a provider or model token is already present, the CLI appends that provider's catalog to the parser epilog; provider and model names are normalized before lookup.

Execution: `vlm.core.router.ingest(Path(src), from_hint=...)`, transform `run(dataset, parsed_transform)`, output `run(dataset, parsed_output)`, and `io.cleanup_dataset_resources(dataset)` always in `finally`.

Uses `build_parser`, `parse_help` and `split_rest` from [pipeline](../pipeline.md#helpers-in-cvsuitebranch_cli); `apply_transform_output_hints` is not used.

### Format detection precedence

- Directory: flat JSON, then shards, then VQA-style, then images.
- File: shards, then VQA-style, then images.

## Constraints

- If a directory matches several structured VLM formats, ingest MUST fail instead of choosing.
- Temporary resources from shard extraction MUST be tracked in dataset metadata and cleaned even on errors or early exit.
- Help MUST stay deterministic and offline; catalog text comes from the shared provider registry ([fm_provider](../common/fm_provider.md)).
- At most one transform MAY appear before the output command, because `vlm` limits its chain to one ([pipeline](../pipeline.md#chaining)).
- Missing source MUST return exit code `1`.
- Older source spellings (`json`, `shards`, `vqa-style`) and some older metadata values are preserved, although canonical adapter ids use underscore names.

## History

- [Branch per domain](../logs/2026-09-20_branch-per-domain.md).
