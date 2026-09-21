# Common CLI

Related: **Read first** [architecture](../architecture.md). **See also** [fm_cache](fm_cache.md).

## Overview

`cvsuite common` is the branch for shared operations. Its public surface is management of the model cache: where it is, what it holds, and how to purge it. It is not a dataset pipeline, so it uses generic command discovery from `cvsuite.branch_cli` instead of the [pipeline](../pipeline.md) shape.

## Map

- `src/cvsuite/common/cli.py`: calls `run_generic_branch(...)`.
- `src/cvsuite/common/cache/commands/`: the command modules.
- Cache behavior: [fm_cache](fm_cache.md).

## Contracts

Usage: `cvsuite common <command> [args]`.

Discovery: `common/**/commands/*.py`; path parts and module names are hyphenated into command names; each module has an optional `attach(parser)` and a required `run(args)`.

Commands:

- `cache-path`: `--json`.
- `cache-list`: provider filter, `--json`, `--sort`, `--bytes`.
- `cache-purge`: explicit provider targets or `--all`; `--part all|venv|weights|pkgs|hub`; `--dry-run`; `--yes`.

Output is human-readable text, JSON for cache listings when requested, and deletions or a dry-run summary for purge.

## Constraints

- Discovery MUST skip `venv` and `__pycache__`.
- `result_to_exit_code()` MUST return `0` for `None` or a `VisionDataset` result.
- A missing or unknown requested cache provider MUST raise an explicit error.
- `--part hub` MUST reject provider names.

Command semantics such as confirmation and purge scope are in [fm_cache](fm_cache.md#constraints).
