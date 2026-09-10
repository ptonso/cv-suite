# Common CLI

## 1. General Description
The `common` branch is the shared-operations branch exposed as `cvsuite common`. Unlike the custom pipeline branches, it uses the generic command-discovery mechanism from `cvsuite.branch_cli`. Today its public surface is cache management for FM providers rather than dataset ingestion or export.

## 2. Inputs and Outputs (I/O)
Canonical usage:
- `cvsuite common <command> [args]`

Currently discovered commands:
- `cache-path`
- `cache-list`
- `cache-purge`

Outputs:
- human-readable console output
- JSON output for cache listing when requested
- cache deletions or dry-run summaries for purge

## 3. Interfaces
Discovery interface:
- scans `common/**/commands/*.py`
- hyphenates path parts and module names into CLI command names
- imports optional `attach(parser)` and required `run(args)` entry points

Current command interfaces:
- `cache-path`
  - `--json`
- `cache-list`
  - provider filter
  - `--json`
  - `--sort`
  - `--bytes`
- `cache-purge`
  - explicit provider targets or `--all`
  - `--part all|venv|weights|pkgs|hub`
  - `--dry-run`
  - `--yes`

## 4. Business Decisions and Strict Policies
- Generic discovery skips `venv` and `__pycache__` directories.
- `result_to_exit_code()` returns `0` for `None` or `VisionDataset` returns, though common commands currently return plain CLI-oriented results.
- Missing or unknown requested cache providers raise explicit errors instead of being ignored.
- `--part hub` addresses the shared Hugging Face hub, which belongs to no single provider, so it rejects provider names.

## 5. Implementation Details
The `common` branch is implemented in `src/cvsuite/common/cli.py` by calling `run_generic_branch(...)`. That helper:
1. discovers command specs,
2. builds argparse subparsers,
3. attaches command-specific arguments,
4. dispatches to the selected `run(args)` function.

Related docs:
- `[[common/fm_cache]]`
- `[[architecture]]`
- `[[cli]]`
