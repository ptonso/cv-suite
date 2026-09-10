# Prep CLI

## 1. General Description
The prep branch is exposed as `cvsuite prep` and is structurally different from the record-pipeline branches. It uses classic subcommands discovered from `commands/` folders and focuses on raw filesystem cleanup and image normalization. Most prep commands do not produce shared dataset outputs, although some internals temporarily use the shared FM runtime for fallback orientation scoring.

## 2. Inputs and Outputs (I/O)
Canonical shape:
- `cvsuite prep <command> [args]`

Discovered commands:
- `arrange`
- `orient`
- `process`
- `sample`

Outputs:
- filesystem mutations or newly created destination trees
- dry-run console reports
- exit code `0` for normal completion

## 3. Interfaces
CLI infrastructure:
- uses `branch_cli`-style discovery concepts but implemented locally in `prep/cli.py`
- recursively scans `commands/` directories beneath the prep package
- each module may expose `register_subcommand(subparsers)`
- registered parser must set `func`

Public command interfaces:
- `arrange <src> <dst> ...`
- `process <src> <dst> ...`
- `orient <src> <dst> ...`
- `sample <srcs...> <dst> ...`

## 4. Business Decisions and Strict Policies
- Unlike other branches, prep does not support transform-plus-output chaining.
- Only command modules discovered under `commands/` are public.
- `venv` and `__pycache__` directories are excluded from command discovery.
- If a registered parser does not set `func`, the CLI errors with “No command selected.”
- Prep is intentionally file-first. Temporary use of the shared dataset contract for orientation fallback does not make prep a generic dataset-conversion branch.

## 5. Implementation Details
Execution flow:
1. Build a root `ArgumentParser`.
2. Discover all prep command modules.
3. Let each module register its subparser and `func`.
4. Parse argv and call `args.func(args)`.
5. Convert `None` to exit code `0`.

Related docs:
- `[[prep/transforms]]`
- `[[prep/output]]`
- `[[architecture]]`
