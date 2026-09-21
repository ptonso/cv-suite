# Prep CLI

Related: **Read first** [architecture](../architecture.md). **See also** [prep transforms](transforms.md), [prep output](output.md).

## Overview

`cvsuite prep` prepares raw files before any dataset format applies. Picture a messy folder of zip files, loose images and videos: `prep` arranges, resizes, orients and samples it so that the other branches can then read it.

It is structurally different from the [pipeline](../pipeline.md) branches. It uses classic subcommands, works on files and folders, and has no transform-then-output chaining. The shared dataset is used only briefly, for the model-based orientation fallback. The reasons are in [the branch note](../logs/2026-09-20_branch-per-domain.md).

## Map

- `prep/cli.py`: discovery and parsing.
- `prep/commands/`: `arrange.py`, `orient.py`, `process.py`, `sample.py`.
- `prep/core/`: implementations ([transforms](transforms.md), [output](output.md)).

## Contracts

Usage: `cvsuite prep <command> [args]`.

- `arrange <src> <dst> ...`
- `process <src> <dst> ...`
- `orient <src> <dst> ...`
- `sample <srcs...> <dst> ...`

Discovery: `prep/cli.py` scans `commands/` directories recursively. A module may expose `register_subcommand(subparsers)`; its parser MUST set `func`. The CLI builds a root `ArgumentParser`, lets each module register, parses argv, calls `args.func(args)`, and turns `None` into exit code `0`.

Results are filesystem changes or new destination trees, dry-run console reports, and exit code `0` on completion.

## Constraints

- Only modules discovered under `commands/` MUST be public. `venv` and `__pycache__` MUST be excluded.
- If a registered parser does not set `func`, the CLI MUST fail with "No command selected."
- `prep` MUST NOT support transform-plus-output chaining.
- `prep` MUST stay file-first. Its temporary use of the shared dataset for orientation MUST NOT turn it into a dataset-conversion branch.

## History

- [Branch per domain](../logs/2026-09-20_branch-per-domain.md): why `prep` works on files instead of the shared dataset.
