# Pipeline

Related: **Read first** [architecture](architecture.md). **See also** [classify cli](classify/cli.md), [label cli](label/cli.md), [vlm cli](vlm/cli.md), [gen cli](gen/cli.md).

## Overview

`class`, `label`, `vlm` and `gen` all run the same shape of command: read a source, apply at most one transformation, write an output.

```
cvsuite <branch> <src> [ingest opts] [<transform> <transform args>] <output command> [args]
```

A worked example: `cvsuite label ./raw sample --count 100 to-yolo ./out` reads `./raw`, keeps 100 records, and writes them as YOLO.

`gen` has no source for `create`, so its first stage is an *action* that builds the dataset, followed by the output command. `prep` and `common` do not follow this shape ([prep cli](prep/cli.md), [common cli](common/cli.md)).

Each branch keeps its own parser because its arguments differ: some transforms take free-form text, others prompt files or provider options, and each branch reads and writes different formats. The reasons are in [the branch-per-domain note](logs/2026-09-20_branch-per-domain.md). What is shared is only the splitting mechanics and the helpers below.

## Language

- **Stage**: one of source, transform, output.
- **Command registry**: the set of transform or output modules a branch discovers under `transform/commands/` (or `transform/<project>/commands/` for `label`) and `output/commands/`. Command names are module names with underscores turned into hyphens.

## Contracts

### Stage splitting

The branch first parses the source and ingest options. The remaining tokens are split into an optional transform (name plus its args) and a required output command (name plus its args). Only one transform is allowed. Splitting looks at recognized command names, not at argparse subparsers, so prompt strings that happen to equal a command name stay valid arguments elsewhere.

### Helpers in `cvsuite.branch_cli`

- `build_parser(module, prog)`: builds an `ArgumentParser` from a command module's `attach()` and docstring.
- `parse_help(parser, argv)`: triggers the parser's help path and returns `0`; re-raises non-help exits.
- `split_rest(rest, commands, transform_commands)`: returns `(transform_name, transform_args, command_name, command_args)`, or `(None, [], None, rest)` when no recognized output command is present.
- `apply_transform_output_hints(prog, transform_name, parsed_transform, command_name, parsed_command)`: see [hardlink hint](#hardlink-hint).

`class`, `label` and `vlm` use `split_rest`. `gen` uses `build_parser` and `parse_help` and keeps a local two-stage split.

### Command modules

- A transform module exposes `attach(parser)` and `run(dataset, args)`. An output module exposes `attach(parser)` and `run(dataset, args)`. `gen` actions expose `run(args)` and return the dataset.
- Ingest is `<branch>.core.router.ingest(...)` (or `classify.core.io.ingest`), which calls `cvsuite.common.io.read_dataset`.

### Help

`--help` is dispatched from the raw argument list to the most specific parser named before it, even when a source path comes first. `cvsuite label ./ds to-coco --help` prints the `to-coco` help. A `--provider <p> -h` form on `vlm` appends that provider's catalog ([vlm cli](vlm/cli.md)).

### Exit codes

- Help: `0`.
- Missing source: `1`.
- Unknown branch or command token: `SystemExit` with a message naming the token.
- Normal completion: `0`, including outputs that return a dataset (`result_to_exit_code`).

### Hardlink hint

If the transform is a sampling transform and the user passed `--hardlink`, `apply_transform_output_hints` copies that flag into the output command, but only when the output command exposes a `hardlink` argument.

## Constraints

- Only one transform MAY appear before the output command.
- A missing output command MUST print usage and exit with a message. An unrecognized token MUST be named in the message.
- A missing source MUST return exit code `1`.
- `--help` MUST stay offline.
- `label sample --hardlink` MUST only be combined with outputs that expose `hardlink`.
- Branch-specific split logic MAY stay local only where the semantics differ from `split_rest` (today: `gen`).

## History

- [Branch per domain](logs/2026-09-20_branch-per-domain.md): why each pipeline branch keeps its own parser.
