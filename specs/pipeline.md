# Pipeline

Related: **Read first** [architecture](architecture.md). **See also** [classify cli](classify/cli.md), [label cli](label/cli.md), [vlm cli](vlm/cli.md), [gen cli](gen/cli.md), [eda cli](eda/cli.md), [split](common/split.md).

## Overview

`class`, `label`, `vlm`, `gen` and `eda` all run the same shape of command: read a source, apply any number of transformations, write an output.

```
cvsuite <branch> <src> [ingest opts] [<transform> <transform args>]... <output command> [args]
```

A worked example: `cvsuite label ./raw select --min-box-px 20 --empty drop split --val-frac 0.2 to-yolo ./out` reads `./raw`, drops small boxes and the images they leave empty, assigns a validation split and writes YOLO. The transforms run left to right on one in-memory dataset, so there are no intermediate folders.

`gen` has no source for `create`, so its first stage is an *action* that builds the dataset, followed by the output command. `prep` and `common` do not follow this shape ([prep cli](prep/cli.md), [common cli](common/cli.md)).

Each branch keeps its own parser because its arguments differ: some transforms take free-form text, others prompt files or provider options, and each branch reads and writes different formats. The reasons are in [the branch-per-domain note](logs/2026-09-20_branch-per-domain.md). What is shared is the stage splitting, the chain runner and the helpers below.

## Language

- **Stage**: the source, a transform, or the output.
- **Command registry**: the set of transform or output modules a branch discovers under `transform/commands/` (or `transform/<project>/commands/` for `label`) and `output/commands/`. Command names are module names with underscores turned into hyphens.
- **Step**: one transform as it ran, recorded in `dataset.meta["steps"]` as `{"name", "args", "info"}`.
- **Model-backed transform**: a transform whose module declares `FM = True`: `ground`, `ocr`, `infer`.

## Contracts

### Chaining

The branch first parses the source and ingest options. The remaining tokens are a sequence of transform stages followed by one output stage.

- A transform stage starts at a token that names a registered transform. It ends at the next token that names a transform or the output command, provided the stage's own parser accepts exactly the tokens in between. A command name that an option consumes as its value stays an argument: in `ground --prompt sample to-yolo ./out` the parser rejects the stage `--prompt`, so the stage extends to `to-yolo`. A stage that no later token satisfies is an error that names the token where it started.
- The output stage starts at the token that ended the last transform stage. Everything after it belongs to the output command.
- Every stage is parsed before ingest, so a bad flag fails before any image is read. Two stages MAY use the same flag name, because each stage is parsed alone.
- Ingest runs, then each transform's `run(dataset, args)` in order, then the output's `run(dataset, args)`.

A model-backed transform serializes the dataset, runs a provider process and reloads a new dataset. A chain MAY hold at most one, and it MUST be the first transform.

After each transform, the chain runner appends a step to `dataset.meta["steps"]`. A transform reports facts about its run (a sample size, the shares a split reached) by setting `dataset.meta["step_info"]` to a dict on the dataset it returns. The runner moves it into the step's `info` and removes the key. `stats.yaml` carries the steps ([stats](common/stats.md)).

A transform that creates temporary files appends their folders to `dataset.meta["cleanup_dirs"]`. The branch removes every listed folder after the output command has run, whether or not the run succeeded.

`vlm` accepts at most one transform, and `gen` keeps its own two-stage split.

### Helpers in `cvsuite.branch_cli`

- `build_parser(module, prog)`: builds an `ArgumentParser` from a command module's `attach()` and docstring.
- `parse_help(parser, argv)`: triggers the parser's help path and returns `0`; re-raises non-help exits.
- `split_rest(rest, transforms, commands, *, prog="cvsuite", max_transforms=None)`: `transforms` maps a name to a callable returning its module, so only the named modules are imported. Returns the transform steps as `Step(name, args, module)` tuples, the output command name and its raw arguments, or `(steps, None, [])` when no output command is present. It enforces the chain rules above and `max_transforms`.
- `run_chain(dataset, steps, *, cleanup)`: runs the transforms in order, records the steps, and collects every folder listed in `cleanup_dirs` into the `cleanup` list. The branch removes that list in a `finally` (`remove_dirs`).

`class`, `label`, `vlm` and `eda` use `split_rest`. `gen` uses `build_parser` and `parse_help` and keeps a local two-stage split.

### Command modules

- A transform module exposes `attach(parser)` and `run(dataset, args)`, and MAY set `FM = True`. An output module exposes `attach(parser)` and `run(dataset, args)`. `gen` actions expose `run(args)` and return the dataset.
- Ingest is `<branch>.core.router.ingest(...)` (or `classify.core.io.ingest`), which calls `cvsuite.common.io.read_dataset`.

### Link mode

Every output that places source images in its destination takes `--link-mode {copy,hard,sym}`, default `copy`: `to-yolo`, `to-labelme`, `to-coco` (unless `--no-images`), `to-class-dir`, `to-json` and `to-vqa-style` (unless `--skip-images`), and `gen to-dst` for original images.

- `hard` links with `os.link`. A link that cannot be made raises and names the path and the reason.
- `sym` writes an absolute symbolic link to the resolved source image. It raises when the source lies under a folder listed in `meta["cleanup_dirs"]`, because the link would dangle.
- No mode falls back to another. `--overwrite` removes a claimed folder without following links ([output claims](common/io.md#output-claims)).

`prep` keeps its own link flags ([prep output](prep/output.md)).

### Help

`--help` is dispatched from the raw argument list to the most specific parser named before it, even when a source path comes first. `cvsuite label ./ds to-coco --help` prints the `to-coco` help. A `--provider <p> -h` form on `vlm` appends that provider's catalog ([vlm cli](vlm/cli.md)).

### Exit codes

- Help: `0`.
- Missing source: `1`.
- Unknown branch or command token: `SystemExit` with a message naming the token.
- Normal completion: `0`, including outputs that return a dataset (`result_to_exit_code`).

## Constraints

- A chain MAY hold any number of transforms before the output command, except in `vlm`, which holds at most one.
- A chain MUST hold at most one model-backed transform, and it MUST come first. This MUST be checked before ingest.
- A stage MUST be parsed alone, and a probe that fails MUST NOT print an argparse error or exit. Only the final failure is reported.
- A missing output command MUST print usage and exit with a message. An unrecognized token MUST be named in the message.
- A missing source MUST return exit code `1`.
- `--help` MUST stay offline.
- Temporary folders created by a transform MUST be removed after the output command ran, including when it failed.
- `dataset.meta["steps"]` MUST hold one entry per transform, in the order they ran.
- A link mode MUST NOT fall back to another mode. `--link-mode sym` MUST raise for an image under a cleanup folder.
- Branch-specific split logic MAY stay local only where the semantics differ from `split_rest` (today: `gen`).

## History

- [Branch per domain](logs/2026-09-20_branch-per-domain.md): why each pipeline branch keeps its own parser.
- [Chained transforms and split as a transform](logs/2026-11-05_chained-transforms-and-split.md): why a run takes several transforms, and why `--link-mode` replaces `--hardlink`.
