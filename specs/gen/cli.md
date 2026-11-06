# Gen CLI

Related: **Read first** [pipeline](../pipeline.md). **See also** [gen transforms](transforms.md), [gen output](output.md), [fm_runtime](../common/fm_runtime.md).

## Overview

`cvsuite gen` generates images from text prompts (`create`) or edits existing images by prompt (`edit`). It follows the [pipeline](../pipeline.md) shape with one difference: the first stage is an *action* that builds the dataset (it needs no source for `create`), followed by an output command that writes the images.

Example: `cvsuite gen create --provider flux --prompt "a red bicycle" --num-images 4 to-dst ./out`.

## Language

- **Action**: the first stage, `create` or `edit`. It plays the role of a transform.

## Map

- `gen/cli.py`: parsing, local `_split_output`.
- `gen/transform/commands/create.py`, `edit.py` ([transforms](transforms.md)).
- `gen/output/commands/to_dst.py` ([output](output.md)).

## Contracts

Usage: `cvsuite gen <action> [action args] <output-command> [args]`. Actions: `create`, `edit`. Outputs: `to-dst`.

Parsing: the first token is an action, or an output command when asking for help. The action parser consumes known args and leaves the first remaining token as the output command. Exactly one action and one output are allowed.

Help: `cvsuite gen --help` prints action and output usage; `cvsuite gen create --help` and `edit --help` print action help with provider catalogs; `cvsuite gen to-dst --help` prints output help. If help appears after the output token the output help is shown, otherwise the action help.

Execution: resolve the action module from `argv[0]`, parse action args, parse the output command and args, run the action's `run(parsed_action)` to get a `VisionDataset`, then run the output's `run(dataset, parsed_output)`.

Registries are discovered from `gen/transform/commands/*.py` and `gen/output/commands/*.py`. Uses `build_parser` and `parse_help` from [pipeline](../pipeline.md#helpers-in-cvsuitebranch_cli).

## Constraints

- The branch MUST NOT use argparse subparsers, so prompts and later tokens cannot interfere with action/output splitting.
- A prompt string equal to an output command name MUST remain a valid prompt, because actions are parsed before the output is finalized.
- `_split_output` MUST stay local to `gen/cli.py`: the two-stage split differs from the three-stage `split_rest`. `split_rest` MUST NOT be used here.
- An output command MUST be present. When missing, usage is printed with "Missing output command."; when unrecognized, the message names the token.
- `edit` MUST reuse label-style ingest for structured sources; `create` MUST NOT require a source.

## History

- [Branch per domain](../logs/2026-09-20_branch-per-domain.md): why `gen` has its own parser and formats.
