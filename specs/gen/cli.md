# Gen CLI

## 1. General Description
The generation branch is exposed as `cvsuite gen` and is organized as an action-plus-output pipeline rather than a source-plus-transform pipeline. The action stage constructs or expands a generation dataset, usually attaches `fm_request`, and the output stage materializes resulting images into a destination file or directory. The branch is specialized for text-to-image creation and image-edit workflows.

## 2. Inputs and Outputs (I/O)
Canonical usage:
- `cvsuite gen <action> [action args] <output-command> [args]`

Registered actions:
- `create`
- `edit`

Registered outputs:
- `to-dst`

Outputs:
- final image file or directory
- optional original-source exports for edit mode
- optional `stats.yaml`

## 3. Interfaces
CLI parsing semantics:
- The first token must be either an action or an output command used for help.
- The action parser consumes known args and leaves the first remaining token to be treated as the output command.
- Only one action stage and one output stage are allowed.

Help semantics:
- `cvsuite gen --help` prints action/output usage.
- `cvsuite gen create --help` and `cvsuite gen edit --help` show action help plus provider catalogs.
- `cvsuite gen to-dst --help` shows output help.
- Help-token location matters: if help appears after the output token, the branch shows output help; otherwise it shows action help.

## 4. Business Decisions and Strict Policies
- The branch is intentionally not built on argparse subparsers because prompts and downstream tokens may otherwise interfere with dynamic action/output splitting.
- Prompt strings that equal output command names are still valid prompt values because action parsing happens before output routing is finalized.
- `edit` reuses label-style ingest logic for structured annotated sources; `create` does not require a source dataset.
- Output-command presence is mandatory. Missing output commands print usage and exit with “Missing output command.” When the token is present but unrecognized, the message includes the bad token name.
- `_split_output` remains local to `gen/cli.py` because gen's two-stage (action + output) split semantics differ from the three-stage (source + transform + output) semantics of `split_rest` in `branch_cli`.

## 5. Implementation Details
Shared helpers from `cvsuite.branch_cli`:
- `build_parser`, `parse_help` — see `[[architecture]]` § "Shared pipeline CLI helpers".
- `split_rest` and `apply_transform_output_hints` are not used; gen uses its own `_split_output` instead.

Action and output registries are discovered dynamically from:
- `gen/transform/commands/*.py`
- `gen/output/commands/*.py`

Execution flow:
1. Resolve action module from argv[0].
2. Parse action args.
3. Parse the remaining output command and its args.
4. Run action module `run(parsed_action)` to produce a `VisionDataset`.
5. Run output module `run(dataset, parsed_output)`.

Related docs:
- `[[gen/transforms]]`
- `[[gen/output]]`
- `[[common/fm_runtime]]`
