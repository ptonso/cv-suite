# CLI

## 1. General Description
The top-level CLI is intentionally thin. Its job is only to discover public branches, print usage, normalize the requested branch token, and import that branch’s `main()` function. Branch-specific routing, help behavior, model-catalog behavior, and pipeline semantics live below this layer in the branch packages described throughout `[[architecture]]`.

## 2. Inputs and Outputs (I/O)
Inputs:
- `cvsuite`
- `cvsuite -h` / `cvsuite --help`
- `cvsuite <branch> [args]`

Outputs:
- A branch-specific exit code.
- Top-level usage text when no branch or help is requested.
- `SystemExit("Unknown branch: ...")` for unsupported branch tokens.

## 3. Interfaces
Public CLI behavior:
- `usage: cvsuite <branch> [args]`
- Public branches are discovered dynamically by scanning `src/cvsuite/*/cli.py`.
- Current discovered public branches are:
  - `class`
  - `common`
  - `gen`
  - `label`
  - `prep`
  - `vlm`
- Internal branch names are converted to public names through:
  - explicit alias `classify -> class`,
  - otherwise underscore-to-hyphen conversion.

Branch dispatch behavior:
- The requested branch token is normalized by replacing underscores with hyphens before lookup.
- Public name is converted back to internal package name using alias reversal or hyphen-to-underscore replacement.
- The branch module is imported as `cvsuite.<branch>.cli`.

## 4. Business Decisions and Strict Policies
- Only packages with a `cli.py` file are exposed as public branches.
- `classify` is intentionally not a public branch name. The only valid public token is `class`.
- Top-level help does not import all branch modules eagerly; it only scans the filesystem.
- Unknown branch handling prints usage first, then raises `SystemExit`.
- Offline branch help is preferred. Model/provider help surfaces exposed by branches are expected to read deterministic source metadata through the shared FM registry rather than performing live catalog lookups during `--help`.

## 5. Implementation Details
The dispatch stack is:
1. `cvsuite.cli._discover_branches()` scans the package directory.
2. `_public_branch_name()` applies alias or underscore-to-hyphen conversion.
3. `main(argv)` validates the first token and imports the branch CLI dynamically.

Two branch-dispatch patterns are then used below the top level:
- Generic command-subparser branches via `cvsuite.branch_cli` for `common`.
- Prep-specific recursive subcommand loading for `prep`.
- Custom pipeline parsers for `classify`, `label`, `vlm`, and `gen`.

Pipeline branch exit-code contract:
- `--help` or recognized `<command> --help` → `0`.
- Missing source argument (no `<src>` or no `<src>` tokens) → `1`.
- Unknown branch or unknown command token → `SystemExit` (non-zero, message includes the bad token).

Shared pipeline parser infrastructure (`cvsuite.branch_cli`):
- `build_parser`, `parse_help`, `split_rest`, and `apply_transform_output_hints` are defined once in `branch_cli` and imported by all pipeline branches.
- See `[[architecture]]` § "Shared pipeline CLI helpers" for the contracts of each function.

See also:
- `[[architecture]]`
- `[[common/cli]]`
- `[[classify/cli]]`
- `[[label/cli]]`
- `[[vlm/cli]]`
- `[[gen/cli]]`
- `[[prep/cli]]`
