# Gen Output

Related: **Read first** [gen cli](cli.md). **See also** [gen transforms](transforms.md), [stats](../common/stats.md).

## Overview

`to-dst` is the only generation output. It moves provider-generated images into their final destination, decides whether the destination is a file or a directory, applies overwrite and naming rules, exports the original source images in edit mode, and optionally writes `stats.yaml`.

## Language

- **Output key** (`gen_output_key`): stable key that names a generated image.
- **Source key** (`gen_source_key`): identity of an edit source image, used to deduplicate original exports.

## Map

- `gen/output/commands/to_dst.py`.

## Contracts

`cvsuite gen ... to-dst <dst> [options]`. Arguments: `dst`, `--overwrite`, `--no-original`, `--link-mode {copy,hard,sym}` (how original source images are placed, [pipeline](../pipeline.md#link-mode)), `--with-stats`.

Destination interpretation:

| dataset | destination | result |
|---|---|---|
| single record | existing directory, or path without suffix | timestamped PNG inside that directory |
| single record | path with a suffix | exactly that file |
| multiple records | path with a suffix | suffix stripped, treated as a directory |
| multiple records | directory | files named `<gen_output_key>.png` |

Reads generation provenance attributes `gen_output_key`, `gen_source_image`, `gen_source_key`. Moves generated images out of the provider output area.

Stats (`build_gen_stats`): generation mode, prompt count, fanout mode, image size, source summary when available, primary artifact paths.

## Constraints

- Existing files MUST be replaced only with `--overwrite`.
- Edit mode MUST export original source images unless `--no-original` is set. Exports MUST be deduplicated by `gen_source_key` and placed with `--link-mode`, which MUST NOT fall back to another mode.
- Output filenames MUST come from stable prompt or sample keys, never from prompt text.

## History

- [Chained transforms and split as a transform](../logs/2026-11-05_chained-transforms-and-split.md): why originals are placed with `--link-mode`.
