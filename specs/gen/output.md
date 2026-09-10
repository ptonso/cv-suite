# Gen Output

## 1. General Description
The generation branch currently has one output writer, `to-dst`, which moves provider-generated images into their final destination layout. It is responsible for file-vs-directory interpretation, overwrite rules, naming rules, original-source export in edit mode, and optional stats emission.

## 2. Inputs and Outputs (I/O)
Command:
- `cvsuite gen ... to-dst <dst> [options]`

Arguments:
- `dst`
- `--overwrite`
- `--no-original`
- `--with-stats`

Outputs:
- one image file for single-record outputs when the destination is a file path
- a directory of images for multi-record outputs or directory-like destinations
- optional original-source exports for edit mode
- optional `stats.yaml`

## 3. Interfaces
Writer interface:
- consumes generated image paths produced by providers
- renames or moves those image files into final destinations
- reads generation provenance attributes such as `gen_output_key`, `gen_source_image`, and `gen_source_key`

Destination interpretation:
- single-record dataset:
  - existing directory or suffix-less destination => timestamped PNG file inside that directory
  - explicit file path with suffix => that exact path
- multi-record dataset:
  - destination with suffix => suffix is stripped and the result is treated as a directory
  - filenames default to `<gen_output_key>.png`

## 4. Business Decisions and Strict Policies
- Existing files are only replaced when `--overwrite` is present.
- Edit mode exports original source images by default.
- Original-source exports are deduplicated by `gen_source_key`; repeated references to the same original image do not create duplicate copies.
- Original source export uses hardlinking where possible and falls back to copying semantics defined in the implementation.
- Output filenames are generated from stable prompt/sample keys rather than from prompt text.

## 5. Implementation Details
Implementation steps:
1. Determine whether destination is file-like or directory-like.
2. Move generated images out of the provider output area.
3. Name outputs using `gen_output_key` unless a single explicit file path was requested.
4. In edit mode, export original images unless `--no-original` is set.
5. Emit stats through `[[common/stats]]` when requested.

Stats content currently includes:
- generation mode
- prompt count
- fanout mode
- image size
- source summary when source metadata is available
- primary artifact path list

Related docs:
- `[[gen/cli]]`
- `[[gen/transforms]]`
- `[[common/stats]]`
