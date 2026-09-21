# Prep Output

Related: **Read first** [prep cli](cli.md). **See also** [prep transforms](transforms.md).

## Overview

`prep` has no output subcommands. Its output is the set of filesystem effects each command produces: normalized raw-image trees, processed images, flat sampled folders, and orientation-corrected mirror trees. This spec covers the destination layouts and file-writing rules.

## Language

- **Manifest**: the `manifest.yaml` run sidecar written by `arrange` and `process`.
- **Link mode**: copy, move or hardlink.

## Map

- `prep/core/actions.py`: plans and applies copy, move or hardlink.
- `prep/core/manifest.py`: manifest writing.
- `prep/tests/manual_flow.py`: manual verification harness that builds a small mixed sample, runs the commands, pauses for inspection and cleans up.

## Contracts

### Destination shapes

| command | destination |
|---|---|
| `arrange` | tree under `dst`; optional `_non_images/`; optional `manifest.yaml`. Preserves or rewrites relative layout depending on flatten mode. |
| `process` | tree under `dst` keeping source-relative structure; optional `_video_frames/` (temporary `_video_frames_tmp/`); optional `manifest.yaml`. |
| `orient` | mirror tree under `dst` with copied, hardlinked or rewritten images. |
| `sample` | flat folder under `dst`; source-folder structure is ignored. |

`arrange` link mode: `copy` copies, `move` moves, `hard` hardlinks. `--dst-subdir <name>` prefixes kept outputs with `<name>/` and places the manifest under `dst/<name>/`. Incremental dedup scans existing `dst` files (`(size, sha1)` for exact, pHash for perceptual). `process` square pad centers the image on a blank square canvas; `cap-long` and `cap-short` only downscale.

### Manifest (`manifest.yaml`)

- `version`: schema version integer (`1`).
- `branch`: always `prep`.
- `command`: `arrange` or `process`.
- `src`, `dst`: path strings.
- `options`: the command config fields excluding `src`, `dst` and `manifest`, with `Path` values stringified.
- `stats`: optional; present only when the command supplies statistics (for example `arrange` puts `skipped_duplicates` here).

It is written under `dst/<dst_subdir>/` when `--dst-subdir` is set, else under `dst/`, on successful non-dry-run runs, unless `--no-manifest`.

## Constraints

- Dry run MUST log planned actions and MUST NOT touch the filesystem.
- Filename collisions in flat output MUST be resolved with `_1`, `_2`, and so on.
- `process --transfer move` MUST delete the source image only after the output write succeeds.
- If OpenCV is unavailable, video sampling in `process` MUST log an error and yield no frames instead of failing the command.
- `orient` rewrites MUST preserve, as far as PIL allows: EXIF orientation normalized to `1`, ICC profile, dpi, xmp, and JPEG quality and subsampling `"keep"`. Stat metadata MUST be copied from the source after rewrite. Unchanged images MAY remain byte-identical via hardlink or copy.
- `sample --hardlink` MUST create inode-sharing outputs when the filesystem allows it.
