# Prep Output

## 1. General Description
Prep commands do not have separate output subcommands, so “output” in this branch means the exact filesystem side effects each command creates. The branch writes normalized raw-image trees, processed images, sampled flat folders, or orientation-corrected mirror trees. This document focuses on those destination layouts and file-writing rules.

## 2. Inputs and Outputs (I/O)
Outputs by command:
- `arrange`
  - destination tree under `dst`
  - optional `_non_images/` collection area
  - optional `manifest.yaml` run sidecar
- `process`
  - destination tree under `dst`
  - optional `_video_frames/` or temp `_video_frames_tmp/` areas during execution
  - optional `manifest.yaml` run sidecar
- `orient`
  - destination mirror tree under `dst`
  - copied, hardlinked, or rewritten images
- `sample`
  - flat destination folder under `dst`

## 3. Interfaces
File action interfaces:
- `prep/core/actions.py` materializes copy, move, or hardlink plans
- dry-run mode logs planned actions without touching the filesystem
- `arrange` and `process` write `manifest.yaml` by default on successful non-dry-run runs
- `arrange` manifests include a `stats.skipped_duplicates` count of files dropped by dedup

Run manifest schema (`manifest.yaml`, written by `prep/core/manifest.py`):
- `version`: schema version integer (currently `1`)
- `branch`: always `prep`
- `command`: the originating command (`arrange` or `process`)
- `src`: source path string
- `dst`: destination path string
- `options`: the command config fields as a mapping, excluding `src`, `dst`, and `manifest`; `Path` values are stringified
- `stats`: optional mapping of run statistics, present only when the command supplies stats (e.g. `arrange` dedup counts)

The manifest is written under `dst/<dst_subdir>/` when `--dst-subdir` is set, otherwise directly under `dst/`. It is suppressed on `--dry-run` and by `--no-manifest`.

Destination-shape interfaces:
- arrange preserves or rewrites relative layout depending on flatten mode
- process preserves source-relative structure for images and many non-image policies
- orient preserves source-relative structure beneath the destination root
- sample always flattens pooled results into one folder

## 4. Business Decisions and Strict Policies
- File-collision handling in flat-output contexts uses suffixing such as `_1`, `_2`, and so on.
- `process --transfer move` deletes the source image after successful output write.
- `process` video sampling with OpenCV unavailable logs an error and yields no frames rather than failing the whole command.
- `orient` rewritten images preserve as much metadata as PIL writing allows:
  - normalized EXIF orientation `1`
  - ICC profile
  - dpi
  - xmp
  - JPEG quality/subsampling `"keep"` when possible
- `orient` copies stat metadata from the source file after rewrite.
- `sample --hardlink` creates inode-sharing outputs when the filesystem allows it.

## 5. Implementation Details
Command-specific output details:
- `arrange`
  - `link_mode=copy` => copy pairs
  - `link_mode=move` => move pairs
  - `link_mode=hard` => hardlink pairs
  - `dst_subdir=<name>` prefixes kept outputs with `<name>/`, and the run `manifest.yaml` is written under `dst/<name>/` alongside those outputs
  - incremental exact dedup scans existing `dst` files by `(size, sha1)` and skips incoming matches
  - incremental perceptual dedup scans existing `dst` images by pHash and skips incoming images within the threshold Hamming distance
- `process`
  - `square pad` centers the image on a blank square canvas
  - `cap-long` and `cap-short` only downscale when the source exceeds the size threshold
  - sampled video frames in `collect` mode go under `_video_frames/...`
  - sampled video frames in `keep` mode go beside the source-relative path under `<stem>_frames/`
- `orient`
  - unchanged images may remain byte-identical via hardlink/copy
  - rotated images are rewritten upright
- `sample`
  - pooled outputs ignore source-folder structure and write all sampled images flat into `dst`

Manual verification harness:
- `src/cvsuite/prep/tests/manual_flow.py` builds a small mixed-format sample dataset, runs the representative prep commands, pauses for inspection, and then cleans up unless interrupted
- it is the manual-flow companion for the behaviors documented in these prep specs

Related docs:
- `[[prep/cli]]`
- `[[prep/transforms]]`
