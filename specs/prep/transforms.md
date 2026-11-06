# Prep Transforms

Related: **Read first** [prep cli](cli.md). **See also** [prep output](output.md), [fm_runtime](../common/fm_runtime.md).

## Overview

The four `prep` commands do the real work of the branch: `arrange` normalizes folder structure, `process` normalizes pixels, `orient` corrects rotation, and `sample` pools images into one flat subset. They operate directly on filesystem trees.

## Language

- **Flatten**: collapsing folders into one level; `plain` keeps basenames, `enc-prefix` and `enc-suffix` encode the parent path into the name.
- **Non-image policy**: what to do with files that are not images: `keep`, `skip`, `collect`.
- **Reserved names**: `_non_images`, `_video_frames`, `manifest.yaml` (`prep/core/fs.py`).

## Map

- Configs: `prep/core/config.py` (`ArrangeConfig`, `ProcessConfig`, `OrientConfig`, `SampleConfig`).
- Entry points: `run_arrange_core`, `run_process_core`, `run_orient_core`, `run_sample_core`.
- `prep/core/actions.py`, `arrange/ops.py`, `process/*`, `orient/deep_orientation.py` (primary fallback), `orient/clip.py` (auxiliary scorer helper).

## Contracts

### arrange

`src`, `dst`, `--unzip`, `--flatten [plain|enc-prefix|enc-suffix]` (no value means `plain`, absent means no flattening), `--exact-dedup`, `--perceptual-dedup [THRESHOLD]`, `--rename-seq`, `--dst-subdir NAME`, `--link-mode {copy,move,hard}`, `--non-image {skip,collect,keep}`, `--dry-run`, `--no-manifest`, `-v/--verbose`.

- Exact dedup: first seen wins by `(size, sha1)`; also skips incoming files already under `dst`.
- Perceptual dedup: first seen wins by pHash Hamming distance `<= THRESHOLD` (default 5; lower is stricter); images only; also skips incoming images near any image already under `dst`. Non-image files pass through under the `--non-image` policy.
- `--rename-seq` names images `000001.ext`, starting after the largest sequence id under `dst`, and keeps relative directories unless `--flatten` is set.
- `--dst-subdir NAME` writes kept files under `dst/NAME/`.
- Non-image policy: `keep` leaves the relative layout, `skip` omits, `collect` routes under `_non_images/`.

### process

`src`, `dst`, `--orient`, `--size`, `--resize-mode {long,short,square,cap-long,cap-short}`, `--square-mode {pad,crop,distort}`, `--grayscale`, `--to-format`, `--backend {opencv,pillow}`, `--jpeg-quality 0..100`, `--video-sample-mode {skip,collect,keep}`, `--video-sample-k`, `--video-sample-min-gap`, `--transfer {copy,move}`, `--non-image {skip,collect,keep}`, `--dry-run`, `--no-manifest`, `-v/--verbose`.

- Image operation order: EXIF transpose, resize, square pad/crop/distort, grayscale, format conversion.
- Default backend OpenCV (area interpolation when shrinking, cubic when enlarging); Pillow on request. JPEG quality defaults to 95 and is recorded with the backend in the manifest.
- Video frames go to `_video_frames/` (`collect`) or beside the source path under `<stem>_frames/` (`keep`).

### orient

`src`, `dst`, `--try-hardlink`, `--batch`, `--device`, `--precision`, `--weights`, `--no-resume`, `--dry-run`, `-v/--verbose`. There is no prompt field.

- EXIF orientation is preferred; otherwise the `deep_orientation` provider scores the image. It builds a temporary shared dataset with classify-family metadata only for this step.
- Class-to-angle mapping: `correct` gives `0`, `rotate_90_clockwise` gives `270`, `rotate_180` gives `180`, `rotate_90_counter_clockwise` gives `90`.

### sample

One or more `srcs`, `dst`, `--count` or `--frac`, `--seed` (default `0`), `--link-mode {copy,hard}` (default `copy`), `--dry-run`, `-v/--verbose`. All sources are pooled before sampling and the output is flat. Selection follows [nested sampling](../common/sampling.md): a file's key is its source index and its path relative to that source, so a smaller `--frac` selects a subset of what a larger one selects. `prep` keeps its own link modes (`copy`, `move`, `hard`, with `sample` taking `copy` and `hard`), because it works on files and has no `sym`.

## Constraints

- `arrange`, `process` and `sample` MUST fail fast when the source is missing, is not a directory, or holds no images (or no videos, for `process`).
- `arrange`: `--dry-run` MUST imply verbose. `--exact-dedup` and `--perceptual-dedup` MUST be mutually exclusive.
- `process`: `--resize-mode` MUST require `--size`; `--size` without a mode MUST default to `long`; `--square-mode` MUST be valid only with `--resize-mode=square`; video mode `skip` MUST clear all video-sampling numeric parameters.
- `orient`: `--batch` MUST be at least 1. `--device` MUST default to `cpu` (the `deep_orientation` venv is CPU-only) and `auto` MUST become `cpu`. The destination MUST be outside the source tree. `--dry-run` MUST only report planned actions.
- `sample` MUST require exactly one of `--count` or `--frac`, MUST reject negative counts, MUST reject annotated datasets (use `cvsuite label sample`), and MUST reject nested directory trees (use `cvsuite class sample`) while ignoring `_non_images/`, `_video_frames/` and `manifest.yaml` so an `arrange` or `process` output can be sampled directly. Each source MUST be a flat image folder with at least one image.
- Long-running `arrange` and `process` loops MUST show tqdm progress bars.
- Prep-artefact names MUST NOT be used for user files.

## History

- [What belongs in cvsuite](../logs/2026-11-05_what-belongs-in-cvsuite.md): why every sampler shares one nested selection rule.
