# Prep Transforms

## 1. General Description
Prep commands are effectively the branch’s transform layer because they perform the actual content and layout mutations. They operate directly on filesystem trees rather than on label-style dataset exports. The four commands cover structure normalization, pixel normalization, orientation correction, and pooled flat-folder sampling.

## 2. Inputs and Outputs (I/O)
`arrange` inputs:
- `src`, `dst`
- `--unzip`
- `--flatten [plain|enc-prefix|enc-suffix]`
- `--exact-dedup`
- `--perceptual-dedup [THRESHOLD]`
- `--rename-seq`
- `--dst-subdir NAME`
- `--link-mode {copy,move,hard}`
- `--non-image {skip,collect,keep}`
- `--dry-run`
- `--no-manifest`
- `-v/--verbose`

`process` inputs:
- `src`, `dst`
- `--orient`
- `--size`
- `--resize-mode {long,short,square,cap-long,cap-short}`
- `--square-mode {pad,crop,distort}`
- `--grayscale`
- `--to-format`
- `--backend {opencv,pillow}`
- `--jpeg-quality 0..100`
- `--video-sample-mode {skip,collect,keep}`
- `--video-sample-k`
- `--video-sample-min-gap`
- `--transfer {copy,move}`
- `--non-image {skip,collect,keep}`
- `--dry-run`
- `--no-manifest`
- `-v/--verbose`

`orient` inputs:
- `src`, `dst`
- `--try-hardlink`
- `--batch`
- `--device`
- `--precision`
- `--weights`
- `--no-resume`
- `--dry-run`
- `-v/--verbose`

`sample` inputs:
- one or more `srcs`
- `dst`
- `--count` or `--frac`
- `--seed`
- `--hardlink`
- `--dry-run`
- `-v/--verbose`

## 3. Interfaces
Dataclass configs in `prep/core/config.py`:
- `ArrangeConfig`
- `ProcessConfig`
- `OrientConfig`
- `SampleConfig`

Core command interfaces:
- `run_arrange_core(config)`
- `run_process_core(config)`
- `run_orient_core(config)`
- `run_sample_core(config)`

Support interfaces:
- file-action planning in `prep/core/actions.py`
- arrange file-entry operations in `prep/core/arrange/ops.py`
- image/video process ops in `prep/core/process/*`
- primary orientation fallback in `prep/core/orient/deep_orientation.py`
- auxiliary CLIP upright scorer helper in `prep/core/orient/clip.py`

## 4. Business Decisions and Strict Policies
`arrange`:
- `--flatten` with no value means `plain`
- absent `--flatten` means no flattening
- `--dry-run` implies verbose output
- `--exact-dedup` and `--perceptual-dedup` are mutually exclusive
- `--exact-dedup` is first-seen-wins using `(size, sha1)` identity, and also skips incoming files whose `(size, sha1)` already exists under `dst`
- `--perceptual-dedup [THRESHOLD]` is first-seen-wins using pHash Hamming distance `<= THRESHOLD` (default 5, lower is stricter), applies to images only, and also skips incoming images within threshold of any image already under `dst`
- in perceptual mode non-image files pass through untouched (only `--non-image` policy applies)
- `--rename-seq` starts after the largest existing sequence id under `dst`
- `--rename-seq` preserves relative directories unless `--flatten` is set
- `--dst-subdir NAME` writes kept current-source files under `dst/NAME/`
- long-running arrange scans and hashing steps emit tqdm progress bars
- non-image policy:
  - `keep` => leave in-place relative layout
  - `skip` => omit
  - `collect` => route under `_non_images/`

`process`:
- `--resize-mode` requires `--size`
- `--size` without `--resize-mode` defaults resize mode to `long`
- `--square-mode` is only valid with `--resize-mode=square`
- OpenCV is the default image backend; Pillow remains available explicitly
- OpenCV uses area interpolation when shrinking and cubic interpolation when enlarging
- JPEG quality defaults to 95 and is recorded with the backend in the process manifest
- video sampling mode `skip` clears all video-sampling numeric parameters
- non-image policy mirrors arrange behavior
- source-file and sampled-frame processing loops emit tqdm progress bars

`orient`:
- `OrientConfig` does not contain a prompt field; current runtime inputs are source/destination, hardlink preference, batch, device, precision, optional weights, verbosity, dry-run, and `no_resume`
- `--batch` must be at least 1
- `--device` defaults to `cpu` (the `deep_orientation` venv is CPU-only); `auto` is normalized to `cpu`
- EXIF orientation is preferred over model fallback
- destination must live outside the source tree
- unchanged files may be hardlinked when requested
- dry-run only reports planned actions
- the main model fallback path is `deep_orientation`; the older CLIP scorer helper still exists in code but is not the primary orientation workflow

`sample`:
- requires exactly one of `--count` or `--frac`
- rejects negative count
- rejects annotated datasets and points the user to `cvsuite label sample`
- rejects nested directory trees and points the user to `cvsuite class sample`, but ignores the prep artefacts `_non_images/`, `_video_frames/`, and `manifest.yaml` so an `arrange`/`process` output can be sampled directly
- each source must be a flat raw-image folder containing at least one image
- all sources are pooled into one sampling set and the output is always flat

`arrange` / `process` / `sample` all fail fast when the source directory is missing, is not a directory, or contains no images (or videos, for `process`).

The names `_non_images`, `_video_frames`, and `manifest.yaml` are reserved prep-artefact names (`prep/core/fs.py`).

## 5. Implementation Details
Important command behaviors:
- `arrange`
  - flatten modes:
    - `plain`: basename only
    - `enc-prefix`: encode all parent parts before filename
    - `enc-suffix`: append encoded parent path after stem
  - `rename_seq` renames images like `000001.ext`
  - successful non-dry-run `arrange` and `process` runs write `manifest.yaml` unless `--no-manifest` is set
- `process`
  - image op order:
    1. EXIF transpose
    2. resize
    3. square pad/crop/distort
    4. grayscale
    5. format conversion
  - video frame sampling can route to `_video_frames/` or sibling `<stem>_frames/`
- `orient`
  - builds a temporary shared dataset only for the fallback FM scoring step
  - uses provider `deep_orientation` with classify-family runtime metadata
  - predicted class mapping is:
    - `correct` => `0`
    - `rotate_90_clockwise` => `270`
    - `rotate_180` => `180`
    - `rotate_90_counter_clockwise` => `90`
- `sample`
  - pools all candidate images across all sources before sampling

Related docs:
- `[[prep/cli]]`
- `[[prep/output]]`
- `[[common/fm_runtime]]`
