# FM Utils

Related: **Read first** [fm_provider](fm_provider.md). **See also** [fm_runtime](fm_runtime.md), [fm_cache](fm_cache.md).

## Overview

`common/fm/core/utils.py` is the helper layer for provider bases and provider modules. It is not the runtime and not a branch adapter. It centralizes path resolution, prompt handling, precision normalization, Hugging Face source handling and small geometry helpers, so providers do not each reimplement them.

## Language

- **Hub dir** vs **weights dir**: two different roots. Hugging Face repos live in `hub_dir`; other per-provider artifacts live in `weights_dir` ([fm_cache](fm_cache.md)).
- **`<class>` token**: the placeholder replaced by each label in class-prompt templates.

## Map

- `common/fm/core/utils.py`: helpers listed below.
- `common/fm/core/prompt_utils.py`: class-template helpers and prompt-map builders.
- `common/fm/core/paths.py`: paths, `.env` discovery, token candidates, hub naming. Imports nothing heavy.

## Contracts

- Path resolution: resolve a record's image path from `record.image.path`, `dataset.root` and `FM_CALLER_CWD`.
- Prompts: class-template helpers and prompt-map builders.
- Precision: alias normalization (`float16`, `half`, `bfloat16`), autocast and quantization helpers.
- Hugging Face sources: `materialize_hf_model_source(model_id, *, hub_dir, stage_dir, ...)`, `materialize_hf_file(...)`, `ensure_hf_caches(cache_root, *, hub=None)` for cache redirection and token validation. Resolution order and staging are in [fm_cache](fm_cache.md#hugging-face-repo-resolution).
- Geometry and data: batching, mask-to-polygon and related conversions.

Provider bases call these helpers before opening any image. Registry and help code do not, since they stay source-only and offline.

## Constraints

- Providers MUST use shared path resolution and MUST NOT assume a branch filesystem layout.
- `.env` discovery MUST search upward from `FM_CALLER_CWD` and the current directory.
- An invalid Hugging Face token MUST warn and fall back to anonymous access when possible.
- Precision aliases MUST be normalized before provider code uses them.
- Heavy optional dependencies MUST be imported lazily; `cv2` only inside geometry helpers that need it.
- Hugging Face repos MUST be addressed by `hub_dir`, never `weights_dir`.
- `paths.py` MUST NOT import heavy libraries, so the CLI can resolve cache roots without loading torch.
- Class-prompt templates MUST use the shared `<class>` helpers.
- Tests: `common/tests/test_fm_prompt_utils.py`, `test_device_utils.py`, `test_fm_paths.py`.

## History

- [Central model cache](../logs/2026-09-20_central-model-cache.md): why Hugging Face repos are reused from a shared root.
