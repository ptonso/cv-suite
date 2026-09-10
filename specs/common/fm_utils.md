# FM Utils

## 1. General Description
`common/fm/core/utils.py` is the shared helper layer used by provider bases and concrete provider modules. It is not the runtime orchestrator and it is not a branch adapter. Its job is to centralize reusable provider-side mechanics such as path resolution, prompt handling, precision normalization, Hugging Face cache behavior, and lightweight geometry helpers.

## 2. Inputs and Outputs (I/O)
Inputs:
- `Record.image.path`, `VisionDataset.root`, and `FM_CALLER_CWD` for path resolution.
- Requested device, precision, and model-id values from `FMRequest`.
- Environment variables and optional `.env` files for Hugging Face or API credentials.
- Model outputs such as masks or logits that need shared post-processing.

Outputs:
- Resolved absolute filesystem paths for provider image reads.
- Normalized precision/runtime decisions.
- Prompt maps and prompt expansions used by classification and VLM flows.
- Cache/source resolution hints for Hugging Face-backed providers.
- Shared geometry conversions such as mask-to-polygon extraction.

## 3. Interfaces
Important helper categories exposed from `common/fm/core/utils.py` and neighboring prompt helpers:
- path resolution
  - resolve a record image path from `record.image.path`, `dataset.root`, and caller cwd
- prompt handling
  - class-template helpers and prompt-map builders from `prompt_utils.py`
- precision/device handling
  - precision alias normalization
  - autocast and quantization helpers
- Hugging Face source handling
  - `materialize_hf_model_source(model_id, *, hub_dir, stage_dir, ...)` and `materialize_hf_file(...)`
  - local path resolution, share probing, staged download, snapshot promotion
  - `ensure_hf_caches(cache_root, *, hub=None)` for cache redirection and token validation
  - `.env` discovery, token candidate iteration, and hub-layout path naming now live in `common/fm/core/paths.py`
- geometry/data helpers
  - batching helpers
  - mask-to-polygon and related conversions

## 4. Business Decisions and Strict Policies
- Providers should use shared path resolution instead of branch-specific filesystem assumptions.
- `.env` discovery is upward-searching from `FM_CALLER_CWD` and current cwd so provider secrets can be loaded without hard-coding one project root.
- Invalid Hugging Face tokens warn and fall back to anonymous access when possible instead of crashing immediately.
- Precision aliases such as `float16`, `half`, and `bfloat16` are normalized before provider code uses them.
- Heavy optional dependencies are imported lazily where possible. Notably, `cv2` is intentionally imported only inside geometry helpers that actually need it.
- Hugging Face repos are addressed by `hub_dir`, never by `weights_dir`. The two are different roots: see `[[common/fm_cache]]`.
- Path and environment resolution lives in `common/fm/core/paths.py`, which imports nothing heavy so the CLI can resolve cache roots without pulling in torch.

## 5. Implementation Details
Shared helper usage across the FM layer:
- provider bases call utility functions to resolve real image paths before opening files
- registry/help code does not use these helpers because it stays source-only and offline
- Hugging Face-backed providers use cache helpers to reuse local snapshots and repo caches
- classification prompt templates rely on the shared `<class>` token helpers rather than reimplementing prompt expansion per provider

Cross-links:
- `[[common/fm_runtime]]`
- `[[common/fm_provider]]`
- `[[common/records_dataset]]`
