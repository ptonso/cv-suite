# FM Runtime

## 1. General Description
The shared FM runtime is the subsystem that turns a shared dataset plus `FMRequest` into a provider subprocess invocation. Its purpose is isolation and repeatability. Branches do not import provider implementations directly for actual inference. Instead, they construct a shared dataset, attach `dataset.fm_request`, and hand the dataset to `FMRunner`, which serializes it, resolves the provider module, prepares or validates the provider environment, and spawns a new Python process.

The dataset containers and the `FMRequest` type both live in `cvsuite.common.core`. `dataset.fm_request` is a real field on `VisionDataset`; `VisionDataset.to_json` / `from_json` carry it into and out of the provider subprocess.

## 2. Inputs and Outputs (I/O)
Inputs:
- shared dataset object (`cvsuite.common.core.VisionDataset`) with a populated `fm_request`
- `RunnerConfig`, either created directly or inferred via `RunnerConfig.from_dataset()`
- optional `no_resume` toggle from branch transforms

Outputs:
- a new shared dataset loaded from provider-written `out.json`
- provider cache directories under `<CVSUITE_HOME>/providers/<provider>/`
- shared Hugging Face snapshots under `<CVSUITE_HOME>/hub/`
- resumable FM run state under the runs root
- temporary stage directories cleaned after each run

## 3. Interfaces
Primary runtime types and functions:
- `common/fm/core/export.py`
  - `dump_dataset(ds, path)` delegates to `VisionDataset.to_json(...)`
  - `load_dataset(path)` delegates to `VisionDataset.from_json(...)`
- `RunnerConfig`
  - main fields: `provider_name`, `provider_family`, `provider_class`, `provider_module`, `model_name`, `task`, `device`, `batch_size`, `prompt`, `precision`, `max_gpu_memory`, `config_path`, `weights_dir`, `model_args`
  - `from_dataset(dataset)` infers provider family from `dataset.fm_request` and dataset metadata
- `FMRunner`
  - `from_dataset(dataset)`
  - `run(dataset, no_resume=False)`
  - `_ensure_runtime_python()`
  - `_ensure_venv()`
  - `_invoke_model(python_bin, in_json, out_json, no_resume=False)`

Runtime path/env interfaces:
- runner attributes include:
  - `caller_cwd`
  - `fm_root` (installed package directory; code only, never written to)
  - `src_root` (import root, derived from `cvsuite.__file__` so it is valid for wheel installs)
  - `cache_root`
  - `hub_dir`
  - `model_cache`
  - `venv_dir`
  - `weights_dir`
  - `pkgs_dir`
  - `stage_dir`
- cache root: resolved by `common/fm/core/paths.cvsuite_home()` — see `[[common/fm_cache]]`
- runs root: temp-directory child `cvsuite-fm-runs`
- provider subprocess env vars include:
  - `PYTHONPATH`
  - `FM_CACHE_DIR`
  - `FM_HUB_DIR`
  - `FM_MODEL_CACHE`
  - `FM_RUNS_ROOT`
  - `FM_WEIGHTS_DIR`
  - `FM_STAGE_DIR`
  - `FM_CALLER_CWD`
  - `HF_MODULES_CACHE`

Provider subprocess CLI contract:
- every provider is launched as:
  - `python -m cvsuite.common.fm.providers.<family>.<provider> --input <in.json> --output <out.json>`
- `--no-resume` is appended only when the branch requested it

## 4. Business Decisions and Strict Policies
- Runtime family inference is strict:
  - `dataset.fm_request.task == "gen"` uses `dataset.meta["gen_mode"]` to choose family `create` or `edit`
  - `task == "classify"` maps to family `classify`
  - otherwise the family equals the task name unless `request.meta["provider_family"]` overrides it
- Local providers run in a managed virtualenv; API providers run with the current interpreter.
- Nothing is written inside the installed package. `FM_WEIGHTS_DIR` stays per-provider for non-Hugging-Face artifacts; `FM_HUB_DIR` is the shared Hugging Face repo cache.
- `PYTHONPATH` carries only the import root; the old repo-root entry was a source-checkout assumption.
- Every run resets the stage directory before execution and empties it again in `finally`, even on failure.
- `--device cpu` explicitly hides GPUs from the provider subprocess by setting `CUDA_VISIBLE_DEVICES=""`.
- Auto allocator hints are only injected when all of the following are true:
  - requested device is `auto`
  - precision is not `nf4`
  - provider is in the managed Hugging Face auto-placement allowlist
  - allocator env vars were not already set by the caller
- `dim_edit` additionally disables CUDA memory caching unless running on CPU.
- Venv health checks are mandatory. Existing venvs are rebuilt when import probes fail or when specific dependency assertions fail.
- CUDA-oriented shared-venv providers inject torch package version/index env vars during setup so their setup scripts can install the expected CUDA wheels.

## 5. Implementation Details
Runtime serialization boundary:
- provider subprocesses only see JSON on disk, not live Python objects
- `VisionDataset.to_json(...)` / `from_json(...)` serialize every field, including `fm_request`, `embeddings`, and `labelme_shapes`; `from_json` re-runs `normalize_dataset(...)` to coerce task hints after decode

Process spawn sequence:
1. `FMRunner.run()` creates cache directories and clears `stage/`.
2. The input dataset is serialized to `in.json`.
3. Runtime Python is resolved:
  - API provider => current `sys.executable`
  - local provider => managed `venv/bin/python`
4. `_invoke_model()` launches:
  - `python -m cvsuite.common.fm.providers.<family>.<provider> --input <in.json> --output <out.json>`
  - plus `--no-resume` when requested
5. The provider writes `out.json`.
6. The runner reloads the dataset from JSON and cleans `stage/`.

Health-check details:
- the venv probe imports both generic dependencies and provider-specific modules
- `sam3` checks for deprecated `timm.models.layers` imports and triggers rebuild if patching is needed
- `internvl` health checks include `einops` and `timm`
- `dim_edit` health checks include `diffusers`, `matplotlib`, `mmcv`, and `transformers`
- generation model health checks include `accelerate`, `diffusers`, and `transformers`
- rebuild failures report the last health-check summary in the thrown error

Related docs:
- `[[common/fm_provider]]`
- `[[common/fm_utils]]`
- `[[common/fm_cache]]`
- `[[common/records_dataset]]`
