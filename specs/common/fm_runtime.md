# FM Runtime

Related: **Read first** [records_dataset](records_dataset.md). **See also** [fm_provider](fm_provider.md), [fm_utils](fm_utils.md), [fm_cache](fm_cache.md).

## Overview

The FM runtime turns a shared dataset carrying an `FMRequest` into a run of a provider in a separate process. Branches never import provider code to run inference. They attach `dataset.fm_request` and hand the dataset to `FMRunner`, which writes it to JSON, resolves the provider module, prepares or checks the provider's virtual environment, starts a new Python process, and reloads the result. The reasons for the process boundary are in [the isolation note](../logs/2026-09-20_isolated-provider-environments.md).

Walk-through of one run: the runner clears the stage directory, writes `in.json`, picks the interpreter (the managed venv for local providers, the current one for API providers), launches the provider with `--input` and `--output`, waits for `out.json`, loads it, and clears the stage directory again.

## Language

- **Runner** (`FMRunner`), **runner config** (`RunnerConfig`): the objects that perform and describe a run.
- **Runs root**: temp-directory child `cvsuite-fm-runs`, holding resumable run state.
- **Health check**: an import probe of a venv that decides whether it must be rebuilt.

## Map

- `common/fm/core/`: runner, config, `export.py`, `paths.py` ([fm_cache](fm_cache.md)).
- Providers: [fm_provider](fm_provider.md). Helpers: [fm_utils](fm_utils.md).

## Contracts

### Types

- `common/fm/core/export.py`: `dump_dataset(ds, path)` and `load_dataset(path)` delegate to `VisionDataset.to_json` / `from_json`.
- `RunnerConfig`: `provider_name`, `provider_family`, `provider_class`, `provider_module`, `model_name`, `task`, `device`, `batch_size`, `prompt`, `precision`, `max_gpu_memory`, `config_path`, `weights_dir`, `model_args`. `from_dataset(dataset)` infers the family from `dataset.fm_request` and dataset metadata.
- `FMRunner`: `from_dataset(dataset)`, `run(dataset, no_resume=False)`, `_ensure_runtime_python()`, `_ensure_venv()`, `_invoke_model(python_bin, in_json, out_json, no_resume=False)`. Attributes: `caller_cwd`, `fm_root` (installed package, never written to), `src_root` (import root derived from `cvsuite.__file__`), `cache_root`, `hub_dir`, `model_cache`, `venv_dir`, `weights_dir`, `pkgs_dir`, `stage_dir`.

### Provider command line

```
python -m cvsuite.common.fm.providers.<family>.<provider> --input <in.json> --output <out.json> [--no-resume]
```

`--no-resume` is appended only when the branch asked for it.

### Environment given to the provider

`PYTHONPATH` (import root only), `FM_CACHE_DIR`, `FM_HUB_DIR` (shared Hugging Face cache), `FM_MODEL_CACHE`, `FM_RUNS_ROOT`, `FM_WEIGHTS_DIR` (per-provider non-Hugging-Face artifacts), `FM_STAGE_DIR`, `FM_CALLER_CWD`, `HF_MODULES_CACHE`. Cache roots come from [fm_cache](fm_cache.md).

### Serialization boundary

The provider sees only JSON on disk. `to_json` / `from_json` carry every field, including `fm_request`, `embeddings` and `labelme_shapes`; `from_json` re-runs `normalize_dataset`.

### Family inference

- `fm_request.task == "gen"`: `dataset.meta["gen_mode"]` chooses family `create` or `edit`.
- `task == "classify"`: family `classify`.
- Otherwise the family equals the task name, unless `request.meta["provider_family"]` overrides it.

## Constraints

- Local providers MUST run in a managed virtual environment and API providers in the current interpreter.
- Nothing MUST be written inside the installed package.
- Every run MUST reset the stage directory before execution and empty it in `finally`, even on failure.
- `--device cpu` MUST hide GPUs from the provider process by setting `CUDA_VISIBLE_DEVICES=""`.
- Allocator hints for auto placement MUST be injected only when the device is `auto`, precision is not `nf4`, the provider is in the managed Hugging Face auto-placement allowlist, and the caller has not set the allocator variables. `dim_edit` additionally disables CUDA memory caching unless running on CPU.
- Venv health checks MUST run. An existing venv MUST be rebuilt when import probes or specific dependency assertions fail, and a failed rebuild MUST report the last health-check summary in the error. The probe imports generic and provider-specific modules: `sam3` (deprecated `timm.models.layers` imports, patched by rebuild), `internvl` (`einops`, `timm`), `dim_edit` (`diffusers`, `matplotlib`, `mmcv`, `transformers`), generation providers (`accelerate`, `diffusers`, `transformers`).
- `PYTHONPATH` MUST carry only the import root.
- Family inference MUST follow the rules above and MUST NOT guess.

## History

- [Isolated provider environments](../logs/2026-09-20_isolated-provider-environments.md): why a run is a separate process.
- [Shared dataset contract](../logs/2026-09-20_shared-dataset-contract.md): why the request rides on the dataset.
- [Central model cache](../logs/2026-09-20_central-model-cache.md): why caches are centralized under one root.
