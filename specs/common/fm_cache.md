# FM Cache

Related: **Read first** [fm_runtime](fm_runtime.md). **See also** [fm_provider](fm_provider.md), [fm_utils](fm_utils.md), [common cli](cli.md).

## Overview

The FM cache owns every byte cvsuite writes to disk on behalf of a foundation model: provider virtual environments, git checkouts, Hugging Face weight snapshots, download staging, and the `cvsuite common` commands that inspect and purge them. It is more than a weights folder: local providers depend on it for their Python runtimes and working directories, so the setup-script contract is part of this spec.

Two properties shape it. The cache lives outside the installed package, because an installed `site-packages` may be read-only, is wiped on upgrade, and is invisible to the user. And Hugging Face weights are shared with the user's own Hugging Face cache by symlink, so a checkpoint already downloaded for another project is not downloaded twice. Everything sits under one root so the user can see which models exist. See [the central cache note](../logs/2026-09-20_central-model-cache.md).

## Language

- **Cache root** (`CVSUITE_HOME`): the single directory holding everything below.
- **Hub**: the shared Hugging Face repo cache under the root, in the standard Hugging Face layout.
- **Owned vs shared repo**: a hub entry cvsuite downloaded, or a symlink into the user's own Hugging Face cache.
- **Stage dir**: throwaway per-provider directory where downloads are assembled before promotion.
- **Part**: one subtree of a provider directory: `venv`, `weights`, `pkgs`, or the shared `hub`.

## Map

- `common/fm/core/paths.py`: root resolution, hub layout, share probing. Imports nothing heavy.
- `common/cache/core.py`: listing and purge logic.
- `common/cache/commands/`: the `cvsuite common` commands ([common cli](cli.md)).
- `common/fm/providers/setup_venv/<provider>.sh`: per-provider environment setup.

## Contracts

### Layout

```
$CVSUITE_HOME
├── hub/                                    # shared Hugging Face repo cache
│   ├── models--laion--CLIP-ViT-H-14/       # owned: cvsuite downloaded it
│   └── models--facebook--sam3 -> ~/.cache/huggingface/hub/models--facebook--sam3
└── providers/<provider>/
    ├── venv/                               # managed runtime
    ├── weights/                            # per-provider non-HF artifacts, script tmp and pip cache
    ├── pkgs/                               # upstream git checkouts (sam3, dim_edit)
    └── stage/                              # download staging, cleared every run
```

`hub/` sits inside `CVSUITE_HOME` next to `providers/*/stage/`, so promoting a staged download is a same-filesystem `Path.replace` and stays atomic. Hugging Face repos are addressed by `hub_dir`, never by `weights_dir`.

### Root resolution (`common/fm/core/paths.py`)

- `cvsuite_home()`: `CVSUITE_HOME`, else `$XDG_CACHE_HOME/cvsuite`, else `~/.cache/cvsuite`.
- `hub_dir()`: `CVSUITE_HUB_DIR`, else `<home>/hub`.
- `providers_dir()`: `<home>/providers`.
- `user_hf_hub_dir()`: `HF_HUB_CACHE`, else `HUGGINGFACE_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`; `None` when that is cvsuite's own hub.
- `shared_repo_target(model_id, revision=None, require_patterns=None)`: the user's repo dir when it holds a usable snapshot, else `None`.
- `link_shared_repo(hub, model_id, target)`, `drop_stale_share(repo_cache, revision=None)`.
- `hf_repo_cache_dir(root, model_id)`, `resolve_cached_hf_snapshot(repo_cache_dir, revision=None)`.

Environment inputs may come from the process environment or an upward-searched `.env`: `CVSUITE_HOME`, `CVSUITE_HUB_DIR`, `XDG_CACHE_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME`.

### Cache core (`common/cache/core.py`)

`resolve_cache_root()`, `resolve_providers_dir()`, `resolve_hub_dir()`, `resolve_user_hf_hub()`, `describe_provider_cache(provider_dir)`, `list_provider_caches(providers=None)`, `describe_hub(hub_root=None)` returning `HubCacheInfo` / `HubRepoInfo`, `missing_providers(requested, infos)`, `format_size(...)`, `plan_purge(infos, part=..., hub=None)`, `apply_purge(actions)`.

### Commands

See [common cli](cli.md) for flags. `cache-path` prints the root, hub, providers dir and detected Hugging Face cache. `cache-list` prints the roots, then the hub table marking each repo owned or shared, then per-provider `venv_size`, `weights_size`, `pkgs_size`, `total_size`. `cache-purge` plans with `plan_purge` and applies with `apply_purge`.

### Managed environment setup

- A local provider's runtime is `<providers_dir>/<provider>/venv`. Setup runs `bash common/fm/providers/setup_venv/<provider>.sh <venv_dir>`, with CUDA torch version and index variables set for CUDA-oriented shared-environment providers.
- Scripts derive their paths from `dirname "$VENV_DIR"`, so the layout above is part of their contract. The runtime assumes the finished venv has `bin/python`.
- Upstream git checkouts go in `pkgs/`, never inside the venv.
- Scripts are declared as `package-data` in `pyproject.toml`.
- `paddleocr.sh` defaults `PADDLE_PDX_MODEL_SOURCE` to `BOS` unless the caller overrides it.

### Hugging Face repo resolution

`materialize_hf_model_source()` resolves a repo id in this order:

1. `<hub>/models--org--name` is a real directory with a valid snapshot: use it.
2. It is a symlink: validate the target; if usable, use it, otherwise `drop_stale_share` unlinks it and resolution continues.
3. Nothing there: probe the user's cache with `shared_repo_target(...)` and, on success, `link_shared_repo(...)`.
4. Otherwise download through the staging path.

### Staged downloads

Downloads assemble in `stage/`, then get promoted into the hub. `ensure_hf_caches(stage_work_dir)` redirects `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `XDG_CACHE_HOME` and `TORCH_HOME` into a throwaway directory, so transitive downloads die with it and the durable cache only receives a finished `models--*` tree. Staging also makes a partial snapshot unreachable: `_is_valid_snapshot()` accepts any non-empty directory and `snapshot_download` fills `snapshots/<sha>/` incrementally, so an interrupted direct download would pass validation and fail later on a missing shard. Promotion unlinks a symlinked `repo_cache` and never `rmtree`s it, since that would follow the link into the user's models. After promotion, `ensure_hf_caches(stage_dir, hub=hub_dir)` points repo fetches at the shared hub so transitive downloads dedupe.

## Constraints

- Nothing MUST be written inside the installed package.
- Roots MUST be resolved from a snapshot of the environment taken when `paths.py` is imported. `ensure_hf_caches()` rewrites `HF_HOME`, `HF_HUB_CACHE` and `XDG_CACHE_HOME` at runtime, so reading live variables would make resolution depend on whether a provider ran first.
- The canonical root MUST be whatever `FMRunner` resolves; cache commands use the same resolution.
- A shared repo MUST be linked only when the user's copy holds a complete snapshot: the requested revision (or `refs/main`) resolves, and every `require_patterns` entry is present. A partial share MUST be rejected and cvsuite downloads a private copy, so it never writes into a cache it does not own.
- A symlinked hub entry that no longer resolves MUST be unlinked and refetched. The link target MUST NOT be deleted.
- Purge MUST NOT follow a symlink. Shared entries are unlinked, and `apply_purge` MUST refuse any path that does not resolve inside `cvsuite_home()`.
- Shared repos MUST report size `0`.
- `--part hub` MUST reject provider names and require `--all`. `--all` and explicit provider names MUST be mutually exclusive.
- Alias normalization MUST apply to cache operations (`minicpm-v` targets `minicpm_v`).
- `cache-purge` MUST require confirmation for non-dry-run deletion unless `--yes` is given.
- Purging a part MUST remove only that subtree. Purging all parts removes the provider directory, and empty `hub/`, `providers/` and root directories.
- `--part venv` MUST preserve `pkgs/`, so a rebuilt venv reuses existing checkouts.
- Missing requested provider caches MUST cause `SystemExit` in listing and purge.
- There is no migration from the old in-package `.cache`; a leftover directory is the user's to delete.
- Tests: `common/tests/test_cache_commands.py`, `test_fm_paths.py`, `test_fm_hub_share.py`.

## History

- [Central model cache](../logs/2026-09-20_central-model-cache.md): why weights are centralized and Hugging Face downloads reused.
- [Isolated provider environments](../logs/2026-09-20_isolated-provider-environments.md): why each local provider has its own venv.
