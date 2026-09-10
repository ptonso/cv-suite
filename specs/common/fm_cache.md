# FM Cache

## 1. General Description
The FM cache subsystem owns every byte cvsuite writes to disk on behalf of a foundation model: provider-local virtualenvs, git checkouts, Hugging Face weight snapshots, download staging, and the `cvsuite common` commands that inspect and purge them. The cache is not just a weights folder. It is a contract that local FM providers depend on for bootstrapped Python runtimes, Hugging Face downloads, staged snapshots, and working directories. This is why the cache spec must include the venv/bash setup-script contract rather than only file sizes.

Two properties shape the whole design. First, the cache lives **outside the installed package**, because a pip-installed `site-packages` may be read-only, is wiped on upgrade, and is invisible to the user. Second, Hugging Face weights are **shared with the user's own HF cache by symlink** rather than duplicated, because a user who already pulled a 30 GB checkpoint for another project should not pay for it twice.

## 2. Inputs and Outputs (I/O)
Inputs:
- Provider names or aliases passed to `cvsuite common cache-list` and `cvsuite common cache-purge`.
- `CVSUITE_HOME`, `CVSUITE_HUB_DIR`, `XDG_CACHE_HOME`, `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` / `HF_HOME`, read from the environment or an upward-searched `.env`.
- Cache-root filesystem contents under the resolved `CVSUITE_HOME`.
- The user's own Hugging Face hub cache, read-only.
- Managed setup scripts under `common/fm/providers/setup_venv/*.sh`.

Outputs:
- Human-readable or JSON cache listings, distinguishing cvsuite-owned bytes from shared ones.
- Planned purge actions.
- Deleted provider subtrees; unlinked shared hub entries.
- Provider-local venvs and Hugging Face snapshot trees created by `[[common/fm_runtime]]`.

## 3. Interfaces

### Root resolution — `common/fm/core/paths.py`
- `cvsuite_home()` — `CVSUITE_HOME`, else `$XDG_CACHE_HOME/cvsuite`, else `~/.cache/cvsuite`
- `hub_dir()` — `CVSUITE_HUB_DIR`, else `<home>/hub`
- `providers_dir()` — `<home>/providers`
- `user_hf_hub_dir()` — `HF_HUB_CACHE`, else `HUGGINGFACE_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`; `None` when that resolves to cvsuite's own hub
- `shared_repo_target(model_id, revision=None, require_patterns=None)` — the user's repo dir when it holds a usable snapshot, else `None`
- `link_shared_repo(hub, model_id, target)` / `drop_stale_share(repo_cache, revision=None)`
- `hf_repo_cache_dir(root, model_id)` / `resolve_cached_hf_snapshot(repo_cache_dir, revision=None)`

### Layout
```
$CVSUITE_HOME
├── hub/                                    # shared Hugging Face repo cache, standard hub layout
│   ├── models--laion--CLIP-ViT-H-14/       # owned: cvsuite downloaded it
│   └── models--facebook--sam3 -> ~/.cache/huggingface/hub/models--facebook--sam3
└── providers/<provider>/
    ├── venv/                               # managed runtime
    ├── weights/                            # per-provider non-HF artifacts, plus script tmp/pip cache
    ├── pkgs/                               # upstream git checkouts (sam3, dim_edit)
    └── stage/                              # download staging, cleared every run
```

`hub/` deliberately sits inside `CVSUITE_HOME` alongside `providers/*/stage/`, so the staging promotion is a same-filesystem `Path.replace` and stays atomic.

### Cache core — `common/cache/core.py`
- `resolve_cache_root()`, `resolve_providers_dir()`, `resolve_hub_dir()`, `resolve_user_hf_hub()`
- `describe_provider_cache(provider_dir)`, `list_provider_caches(providers=None)`
- `describe_hub(hub_root=None)` returning `HubCacheInfo` / `HubRepoInfo`
- `missing_providers(requested, infos)`, `format_size(...)`
- `plan_purge(infos, part=..., hub=None)`, `apply_purge(actions)`

### CLI commands discovered through `cvsuite common`
- `cache-path`
  - prints the resolved cache root, hub, providers dir, and detected HF cache
  - flags: `--json`
- `cache-list`
  - provider filter, `--json`, `--sort name|size`, `--bytes`
- `cache-purge`
  - targets: explicit providers or `--all`
  - flags: `--part all|venv|weights|pkgs|hub`, `--dry-run`, `--yes`

### Managed venv/setup contract
- Local providers keep their runtime in `<providers_dir>/<provider>/venv`.
- Per-provider non-HF artifacts live in `<providers_dir>/<provider>/weights`; shared HF repos live in `<hub_dir>`.
- Setup is delegated to `common/fm/providers/setup_venv/<provider>.sh`.
- The runtime invokes those scripts using `bash <script> <venv_dir>`, optionally with CUDA torch env vars populated for shared CUDA-model families.
- Scripts derive their own paths from `dirname "$VENV_DIR"`, so the provider directory layout above is part of their contract.
- Upstream git checkouts go in `<providers_dir>/<provider>/pkgs`, never inside the venv. `_ensure_venv` rmtree's the venv before rerunning setup, so a checkout placed inside it is re-cloned on every rebuild, and `pkgs_dir` / `load_repo_checkout()` would not find it.
- The scripts are declared as `package-data` in `pyproject.toml`; without that they are absent from the wheel and every local provider fails at `Missing venv setup script`.

## 4. Business Decisions and Strict Policies
- The canonical cache root is whatever `FMRunner` resolves; cache commands intentionally piggyback on that runtime contract.
- Nothing is ever written inside the installed package.
- Roots are resolved from a snapshot of the environment taken when `paths.py` is imported. `ensure_hf_caches()` rewrites `HF_HOME`, `HF_HUB_CACHE` and `XDG_CACHE_HOME` in `os.environ` at runtime, and every root above is derived from one of those; reading live env would make resolution depend on whether a provider happened to run first.
- A shared repo is only linked when the user's copy holds a **complete** snapshot: the requested revision (or `refs/main`) must resolve, and every `require_patterns` entry must be present. A partial share is rejected and cvsuite downloads a private copy, so it never has to write into a cache it does not own.
- A symlinked hub entry that no longer resolves is unlinked and refetched. The link target is never deleted — a share belongs to whoever created it.
- Purge never follows a symlink: shared entries are unlinked, and `apply_purge` refuses any path that does not resolve inside `cvsuite_home()`.
- Shared repos report size `0`, because those bytes are not cvsuite's to reclaim.
- `--part hub` addresses the shared hub, which no single provider owns, so it rejects provider names and requires `--all`.
- Alias normalization is applied to cache operations. Example: `minicpm-v` targets `minicpm_v`.
- `cache-purge` requires explicit confirmation for non-dry-run deletions unless `--yes` is supplied.
- `--all` and explicit provider names are mutually exclusive.
- Purging a part only removes that subtree. Purging all parts removes the provider directory, and empty `hub/`, `providers/` and cache-root directories are removed too.
- `--part venv` preserves `pkgs/`, so rebuilding a venv reuses the existing checkouts rather than re-cloning them.
- Missing requested provider caches cause a `SystemExit` in listing or purge flows rather than being silently ignored.
- Setup-script behavior is part of the contract. Example: `paddleocr.sh` defaults `PADDLE_PDX_MODEL_SOURCE` to `BOS` unless the caller overrides it.
- There is no migration from the old in-package `.cache`. Per the repo's forward-only policy, a leftover directory is the user's to delete.

## 5. Implementation Details

### Hugging Face repo resolution
`materialize_hf_model_source()` resolves a repo id in this order:
1. `<hub>/models--org--name` is a real directory with a valid snapshot — use it.
2. It is a symlink — validate the target; usable means use it, otherwise `drop_stale_share` unlinks it and resolution continues.
3. Nothing there — probe the user's HF cache via `shared_repo_target(...)` and, on success, `link_shared_repo(...)`.
4. Otherwise download through the staging path below.

### Why downloads stage
Staging is not redundant with `huggingface_hub`'s own resume logic. It does two things nothing else does:
- `ensure_hf_caches(stage_work_dir)` redirects `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `XDG_CACHE_HOME` and `TORCH_HOME` into a throwaway directory, so everything a download pulls in transitively dies with it and the durable cache only ever receives a finished `models--*` tree.
- It makes a partially-downloaded snapshot unreachable. `_is_valid_snapshot()` accepts any non-empty directory and `snapshot_download` fills `snapshots/<sha>/` incrementally, so a direct download interrupted midway would pass validation, be returned with `local_files_only=True`, and fail much later inside `from_pretrained` on a missing shard.

Promotion therefore unlinks a symlinked `repo_cache` rather than calling `shutil.rmtree` on it; rmtree would follow the link and delete the user's models. After promotion, `ensure_hf_caches(stage_dir, hub=hub_dir)` points repo fetches at the shared hub so transitive downloads dedupe too.

### List and purge behavior
List:
- prints the resolved roots, then the hub table marking each repo owned or shared,
- walks provider directories computing `venv_size`, `weights_size`, `pkgs_size` and `total_size`,
- sorts by name or size, and emits JSON when requested.

Purge:
- `plan_purge()` determines which directories, files or links would be removed,
- `apply_purge()` performs removal via `_remove_cache_path`, which unlinks symlinks and refuses anything outside the cache root,
- dry-run leaves the filesystem unchanged,
- partial purge of one part preserves the others.

The setup-script side of the cache contract matters because the runtime assumes:
- the provider venv will contain `bin/python`,
- the health-check code can be run against that interpreter,
- provider scripts can install extra packages, patch repos, or pin CUDA wheels as needed.

Related docs:
- `[[common/fm_runtime]]`
- `[[common/fm_provider]]`
- `[[common/fm_utils]]`
- `[[cli]]`
