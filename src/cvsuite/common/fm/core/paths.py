from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# Snapshot of the environment as it looked before anything mutated it. `ensure_hf_caches`
# rewrites HF_HOME, HF_HUB_CACHE and XDG_CACHE_HOME in os.environ, and every root below is
# derived from one of those. Resolving from a snapshot taken at import makes the roots
# independent of whether a provider happened to call `ensure_hf_caches` first.
_ENV_AT_IMPORT: Dict[str, str] = dict(os.environ)


def _dotenv_search_roots() -> List[Path]:
    roots: List[Path] = []
    for raw in (os.environ.get("FM_CALLER_CWD"), str(Path.cwd())):
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if path not in roots:
            roots.append(path)
    return roots


def _dotenv_candidates() -> List[Path]:
    candidates: List[Path] = []
    seen: set[Path] = set()
    for root in _dotenv_search_roots():
        for base in (root, *root.parents):
            env_path = base / ".env"
            if env_path in seen:
                continue
            seen.add(env_path)
            if env_path.is_file():
                candidates.append(env_path)
    return candidates


def _parse_dotenv_line(line: str) -> Tuple[str, str] | None:
    text = str(line).strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].lstrip()
    if "=" not in text:
        return None
    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key or any(ch.isspace() for ch in key):
        return None
    value = raw_value.strip()
    if not value:
        return key, ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return key, value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return key, value


def _load_dotenv_values(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return values
    for line in lines:
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        values.setdefault(key, value)
    return values


def iter_env_var_candidates(
    keys: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> Iterable[Tuple[str, str, str]]:
    source_env = os.environ if env is None else env
    seen: set[str] = set()
    for key in keys:
        value = str(source_env.get(key) or "").strip()
        if value and value not in seen:
            seen.add(value)
            yield key, value, f"environment variable {key}"
    for env_path in _dotenv_candidates():
        values = _load_dotenv_values(env_path)
        for key in keys:
            value = str(values.get(key) or "").strip()
            if value and value not in seen:
                seen.add(value)
                yield key, value, f"{env_path} ({key})"


def _snapshot_env_path(*keys: str) -> Path | None:
    for _key, value, _source in iter_env_var_candidates(keys, env=_ENV_AT_IMPORT):
        return Path(value).expanduser()
    return None


def cvsuite_home() -> Path:
    """Root of everything cvsuite caches: provider runtimes, weights and staging."""

    explicit = _snapshot_env_path("CVSUITE_HOME")
    if explicit is not None:
        return explicit
    xdg_cache = _snapshot_env_path("XDG_CACHE_HOME")
    if xdg_cache is not None:
        return xdg_cache / "cvsuite"
    return Path.home() / ".cache" / "cvsuite"


def hub_dir() -> Path:
    """Shared Hugging Face repo cache, in standard hub layout."""

    explicit = _snapshot_env_path("CVSUITE_HUB_DIR")
    if explicit is not None:
        return explicit
    return cvsuite_home() / "hub"


def providers_dir() -> Path:
    return cvsuite_home() / "providers"


def user_hf_hub_dir() -> Path | None:
    """The user's own Hugging Face hub cache, or None when it is cvsuite's own hub."""

    resolved = _snapshot_env_path("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
    if resolved is None:
        hf_home = _snapshot_env_path("HF_HOME")
        resolved = (hf_home / "hub") if hf_home is not None else Path.home() / ".cache" / "huggingface" / "hub"
    if _same_path(resolved, hub_dir()):
        return None
    return resolved


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _hf_repo_cache_dir_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def hf_repo_cache_dir(cache_root: Path, model_id: str) -> Path:
    return cache_root / _hf_repo_cache_dir_name(model_id)


def resolve_cached_hf_snapshot(repo_cache_dir: Path, revision: str | None = None) -> tuple[Path | None, str | None]:
    snapshots_dir = repo_cache_dir / "snapshots"

    def _is_valid_snapshot(path: Path) -> bool:
        if not path.is_dir():
            return False
        try:
            next(path.iterdir())
        except (OSError, StopIteration):
            return False
        return True

    if revision:
        snapshot_dir = snapshots_dir / revision
        return (snapshot_dir, revision) if _is_valid_snapshot(snapshot_dir) else (None, None)

    ref_path = repo_cache_dir / "refs" / "main"
    try:
        cached_revision = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        cached_revision = ""
    if cached_revision:
        snapshot_dir = snapshots_dir / cached_revision
        if _is_valid_snapshot(snapshot_dir):
            return snapshot_dir, cached_revision

    if not snapshots_dir.is_dir():
        return None, None

    candidates: list[Path] = []
    for child in snapshots_dir.iterdir():
        if _is_valid_snapshot(child):
            candidates.append(child)
    if not candidates:
        return None, None

    def _sort_key(path: Path) -> tuple[int, str]:
        try:
            mtime_ns = int(path.stat().st_mtime_ns)
        except OSError:
            mtime_ns = 0
        return mtime_ns, path.name

    candidates.sort(key=_sort_key)
    snapshot_dir = candidates[-1]
    return snapshot_dir, snapshot_dir.name


def shared_repo_target(
    model_id: str,
    revision: str | None = None,
    require_patterns: Sequence[str] | None = None,
) -> Path | None:
    """The user's HF cache dir for `model_id`, but only when it holds a usable snapshot.

    A share is rejected when the snapshot is missing, is the wrong revision, or does not
    contain every `require_patterns` entry the caller needs -- a user may well have fetched
    only part of a repo. Rejecting means the caller downloads a private copy, so cvsuite
    never has to write into a cache it does not own.
    """

    user_hub = user_hf_hub_dir()
    if user_hub is None or not user_hub.is_dir():
        return None
    repo_dir = hf_repo_cache_dir(user_hub, model_id)
    if not repo_dir.is_dir():
        return None
    snapshot_path, _revision = resolve_cached_hf_snapshot(repo_dir, revision)
    if snapshot_path is None:
        return None
    for pattern in require_patterns or ():
        if not any(snapshot_path.glob(pattern)):
            return None
    return repo_dir


def link_shared_repo(hub: Path, model_id: str, target: Path) -> Path:
    """Point `<hub>/models--org--name` at `target`, replacing a stale link if present."""

    repo_cache = hf_repo_cache_dir(hub, model_id)
    if repo_cache.is_symlink():
        if _same_path(repo_cache, target):
            return repo_cache
        repo_cache.unlink()
    hub.mkdir(parents=True, exist_ok=True)
    repo_cache.symlink_to(target, target_is_directory=True)
    return repo_cache


def drop_stale_share(repo_cache: Path, revision: str | None = None) -> bool:
    """Unlink `repo_cache` when it is a symlink that no longer resolves to a usable repo.

    Returns True when the caller must treat the repo as absent. The link target is never
    deleted: a share belongs to whoever created it.
    """

    if not repo_cache.is_symlink():
        return False
    snapshot_path, _revision = resolve_cached_hf_snapshot(repo_cache, revision)
    if snapshot_path is not None:
        return False
    repo_cache.unlink()
    return True
