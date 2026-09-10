from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from cvsuite.common.fm.core.paths import hub_dir as resolve_hub_root
from cvsuite.common.fm.core.paths import providers_dir as resolve_providers_root
from cvsuite.common.fm.core.paths import user_hf_hub_dir
from cvsuite.common.fm.core.runner import get_fm_cache_root

PROVIDER_CACHE_ALIASES = {
    "minicpm-v": "minicpm_v",
}

PURGE_PARTS = ("all", "venv", "weights", "pkgs", "hub")


@dataclass(frozen=True)
class ProviderCacheInfo:
    provider: str
    root: Path
    venv_dir: Path
    weights_dir: Path
    pkgs_dir: Path
    total_size: int
    venv_size: int
    weights_size: int
    pkgs_size: int
    other_size: int
    has_venv: bool
    has_weights: bool


@dataclass(frozen=True)
class HubRepoInfo:
    """One Hugging Face repo in the shared hub.

    `shared` means the entry is a symlink into a cache cvsuite does not own, so its bytes
    are not cvsuite's to count or to delete -- only the link is.
    """

    repo: str
    path: Path
    shared: bool
    target: Path | None
    size: int


@dataclass(frozen=True)
class HubCacheInfo:
    root: Path
    repos: list[HubRepoInfo]
    owned_size: int
    shared_count: int


@dataclass(frozen=True)
class PurgeAction:
    provider: str
    part: str
    path: Path
    size: int


def resolve_cache_root() -> Path:
    return get_fm_cache_root()


def resolve_providers_dir() -> Path:
    return resolve_providers_root()


def resolve_hub_dir() -> Path:
    return resolve_hub_root()


def resolve_user_hf_hub() -> Path | None:
    return user_hf_hub_dir()


def normalize_provider_cache_name(provider: str) -> str:
    normalized = str(provider).strip().lower()
    return PROVIDER_CACHE_ALIASES.get(normalized, normalized)


def _walk_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_symlink():
        try:
            return int(path.lstat().st_size)
        except OSError:
            return 0
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0

    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_symlink():
                total += int(child.lstat().st_size)
                continue
            if child.is_file():
                total += int(child.stat().st_size)
        except OSError:
            continue
    return total


def _iter_provider_dirs(providers_root: Path) -> Iterable[Path]:
    if not providers_root.exists():
        return []
    return [child for child in sorted(providers_root.iterdir()) if child.is_dir()]


def describe_provider_cache(provider_dir: Path) -> ProviderCacheInfo:
    venv_dir = provider_dir / "venv"
    weights_dir = provider_dir / "weights"
    pkgs_dir = provider_dir / "pkgs"
    venv_size = _walk_size(venv_dir)
    weights_size = _walk_size(weights_dir)
    pkgs_size = _walk_size(pkgs_dir)
    total_size = _walk_size(provider_dir)
    other_size = max(0, total_size - venv_size - weights_size - pkgs_size)
    return ProviderCacheInfo(
        provider=provider_dir.name,
        root=provider_dir,
        venv_dir=venv_dir,
        weights_dir=weights_dir,
        pkgs_dir=pkgs_dir,
        total_size=total_size,
        venv_size=venv_size,
        weights_size=weights_size,
        pkgs_size=pkgs_size,
        other_size=other_size,
        has_venv=venv_dir.exists(),
        has_weights=weights_dir.exists(),
    )


def list_provider_caches(
    providers: Sequence[str] | None = None,
    *,
    providers_root: Path | None = None,
) -> list[ProviderCacheInfo]:
    root = providers_root or resolve_providers_dir()
    infos = [describe_provider_cache(provider_dir) for provider_dir in _iter_provider_dirs(root)]
    if not providers:
        return infos
    wanted = {normalize_provider_cache_name(provider) for provider in providers}
    return [info for info in infos if info.provider in wanted]


def describe_hub(hub_root: Path | None = None) -> HubCacheInfo:
    root = hub_root or resolve_hub_dir()
    repos: list[HubRepoInfo] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.name.startswith("models--"):
                continue
            shared = child.is_symlink()
            target = child.resolve() if shared else None
            repos.append(
                HubRepoInfo(
                    repo=child.name,
                    path=child,
                    shared=shared,
                    target=target,
                    size=0 if shared else _walk_size(child),
                )
            )
    return HubCacheInfo(
        root=root,
        repos=repos,
        owned_size=sum(repo.size for repo in repos),
        shared_count=sum(1 for repo in repos if repo.shared),
    )


def missing_providers(providers: Sequence[str], infos: Sequence[ProviderCacheInfo]) -> list[str]:
    found = {info.provider for info in infos}
    return sorted(provider for provider in providers if normalize_provider_cache_name(provider) not in found)


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        size /= 1024.0
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
    return f"{num_bytes} B"


def plan_purge(
    infos: Sequence[ProviderCacheInfo],
    *,
    part: str,
    hub: HubCacheInfo | None = None,
) -> list[PurgeAction]:
    actions: list[PurgeAction] = []
    for info in infos:
        if part == "all":
            if info.root.exists():
                actions.append(PurgeAction(provider=info.provider, part="all", path=info.root, size=info.total_size))
            continue
        if part == "venv" and info.venv_dir.exists():
            actions.append(PurgeAction(provider=info.provider, part="venv", path=info.venv_dir, size=info.venv_size))
        if part == "weights" and info.weights_dir.exists():
            actions.append(PurgeAction(provider=info.provider, part="weights", path=info.weights_dir, size=info.weights_size))
        if part == "pkgs" and info.pkgs_dir.exists():
            actions.append(PurgeAction(provider=info.provider, part="pkgs", path=info.pkgs_dir, size=info.pkgs_size))
    if hub is not None and part in {"all", "hub"}:
        for repo in hub.repos:
            actions.append(PurgeAction(provider=repo.repo, part="hub", path=repo.path, size=repo.size))
    return actions


def apply_purge(actions: Sequence[PurgeAction]) -> None:
    cache_root = resolve_cache_root().resolve()
    for action in actions:
        _remove_cache_path(action.path, cache_root)
        if action.part in {"venv", "weights", "pkgs"}:
            provider_root = action.path.parent
            if provider_root.is_dir() and not any(provider_root.iterdir()):
                provider_root.rmdir()

    for directory in (resolve_hub_dir(), resolve_providers_dir(), cache_root):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def _remove_cache_path(path: Path, cache_root: Path) -> None:
    """Delete one cache entry, refusing anything cvsuite does not own.

    A symlink is only ever unlinked: a shared hub entry points into the user's own Hugging
    Face cache, and following it would delete models other applications depend on.
    """

    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return
    resolved = path.resolve()
    if resolved != cache_root and cache_root not in resolved.parents:
        raise RuntimeError(f"Refusing to purge {resolved}, which is outside the cvsuite cache at {cache_root}.")
    shutil.rmtree(resolved)
