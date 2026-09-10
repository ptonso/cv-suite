"""List shared FM cache entries and their disk usage."""

from __future__ import annotations

import argparse
import json

from cvsuite.common.cache import core


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("providers", nargs="*", help="Optional provider ids to filter, e.g. deep_orientation clip.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--sort",
        choices=["name", "size"],
        default="name",
        help="Sort by provider name or total cache size.",
    )
    parser.add_argument("--bytes", action="store_true", help="Show raw byte counts instead of human-readable sizes.")


def _sort_infos(infos: list[core.ProviderCacheInfo], sort_key: str) -> list[core.ProviderCacheInfo]:
    if sort_key == "size":
        return sorted(infos, key=lambda info: (-info.total_size, info.provider))
    return sorted(infos, key=lambda info: info.provider)


def _render_size(num_bytes: int, raw_bytes: bool) -> str:
    return str(num_bytes) if raw_bytes else core.format_size(num_bytes)


def _print_roots() -> None:
    print(f"cache root: {core.resolve_cache_root()}")
    print(f"hub:        {core.resolve_hub_dir()}")
    user_hub = core.resolve_user_hf_hub()
    print(f"hf cache:   {user_hub if user_hub else '(none detected)'}")


def _print_hub(hub: core.HubCacheInfo, *, raw_bytes: bool) -> None:
    if not hub.repos:
        return
    print(f"hub repos: {len(hub.repos)} ({hub.shared_count} shared)")
    for repo in hub.repos:
        if repo.shared:
            print(f"  shared  {repo.repo} -> {repo.target}")
        else:
            print(f"  owned   {repo.repo}  {_render_size(repo.size, raw_bytes)}")
    print(f"  owned total: {_render_size(hub.owned_size, raw_bytes)}")


def _print_human(infos: list[core.ProviderCacheInfo], hub: core.HubCacheInfo, *, raw_bytes: bool) -> None:
    _print_roots()
    _print_hub(hub, raw_bytes=raw_bytes)
    if not infos:
        if not hub.repos:
            print("no shared FM caches found")
        return

    for info in infos:
        print(f"provider: {info.provider}")
        print(f"  total:   {_render_size(info.total_size, raw_bytes)}")
        print(f"  venv:    {_render_size(info.venv_size, raw_bytes)}  {info.venv_dir}")
        print(f"  weights: {_render_size(info.weights_size, raw_bytes)}  {info.weights_dir}")
        if info.pkgs_size:
            print(f"  pkgs:    {_render_size(info.pkgs_size, raw_bytes)}  {info.pkgs_dir}")
        if info.other_size:
            print(f"  other:   {_render_size(info.other_size, raw_bytes)}  {info.root}")


def run(args: argparse.Namespace) -> int:
    infos = core.list_provider_caches(args.providers)
    if args.providers:
        missing = core.missing_providers(args.providers, infos)
        if missing:
            raise SystemExit(f"No cache entry found for: {', '.join(missing)}")
    infos = _sort_infos(infos, args.sort)
    hub = core.describe_hub()

    if args.json:
        payload = {
            "cache_root": str(core.resolve_cache_root()),
            "hub_dir": str(hub.root),
            "hf_cache": str(core.resolve_user_hf_hub() or ""),
            "hub": {
                "owned_size": hub.owned_size,
                "shared_count": hub.shared_count,
                "repos": [
                    {
                        "repo": repo.repo,
                        "path": str(repo.path),
                        "shared": repo.shared,
                        "target": str(repo.target) if repo.target else None,
                        "size": repo.size,
                    }
                    for repo in hub.repos
                ],
            },
            "providers": [
                {
                    "provider": info.provider,
                    "root": str(info.root),
                    "venv_dir": str(info.venv_dir),
                    "weights_dir": str(info.weights_dir),
                    "pkgs_dir": str(info.pkgs_dir),
                    "has_venv": info.has_venv,
                    "has_weights": info.has_weights,
                    "total_size": info.total_size,
                    "venv_size": info.venv_size,
                    "weights_size": info.weights_size,
                    "pkgs_size": info.pkgs_size,
                    "other_size": info.other_size,
                }
                for info in infos
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    _print_human(infos, hub, raw_bytes=bool(args.bytes))
    return 0
