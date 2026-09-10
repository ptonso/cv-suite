"""Print the resolved cvsuite cache roots."""

from __future__ import annotations

import argparse
import json

from cvsuite.common.cache import core


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def run(args: argparse.Namespace) -> int:
    user_hub = core.resolve_user_hf_hub()
    roots = {
        "cache_root": str(core.resolve_cache_root()),
        "hub_dir": str(core.resolve_hub_dir()),
        "providers_dir": str(core.resolve_providers_dir()),
        "hf_cache": str(user_hub) if user_hub else None,
    }
    if args.json:
        print(json.dumps(roots, indent=2))
        return 0
    print(f"cache root:    {roots['cache_root']}   (set CVSUITE_HOME to move it)")
    print(f"hub:           {roots['hub_dir']}")
    print(f"providers:     {roots['providers_dir']}")
    print(f"hf cache:      {roots['hf_cache'] or '(none detected)'}   (reused by symlink when it already has a model)")
    return 0
