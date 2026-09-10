"""Purge shared FM caches wholly or partially."""

from __future__ import annotations

import argparse
import sys

from cvsuite.common.cache import core


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("providers", nargs="*", help="Provider ids to purge, e.g. deep_orientation clip.")
    parser.add_argument("--all", action="store_true", help="Purge all shared FM caches.")
    parser.add_argument(
        "--part",
        choices=list(core.PURGE_PARTS),
        default="all",
        help=(
            "Remove everything, or only the provider venv, weights, pkgs, or the shared hub. "
            "Shared hub entries are unlinked; the models in your Hugging Face cache are never deleted."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting anything.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the interactive confirmation prompt.")


def _resolve_targets(args: argparse.Namespace) -> tuple[list[core.ProviderCacheInfo], core.HubCacheInfo | None]:
    if args.all and args.providers:
        raise SystemExit("Pass either --all or one or more PROVIDER arguments, not both.")
    if not args.all and not args.providers:
        raise SystemExit("Pass --all or at least one PROVIDER to purge.")

    # The hub is shared across providers, so there is no provider it can be scoped to.
    if args.part == "hub":
        if args.providers:
            raise SystemExit("--part hub purges the shared hub; pass --all instead of provider names.")
        return [], core.describe_hub()

    infos = core.list_provider_caches(None if args.all else args.providers)
    if args.providers:
        missing = core.missing_providers(args.providers, infos)
        if missing:
            raise SystemExit(f"No cache entry found for: {', '.join(missing)}")
    return infos, core.describe_hub() if args.all else None


def _print_plan(actions: list[core.PurgeAction], *, dry_run: bool) -> None:
    if not actions:
        print("no cache files matched the requested purge target")
        return
    verb = "would remove" if dry_run else "removing"
    for action in actions:
        note = " (shared: unlink only)" if action.path.is_symlink() else ""
        print(
            f"{verb}: provider={action.provider} part={action.part} "
            f"size={core.format_size(action.size)} path={action.path}{note}"
        )
    total = sum(action.size for action in actions)
    label = "reclaimable" if dry_run else "reclaimed"
    print(f"{label}: {core.format_size(total)}")


def _confirm(actions: list[core.PurgeAction]) -> bool:
    if not actions:
        return False
    total = core.format_size(sum(action.size for action in actions))
    answer = input(f"Delete {len(actions)} cache target(s) and reclaim about {total}? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def run(args: argparse.Namespace) -> int:
    infos, hub = _resolve_targets(args)
    actions = core.plan_purge(infos, part=args.part, hub=hub)
    _print_plan(actions, dry_run=bool(args.dry_run))

    if args.dry_run or not actions:
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            raise SystemExit("Refusing to purge without --yes in non-interactive mode.")
        if not _confirm(actions):
            print("aborted")
            return 1

    core.apply_purge(actions)
    return 0
