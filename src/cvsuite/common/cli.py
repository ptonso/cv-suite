"""Model cache: show where foundation-model weights and environments live, list them, and purge them."""

from __future__ import annotations

from cvsuite.branch_cli import run_generic_branch

EPILOG = """\
Models run in per-provider environments under one cache; `cache-path` shows its location (move it with CVSUITE_HOME).
Flags of one command: cvsuite common COMMAND --help.

examples:
  cvsuite common cache-path
  cvsuite common cache-list --json
  cvsuite common cache-purge clip --dry-run
  cvsuite common cache-purge --all --part hub
"""


def main(argv: list[str] | None = None) -> int:
    return run_generic_branch(
        package_name=__package__ or "cvsuite.common",
        prog="cvsuite common",
        description=__doc__,
        epilog=EPILOG,
        argv=argv,
    )
