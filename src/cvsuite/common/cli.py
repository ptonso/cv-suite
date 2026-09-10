from __future__ import annotations

from cvsuite.branch_cli import run_generic_branch


def main(argv: list[str] | None = None) -> int:
    return run_generic_branch(
        package_name=__package__ or "cvsuite.common",
        prog="cvsuite common",
        description="Shared FM runtime, contracts, and cache-management commands.",
        argv=argv,
    )
