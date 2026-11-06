from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path


PROG = "cvsuite"
PUBLIC_BRANCH_ALIASES = {
    "classify": "class",
}
PUBLIC_TO_INTERNAL_BRANCHES = {
    public: internal
    for internal, public in PUBLIC_BRANCH_ALIASES.items()
}


def _public_branch_name(internal_name: str) -> str:
    alias = PUBLIC_BRANCH_ALIASES.get(internal_name)
    if alias is not None:
        return alias
    return internal_name.replace("_", "-")


def _discover_branches() -> list[str]:
    root = Path(__file__).resolve().parent
    branches: set[str] = set()
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("__"):
            continue
        if (child / "cli.py").exists():
            branches.add(_public_branch_name(child.name))
    return sorted(branches)


EXAMPLES = [
    "cvsuite label ./yolo to-coco ./coco",
    'cvsuite class ./images infer --provider clip --prompt "good,bad" to-class-dir ./sorted',
    "cvsuite prep arrange ./dump ./clean --unzip --exact-dedup --rename-seq",
]


def _branch_summary(public_name: str) -> str:
    """First docstring line of a branch's cli module, read without importing it."""
    internal = PUBLIC_TO_INTERNAL_BRANCHES.get(public_name, public_name.replace("-", "_"))
    source = (Path(__file__).resolve().parent / internal / "cli.py").read_text()
    return (ast.get_docstring(ast.parse(source)) or "").splitlines()[0]


def _print_usage(branches: list[str]) -> None:
    rows = "\n".join(f"  {name:<8}{_branch_summary(name)}" for name in branches)
    print(f"usage: {PROG} <branch> [args]")
    print(f"branches: {', '.join(branches) if branches else 'none'}")
    print(
        "\ncvsuite reads a dataset on disk, runs transforms, and writes the result to disk. The pipeline branches read\n"
        "left to right, and `prep` and `common` are plain subcommands:\n\n"
        f"  {PROG} <branch> <src> [<transform> [args]]... <output-command> [args]\n\n"
        f"branches:\n{rows}\n\n"
        f"Each level has its own help, from general to specific: `{PROG} <branch> --help` lists the transforms and output\n"
        f"commands, and `{PROG} <branch> <src> <command> --help` shows the flags of one command (any path works as <src>).\n\n"
        "examples:\n" + "\n".join(f"  {line}" for line in EXAMPLES)
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    branches = _discover_branches()
    if not argv or argv[0] in {"-h", "--help"}:
        _print_usage(branches)
        return 0

    requested = argv[0].replace("_", "-")
    if requested not in branches:
        _print_usage(branches)
        raise SystemExit(f"Unknown branch: {argv[0]}")

    branch = PUBLIC_TO_INTERNAL_BRANCHES.get(requested, requested.replace("-", "_"))
    module = importlib.import_module(f"{__package__}.{branch}.cli")
    return int(module.main(argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
