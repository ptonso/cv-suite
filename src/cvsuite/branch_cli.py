from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from cvsuite.common.core import VisionDataset


@dataclass(frozen=True)
class CommandSpec:
    cli_name: str
    import_path: str
    rel_parts: tuple[str, ...]


def _package_root(package_name: str) -> Path:
    pkg = importlib.import_module(package_name)
    if hasattr(pkg, "__file__") and pkg.__file__ is not None:
        return Path(pkg.__file__).resolve().parent
    return Path(next(iter(pkg.__path__))).resolve()


def _iter_commands_dirs(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in {"venv", "__pycache__"}]
        current = Path(dirpath)
        if current.name == "commands":
            yield current


def discover_commands(package_name: str) -> list[CommandSpec]:
    root = _package_root(package_name)
    discovered: dict[str, CommandSpec] = {}
    for commands_dir in sorted(_iter_commands_dirs(root)):
        rel_dir = commands_dir.relative_to(root)
        rel_parts = tuple(rel_dir.parts[:-1])
        import_base = ".".join([package_name, *rel_dir.parts])
        for mod in pkgutil.iter_modules([str(commands_dir)]):
            if mod.ispkg:
                continue
            cli_parts = [part.replace("_", "-") for part in rel_parts]
            cli_parts.append(mod.name.replace("_", "-"))
            cli_name = "-".join(part for part in cli_parts if part)
            discovered[cli_name] = CommandSpec(
                cli_name=cli_name,
                import_path=f"{import_base}.{mod.name}",
                rel_parts=rel_parts,
            )
    return [discovered[name] for name in sorted(discovered)]


def build_generic_parser(package_name: str, prog: str, description: str) -> tuple[argparse.ArgumentParser, list[CommandSpec]]:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    specs = discover_commands(package_name)
    if not specs:
        return parser, specs

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for spec in specs:
        module = importlib.import_module(spec.import_path)
        help_text = (module.__doc__ or "").strip()
        subparser = subparsers.add_parser(spec.cli_name, help=help_text, description=help_text)
        attach = getattr(module, "attach", None)
        if attach is not None:
            attach(subparser)
        subparser.set_defaults(_run=getattr(module, "run", None))
    return parser, specs


def result_to_exit_code(result: Any) -> int:
    if result is None or isinstance(result, VisionDataset):
        return 0
    return int(result)


def run_generic_branch(package_name: str, prog: str, description: str, argv: list[str] | None = None) -> int:
    parser, specs = build_generic_parser(package_name, prog=prog, description=description)
    if not specs:
        parser.parse_args(argv)
        parser.print_help()
        return 0

    args = parser.parse_args(argv)
    func = getattr(args, "_run", None)
    if func is None:
        parser.error("No command selected.")
    result = func(args)
    return result_to_exit_code(result)


# ---------------------------------------------------------------------------
# Shared pipeline CLI helpers (used by classify, label, vlm, gen)
# ---------------------------------------------------------------------------

def build_parser(module: Any, prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(module.__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    attach = getattr(module, "attach", None)
    if attach is not None:
        attach(parser)
    return parser


def parse_help(parser: argparse.ArgumentParser, argv: list[str]) -> int:
    try:
        parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        raise
    return 0


def split_rest(
    rest: list[str],
    commands: dict[str, Any],
    transform_commands: dict[str, Any],
) -> tuple[str | None, list[str], str | None, list[str]]:
    if not rest:
        return None, [], None, []
    command_index = next((i for i, t in enumerate(rest) if t in commands), None)
    if command_index is None:
        return None, [], None, rest
    transform_name: str | None = None
    transform_args: list[str] = []
    if command_index > 0:
        if rest[0] not in transform_commands:
            return None, [], None, rest
        transform_name = rest[0]
        transform_args = rest[1:command_index]
    return transform_name, transform_args, rest[command_index], rest[command_index + 1:]


def apply_transform_output_hints(
    prog: str,
    transform_name: str | None,
    parsed_transform: argparse.Namespace | None,
    command_name: str,
    parsed_command: argparse.Namespace,
) -> None:
    if transform_name != "sample" or parsed_transform is None:
        return
    if not getattr(parsed_transform, "hardlink", False):
        return
    if not hasattr(parsed_command, "hardlink"):
        raise SystemExit(
            f"`{prog} sample --hardlink` can't be used with `{command_name}` because "
            "that output command does not support `--hardlink`."
        )
    setattr(parsed_command, "hardlink", True)
