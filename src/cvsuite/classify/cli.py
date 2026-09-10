"""Class branch: ingest image datasets, optionally transform them, and write class directories."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

from ..branch_cli import apply_transform_output_hints, build_parser, parse_help, result_to_exit_code, split_rest
from .core import io

OUTPUT_PKG = f"{__package__}.output.commands"
TRANSFORMS_PKG = f"{__package__}.transform.commands"
PROG = "cvsuite class"


def _iter_output_commands() -> dict[str, str]:
    pkg = importlib.import_module(OUTPUT_PKG)
    out: dict[str, str] = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.ispkg:
            continue
        out[mod.name.replace("_", "-")] = mod.name
    return out


def _iter_transform_commands() -> dict[str, str]:
    out: dict[str, str] = {}
    transforms_pkg = importlib.import_module(TRANSFORMS_PKG)
    for mod in pkgutil.iter_modules(transforms_pkg.__path__):
        if mod.ispkg:
            continue
        out[mod.name.replace("_", "-")] = mod.name
    return out


COMMANDS = _iter_output_commands()
TRANSFORM_COMMANDS = _iter_transform_commands()


def _print_usage() -> None:
    commands = ", ".join(sorted(COMMANDS)) or "none"
    transforms = ", ".join(sorted(TRANSFORM_COMMANDS)) or "none"
    print(f"usage: {PROG} <src> [ingest opts] [<transform> <transform args>] <command> [args]")
    print("ingest opts:")
    print("  --from {auto,images,unstructured,multi-class}")
    print("formats: auto, images, unstructured, multi-class")
    print(f"output commands: {commands}")
    print(f"transforms: {transforms}")


def _dispatch_help(argv: list[str]) -> int:
    """Route `--help` to the most specific transform/command parser named in argv."""
    help_pos = next(i for i, t in enumerate(argv) if t in {"-h", "--help"})
    before = [t for t in argv[:help_pos] if t in COMMANDS or t in TRANSFORM_COMMANDS]
    known = [t for t in argv if t in COMMANDS or t in TRANSFORM_COMMANDS]
    target = before[-1] if before else (known[0] if known else None)
    if target in TRANSFORM_COMMANDS:
        module = importlib.import_module(f"{TRANSFORMS_PKG}.{TRANSFORM_COMMANDS[target]}")
        return parse_help(build_parser(module, prog=f"{PROG} {target}"), ["-h"])
    if target in COMMANDS:
        module = importlib.import_module(f"{OUTPUT_PKG}.{COMMANDS[target]}")
        return parse_help(build_parser(module, prog=f"{PROG} {target}"), ["-h"])
    _print_usage()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if any(token in {"-h", "--help"} for token in argv):
        return _dispatch_help(argv)

    base = argparse.ArgumentParser(add_help=False, prog=PROG, allow_abbrev=False)
    base.add_argument("src", nargs="?")
    base.add_argument("--from", dest="from_fmt", choices=["auto", "images", "unstructured", "multi-class"], default="auto")

    parsed, rest = base.parse_known_args(argv)
    if not parsed.src:
        _print_usage()
        return 1

    transform_name, transform_args, command_name, command_args = split_rest(rest, COMMANDS, TRANSFORM_COMMANDS)
    if not command_name or command_name not in COMMANDS:
        _print_usage()
        raise SystemExit(f"Unknown command: {command_name!r}" if command_name else "Missing output command.")

    command_module = importlib.import_module(f"{OUTPUT_PKG}.{COMMANDS[command_name]}")
    command_parser = build_parser(command_module, prog=f"{PROG} {command_name}")
    if any(token in {"-h", "--help"} for token in command_args):
        parse_help(command_parser, command_args)
        return 0
    parsed_command = command_parser.parse_args(command_args)

    source_path = Path(parsed.src)
    if transform_name == "sample" and io.looks_like_annotated_dataset(source_path):
        raise SystemExit("For labeled datasets, use `cvsuite label sample` instead.")

    dataset = io.ingest(source_path, from_hint=parsed.from_fmt)

    if transform_name:
        mod_name = TRANSFORM_COMMANDS[transform_name]
        transform_module = importlib.import_module(f"{TRANSFORMS_PKG}.{mod_name}")
        transform_parser = build_parser(transform_module, prog=f"{PROG} {transform_name}")
        if any(token in {"-h", "--help"} for token in transform_args):
            parse_help(transform_parser, transform_args)
            return 0
        parsed_transform = transform_parser.parse_args(transform_args)
        setattr(parsed_transform, "source_path", source_path)
        apply_transform_output_hints(PROG, transform_name, parsed_transform, command_name, parsed_command)
        dataset = transform_module.run(dataset, parsed_transform)

    result = command_module.run(dataset, parsed_command)
    return result_to_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
