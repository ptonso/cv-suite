"""Class-folder datasets: sort images into class folders with zero-shot classifiers, sample, split, and rewrite class trees."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from functools import partial
from pathlib import Path

from ..branch_cli import build_parser, help_target, parse_help, print_branch_help, result_to_exit_code, run_chain, split_rest
from ..common.io.common import remove_dirs
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


USAGE = f"usage: {PROG} <src> [ingest opts] [<transform> <transform args>]... <command> [args]"
EXAMPLES = [
    'cvsuite class ./images infer --provider clip --prompt "good,bad" to-class-dir ./sorted',
    "cvsuite class ./sorted sample --max-n-per-class 100 split --val-frac 0.2 to-class-dir ./subset",
]
SUMMARY = (
    "Reads an image folder or a class-folder tree (one folder per class), runs the transforms left to right on\n"
    "one in-memory dataset, and writes class folders with `to-class-dir`. For annotated datasets use\n"
    "`cvsuite label`. An output refuses a destination that already holds files unless --overwrite."
)


def build_ingest_parser() -> argparse.ArgumentParser:
    base = argparse.ArgumentParser(add_help=False, prog=PROG, allow_abbrev=False, usage=argparse.SUPPRESS)
    base.add_argument("src", nargs="?", help="Source folder.")
    group = base.add_argument_group("ingest options")
    group.add_argument(
        "--from",
        dest="from_fmt",
        choices=["auto", "images", "unstructured", "multi-class"],
        default="auto",
        help="Source format (default auto). images: a plain image folder; unstructured: one folder per class; "
        "multi-class: class folders where one image sits in several, merged by file name.",
    )
    return base


def _print_usage() -> None:
    print_branch_help(
        USAGE,
        SUMMARY,
        ingest=build_ingest_parser(),
        transforms={name: partial(_transform_module, name) for name in TRANSFORM_COMMANDS},
        commands={name: partial(_output_module, name) for name in COMMANDS},
        examples=EXAMPLES,
        hint=f"flags of one command: {PROG} <src> <command> --help (any path works as <src>)",
    )


def _transform_module(name: str):
    return importlib.import_module(f"{TRANSFORMS_PKG}.{TRANSFORM_COMMANDS[name]}")


def _output_module(name: str):
    return importlib.import_module(f"{OUTPUT_PKG}.{COMMANDS[name]}")


def _dispatch_help(argv: list[str]) -> int:
    """Route `--help` to the most specific transform/command parser named in argv."""
    target = help_target(argv, [*COMMANDS, *TRANSFORM_COMMANDS])
    if target is None:
        _print_usage()
        return 0
    name, sub = target
    module = _transform_module(name) if name in TRANSFORM_COMMANDS else _output_module(name)
    return parse_help(build_parser(module, prog=f"{PROG} {name}"), [*sub, "-h"])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if any(token in {"-h", "--help"} for token in argv):
        return _dispatch_help(argv)

    base = build_ingest_parser()
    parsed, rest = base.parse_known_args(argv)
    if not parsed.src:
        _print_usage()
        return 1

    transforms = {name: partial(_transform_module, name) for name in TRANSFORM_COMMANDS}
    steps, command_name, command_args = split_rest(rest, transforms, COMMANDS, prog=PROG)
    if command_name is None:
        _print_usage()
        raise SystemExit("Missing output command.")

    command_module = importlib.import_module(f"{OUTPUT_PKG}.{COMMANDS[command_name]}")
    parsed_command = build_parser(command_module, prog=f"{PROG} {command_name}").parse_args(command_args)

    source_path = Path(parsed.src)
    if any(step.name == "sample" for step in steps) and io.looks_like_annotated_dataset(source_path):
        raise SystemExit("For labeled datasets, use `cvsuite label sample` instead.")
    for step in steps:
        setattr(step.args, "source_path", source_path)

    cleanup: list[str] = []
    try:
        dataset = io.ingest(source_path, from_hint=parsed.from_fmt)
        dataset = run_chain(dataset, steps, cleanup=cleanup)
        return result_to_exit_code(command_module.run(dataset, parsed_command))
    finally:
        remove_dirs(cleanup)


if __name__ == "__main__":
    raise SystemExit(main())
