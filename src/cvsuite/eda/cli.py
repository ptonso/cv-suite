"""Dataset exploration: ingest like `label`, narrow or split the data, and write figures and tables into a report folder."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from functools import partial
from pathlib import Path

from ..branch_cli import build_parser, help_target, parse_help, print_branch_help, result_to_exit_code, run_chain, split_rest
from ..common.io.common import remove_dirs
from ..label import cli as label_cli
from ..label.core import router
from .core import style

OUTPUT_PKG = f"{__package__}.output.commands"
PROG = "cvsuite eda"


def _iter_output_commands() -> dict[str, str]:
    pkg = importlib.import_module(OUTPUT_PKG)
    return {mod.name.replace("_", "-"): mod.name for mod in pkgutil.iter_modules(pkg.__path__) if not mod.ispkg}


def _is_model_backed(name: str) -> bool:
    return bool(getattr(label_cli.transform_module(name), "FM", False))


COMMANDS = _iter_output_commands()
TRANSFORM_COMMANDS = {name: target for name, target in label_cli.TRANSFORM_COMMANDS.items() if not _is_model_backed(name)}


MODEL_BACKED = sorted(set(label_cli.TRANSFORM_COMMANDS) - set(TRANSFORM_COMMANDS))


USAGE = f"usage: {PROG} <src> [<src> ...] [ingest opts] [<transform> <transform args>]... to-report <out> [options]"
EXAMPLES = [
    "cvsuite eda ./dataset to-report ./report",
    "cvsuite eda ./visa select --drop-class Background --empty drop split --group-by dir:0 --val-frac 0.15 --test-frac 0.15 to-report ./report --group-by dir:0",
]
SUMMARY = (
    "Reads a dataset like `label` does, optionally narrows or splits it with the transforms below, and writes\n"
    "figures and tables per section into a report folder. The analysis never changes its source and runs no\n"
    "models: write labels with `cvsuite label` first. Needs `pip install \"cv-suite[eda]\"`."
)


def _print_usage() -> None:
    print_branch_help(
        USAGE,
        SUMMARY,
        ingest=label_cli.build_ingest_parser(PROG),
        transforms={name: partial(label_cli.transform_module, name) for name in TRANSFORM_COMMANDS},
        commands={name: partial(_resolve_command_module, name) for name in COMMANDS},
        examples=EXAMPLES,
        hint=f"flags of one command: {PROG} <src> <command> --help (any path works as <src>)",
    )


def _resolve_command_module(name: str):
    return importlib.import_module(f"{OUTPUT_PKG}.{COMMANDS[name]}")


def _split_sources_and_rest(argv: list[str]) -> tuple[list[str], list[str]]:
    sources: list[str] = []
    for index, token in enumerate(argv):
        if token in MODEL_BACKED:
            raise SystemExit(f"`{token}` is a model-backed transform. eda runs no models: write the labels with `cvsuite label` first.")
        if token in TRANSFORM_COMMANDS or token in COMMANDS or token.startswith("-"):
            return sources, argv[index:]
        sources.append(token)
    return sources, []


def _dispatch_help(argv: list[str]) -> int:
    target = help_target(argv, [*COMMANDS, *TRANSFORM_COMMANDS])
    if target is None:
        _print_usage()
        return 0
    name, sub = target
    module = label_cli.transform_module(name) if name in TRANSFORM_COMMANDS else _resolve_command_module(name)
    return parse_help(build_parser(module, prog=f"{PROG} {name}"), [*sub, "-h"])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if any(token in {"-h", "--help"} for token in argv):
        return _dispatch_help(argv)

    base = label_cli.build_ingest_parser(PROG)
    parsed, remaining = base.parse_known_args(argv)
    sources, rest = _split_sources_and_rest(remaining)
    if not sources:
        _print_usage()
        return 1

    transforms = {name: partial(label_cli.transform_module, name) for name in TRANSFORM_COMMANDS}
    steps, command_name, command_args = split_rest(rest, transforms, COMMANDS, prog=PROG)
    if command_name is None:
        _print_usage()
        raise SystemExit("Missing output command.")
    command_module = _resolve_command_module(command_name)
    parsed_command = build_parser(command_module, prog=f"{PROG} {command_name}").parse_args(command_args)

    style.ensure_matplotlib()
    cleanup: list[str] = []
    try:
        dataset = router.ingest_many(
            srcs=[Path(token) for token in sources],
            from_hint=parsed.from_fmt,
            keypoints_file=parsed.keypoints_file,
            max_depth=parsed.max_depth,
            images_dir=parsed.images_dir,
        )
        dataset = run_chain(dataset, steps, cleanup=cleanup)
        return result_to_exit_code(command_module.run(dataset, parsed_command))
    finally:
        remove_dirs(cleanup)


if __name__ == "__main__":
    raise SystemExit(main())
