"""Eda branch: ingest like `label`, apply transforms, and write an exploratory report."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from functools import partial
from pathlib import Path

from ..branch_cli import build_parser, parse_help, result_to_exit_code, run_chain, split_rest
from ..common.io.common import remove_dirs
from ..label import cli as label_cli
from ..label.core import router
from .core import style

OUTPUT_PKG = f"{__package__}.output.commands"
PROG = "cvsuite eda"
INGEST_FORMATS = ["auto", "yolo", "labelme", "coco", "images", "class-dir", "semseg-mask"]


def _iter_output_commands() -> dict[str, str]:
    pkg = importlib.import_module(OUTPUT_PKG)
    return {mod.name.replace("_", "-"): mod.name for mod in pkgutil.iter_modules(pkg.__path__) if not mod.ispkg}


def _is_model_backed(name: str) -> bool:
    return bool(getattr(label_cli.transform_module(name), "FM", False))


COMMANDS = _iter_output_commands()
TRANSFORM_COMMANDS = {name: target for name, target in label_cli.TRANSFORM_COMMANDS.items() if not _is_model_backed(name)}


MODEL_BACKED = sorted(set(label_cli.TRANSFORM_COMMANDS) - set(TRANSFORM_COMMANDS))


def _print_usage() -> None:
    print(f"usage: {PROG} <src> [<src> ...] [ingest opts] [<transform> <transform args>]... to-report <out> [options]")
    print(f"output commands: {', '.join(sorted(COMMANDS))}")
    print(f"transforms: {', '.join(sorted(TRANSFORM_COMMANDS))}")


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
    help_pos = next(i for i, t in enumerate(argv) if t in {"-h", "--help"})
    before = [t for t in argv[:help_pos] if t in COMMANDS or t in TRANSFORM_COMMANDS]
    known = [t for t in argv if t in COMMANDS or t in TRANSFORM_COMMANDS]
    target = before[-1] if before else (known[0] if known else None)
    if target in TRANSFORM_COMMANDS:
        return parse_help(build_parser(label_cli.transform_module(target), prog=f"{PROG} {target}"), ["-h"])
    if target in COMMANDS:
        return parse_help(build_parser(_resolve_command_module(target), prog=f"{PROG} {target}"), ["-h"])
    _print_usage()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if any(token in {"-h", "--help"} for token in argv):
        return _dispatch_help(argv)

    base = argparse.ArgumentParser(add_help=False, prog=PROG, allow_abbrev=False)
    base.add_argument("--from", dest="from_fmt", choices=INGEST_FORMATS, default="auto")
    base.add_argument("--keypoints-file", type=Path)
    base.add_argument("--max-depth", dest="max_depth", type=int, default=-1)
    base.add_argument("--images-dir", dest="images_dir", type=Path, action="append", default=[])
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
