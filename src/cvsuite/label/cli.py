"""Annotated datasets: read YOLO, COCO, LabelMe, image or class folders and masks, auto-label, transform, and write any label format."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from functools import partial
from pathlib import Path

from ..branch_cli import build_parser, help_target, parse_help, print_branch_help, result_to_exit_code, run_chain, split_rest
from ..common.io.common import remove_dirs
from .core import router

OUTPUT_PKG = f"{__package__}.output.commands"
TRANSFORMS_PKG = f"{__package__}.transform"
PROG = "cvsuite label"
TRANSFORM_ALIASES: dict[tuple[str, str], str] = {("remap_classes", "run"): "remap-classes"}
HIDDEN_TRANSFORMS: set[tuple[str, str]] = {
    ("filter", "apply"),
    ("filter", "logistic_opt"),
    ("filter", "threshold"),
    ("ops", "rebox"),
    ("ops", "make_negatives"),
    ("ops", "crop_dets"),
}


def _iter_output_commands() -> dict[str, str]:
    pkg = importlib.import_module(OUTPUT_PKG)
    out: dict[str, str] = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.ispkg:
            continue
        out[mod.name.replace("_", "-")] = mod.name
    return out


def _iter_transform_commands() -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    transforms_pkg = importlib.import_module(TRANSFORMS_PKG)
    for proj in pkgutil.iter_modules(transforms_pkg.__path__):
        if not proj.ispkg:
            continue
        proj_name = proj.name
        try:
            commands_pkg = importlib.import_module(f"{TRANSFORMS_PKG}.{proj_name}.commands")
        except ModuleNotFoundError:
            continue
        for mod in pkgutil.iter_modules(commands_pkg.__path__):
            if mod.ispkg:
                continue
            if (proj_name, mod.name) in HIDDEN_TRANSFORMS:
                continue
            cli_name = TRANSFORM_ALIASES.get((proj_name, mod.name))
            if cli_name is None:
                cli_name = proj_name if mod.name == "run" else f"{proj_name}-{mod.name.replace('_', '-')}"
            out[cli_name] = (proj_name, mod.name)
    return out


COMMANDS = _iter_output_commands()
TRANSFORM_COMMANDS = _iter_transform_commands()


def _resolve_command_module(command_name: str):
    return importlib.import_module(f"{OUTPUT_PKG}.{COMMANDS[command_name]}")


USAGE = "usage: cvsuite label <src> [<src> ...] [ingest opts] [<transform> <transform args>]... <command> [args]"
EXAMPLES = [
    "cvsuite label ./yolo to-coco ./coco",
    'cvsuite label ./images ground --provider gdino --prompt "car,person" split --val-frac 0.2 to-yolo ./out',
    "cvsuite label ./raw select --min-box-px 20 --empty drop sample --count 500 to-results ./preview",
]
SUMMARY = (
    "Reads one or more sources (several sources merge by class name), runs the transforms left to right on one\n"
    "in-memory dataset, and writes it with exactly one output command. Split with `split` before writing, since\n"
    "outputs keep the splits the records carry. An output refuses a destination that already holds files unless\n"
    "--overwrite. `inspect` opens a window; `audit` and `to-results` work without a display."
)


def build_ingest_parser(prog: str) -> argparse.ArgumentParser:
    """The source options shared by `label` and `eda`, rendered by `--help` and used to parse them."""
    base = argparse.ArgumentParser(add_help=False, prog=prog, allow_abbrev=False, usage=argparse.SUPPRESS)
    group = base.add_argument_group("ingest options")
    group.add_argument(
        "--from",
        dest="from_fmt",
        choices=["auto", "yolo", "labelme", "coco", "images", "class-dir", "semseg-mask"],
        default="auto",
        help="Source format (default auto). auto detects it and fails on an ambiguous folder. "
        "images: a plain image folder; class-dir: one folder per class; semseg-mask: images plus mask files.",
    )
    group.add_argument("--keypoints-file", type=Path, help="LabelMe only: optional keypoint ordering spec.")
    group.add_argument("--images-dir", dest="images_dir", type=Path, action="append", default=[], help="COCO only: a folder that holds the listed images. Repeat for several folders.")
    group.add_argument(
        "--max-depth",
        dest="max_depth",
        type=int,
        default=-1,
        help="Limit recursion into source dirs (flat/class-dir readers). "
        "-1 = unlimited (default); 0 = only files directly in the source dir.",
    )
    return base


def _print_usage() -> None:
    print_branch_help(
        USAGE,
        SUMMARY,
        ingest=build_ingest_parser(PROG),
        transforms={name: partial(transform_module, name) for name in TRANSFORM_COMMANDS},
        commands={name: partial(_resolve_command_module, name) for name in COMMANDS},
        examples=EXAMPLES,
        hint=f"flags of one command: {PROG} <src> <command> --help (any path works as <src>; namespaces take a subcommand: {PROG} . ops rebox --help)",
    )


def _split_sources_and_rest(argv: list[str]) -> tuple[list[str], list[str]]:
    srcs: list[str] = []
    for idx, token in enumerate(argv):
        if token in TRANSFORM_COMMANDS or token in COMMANDS:
            return srcs, argv[idx:]
        if token.startswith("-"):
            return srcs, argv[idx:]
        srcs.append(token)
    return srcs, []


def _infer_target_dir(command_name: str, command_args: argparse.Namespace) -> Path | None:
    dst = getattr(command_args, "dst", None)
    if isinstance(dst, Path):
        return dst
    return None


def transform_module(name: str):
    proj, mod_name = TRANSFORM_COMMANDS[name]
    return importlib.import_module(f"{TRANSFORMS_PKG}.{proj}.commands.{mod_name}")


def _dispatch_help(argv: list[str]) -> int:
    """Route `--help` to the most specific transform/command parser named in argv."""
    target = help_target(argv, [*COMMANDS, *TRANSFORM_COMMANDS])
    if target is None:
        _print_usage()
        return 0
    name, sub = target
    module = transform_module(name) if name in TRANSFORM_COMMANDS else _resolve_command_module(name)
    return parse_help(build_parser(module, prog=f"{PROG} {name}"), [*sub, "-h"])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    wants_help = any(token in {"-h", "--help"} for token in argv)

    base = build_ingest_parser(PROG)

    if wants_help:
        return _dispatch_help(argv)

    parsed, remaining = base.parse_known_args(argv)
    src_tokens, rest = _split_sources_and_rest(remaining)
    if not src_tokens:
        _print_usage()
        return 1

    transforms = {name: partial(transform_module, name) for name in TRANSFORM_COMMANDS}
    steps, command_name, command_args = split_rest(rest, transforms, COMMANDS, prog=PROG)
    if command_name is None:
        _print_usage()
        raise SystemExit("Missing output command.")

    command_module = _resolve_command_module(command_name)
    parsed_command = build_parser(command_module, prog=f"{PROG} {command_name}").parse_args(command_args)
    target_dir = _infer_target_dir(command_name, parsed_command)
    if target_dir is not None:
        for step in steps:
            setattr(step.args, "target_dir", target_dir)

    cleanup: list[str] = []
    try:
        dataset = router.ingest_many(
            srcs=[Path(token) for token in src_tokens],
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
