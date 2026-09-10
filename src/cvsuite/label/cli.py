"""Label branch: ingest annotated datasets, apply optional transforms, and export results."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

from ..branch_cli import apply_transform_output_hints, build_parser, parse_help, result_to_exit_code, split_rest
from .core import router

OUTPUT_PKG = f"{__package__}.output.commands"
TRANSFORMS_PKG = f"{__package__}.transform"
PROG = "cvsuite label"
TRANSFORM_ALIASES: dict[tuple[str, str], str] = {}
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


def _print_usage() -> None:
    commands = ", ".join(sorted(COMMANDS))
    transforms = ", ".join(sorted(TRANSFORM_COMMANDS)) or "none"
    print(f"usage: {PROG} <src> [<src> ...] [ingest opts] [<transform> <transform args>] <command> [args]")
    print(f"output commands: {commands}")
    print(f"transforms: {transforms}")


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


def _transform_module(name: str):
    proj, mod_name = TRANSFORM_COMMANDS[name]
    return importlib.import_module(f"{TRANSFORMS_PKG}.{proj}.commands.{mod_name}")


def _dispatch_help(argv: list[str]) -> int:
    """Route `--help` to the most specific transform/command parser named in argv."""
    help_pos = next(i for i, t in enumerate(argv) if t in {"-h", "--help"})
    before = [t for t in argv[:help_pos] if t in COMMANDS or t in TRANSFORM_COMMANDS]
    known = [t for t in argv if t in COMMANDS or t in TRANSFORM_COMMANDS]
    target = before[-1] if before else (known[0] if known else None)
    if target in TRANSFORM_COMMANDS:
        return parse_help(build_parser(_transform_module(target), prog=f"{PROG} {target}"), ["-h"])
    if target in COMMANDS:
        return parse_help(build_parser(_resolve_command_module(target), prog=f"{PROG} {target}"), ["-h"])
    _print_usage()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    wants_help = any(token in {"-h", "--help"} for token in argv)

    base = argparse.ArgumentParser(add_help=False, prog=PROG, allow_abbrev=False)
    base.add_argument(
        "--from",
        dest="from_fmt",
        choices=["auto", "yolo", "labelme", "coco", "images", "class-dir", "semseg-mask"],
        default="auto",
    )
    base.add_argument("--keypoints-file", type=Path, help="LabelMe only: optional keypoint ordering spec.")
    base.add_argument(
        "--max-depth",
        dest="max_depth",
        type=int,
        default=-1,
        help="Limit recursion into source dirs (flat/class-dir readers). "
        "-1 = unlimited (default); 0 = only files directly in the source dir.",
    )

    if wants_help:
        return _dispatch_help(argv)

    parsed, remaining = base.parse_known_args(argv)
    src_tokens, rest = _split_sources_and_rest(remaining)
    if not src_tokens:
        _print_usage()
        return 1

    transform_name, transform_args, command_name, command_args = split_rest(rest, COMMANDS, TRANSFORM_COMMANDS)
    if not command_name or command_name not in COMMANDS:
        _print_usage()
        raise SystemExit(f"Unknown command: {command_name!r}" if command_name else "Missing output command.")

    command_module = _resolve_command_module(command_name)
    command_parser = build_parser(command_module, prog=f"{PROG} {command_name}")
    if any(token in {"-h", "--help"} for token in command_args):
        return parse_help(command_parser, command_args)
    parsed_command = command_parser.parse_args(command_args)

    parsed_transform = None
    transform_module = None
    if transform_name:
        proj, mod_name = TRANSFORM_COMMANDS[transform_name]
        transform_module = importlib.import_module(f"{TRANSFORMS_PKG}.{proj}.commands.{mod_name}")
        transform_parser = build_parser(transform_module, prog=f"{PROG} {transform_name}")
        if any(token in {"-h", "--help"} for token in transform_args):
            return parse_help(transform_parser, transform_args)
        parsed_transform = transform_parser.parse_args(transform_args)
        target_dir = _infer_target_dir(command_name, parsed_command)
        if target_dir is not None:
            setattr(parsed_transform, "target_dir", target_dir)
        apply_transform_output_hints(PROG, transform_name, parsed_transform, command_name, parsed_command)

    dataset = router.ingest_many(
        srcs=[Path(token) for token in src_tokens],
        from_hint=parsed.from_fmt,
        keypoints_file=parsed.keypoints_file,
        max_depth=parsed.max_depth,
    )
    if transform_module is not None and parsed_transform is not None:
        dataset = transform_module.run(dataset, parsed_transform)

    result = command_module.run(dataset, parsed_command)
    return result_to_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
