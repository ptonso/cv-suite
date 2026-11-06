"""Vision-language data: caption images or answer questions about them with VLMs, then write JSON, shards or VQA-style manifests."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from functools import partial
from pathlib import Path

from ..branch_cli import MODEL_NOTE, build_parser, help_target, parse_help, print_branch_help, result_to_exit_code, run_chain, split_rest
from ..common.io.common import remove_dirs
from .core import router

OUTPUT_PKG = f"{__package__}.output.commands"
TRANSFORMS_PKG = f"{__package__}.transform.commands"
PROG = "cvsuite vlm"


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


USAGE = f"usage: {PROG} <src> [ingest opts] [<transform>] <command> [args]"
EXAMPLES = [
    "cvsuite vlm ./images caption --provider qwen to-json ./captions",
    'cvsuite vlm ./images/dog.jpg ask --prompt "How many dogs are in this image?" --provider qwen to-json ./answers',
    "cvsuite vlm --provider qwen -h",
]
SUMMARY = (
    "Reads an image folder or a vision-language dataset, runs at most one VLM transform, and writes the result\n"
    "with exactly one output command. Without a transform it converts between the formats. An output refuses a\n"
    "destination that already holds files unless --overwrite."
)


def build_ingest_parser() -> argparse.ArgumentParser:
    base = argparse.ArgumentParser(add_help=False, prog=PROG, allow_abbrev=False, usage=argparse.SUPPRESS)
    base.add_argument("src", nargs="?", help="Source image, folder or dataset.")
    group = base.add_argument_group("ingest options")
    group.add_argument(
        "--from",
        dest="from_fmt",
        choices=["auto", "images", "json", "shards", "vqa-style"],
        default="auto",
        help="Source format (default auto). images: image-only folder; json: flat folder of image + JSON pairs; "
        "shards: TAR shards of image + JSON pairs; vqa-style: normalized VQA manifest.",
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
        notes=[MODEL_NOTE],
        hint=f"flags of one command: {PROG} <src> <command> --help (any path works as <src>)\n"
        f"provider catalog (models, checkpoints, precision): {PROG} --provider <provider> -h",
    )


def _print_model_help(provider_name: str) -> None:
    from .transform.core import runtime as transform_runtime

    normalized = transform_runtime.normalize_model_name(provider_name)
    try:
        spec = transform_runtime.resolve_provider_spec(normalized, family="vlm")
    except ValueError:
        supported = ", ".join(transform_runtime.provider_choices_for_family("vlm"))
        raise SystemExit(f"Unknown VLM provider {provider_name!r}. Expected one of: {supported}.")
    display_name = next(iter(spec.aliases), spec.provider_name)

    print(f"usage: {PROG} <transform> --provider {display_name} [--model-id MODEL_ID] ...")
    print(transform_runtime.render_model_catalog(display_name))
    print()
    print(transform_runtime.render_model_help_hint())


def _extract_selected_model(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token in {"--model", "--provider"} and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--model=") or token.startswith("--provider="):
            return token.split("=", 1)[1]
    return None


def _transform_module(name: str):
    return importlib.import_module(f"{TRANSFORMS_PKG}.{TRANSFORM_COMMANDS[name]}")


def _output_module(name: str):
    return importlib.import_module(f"{OUTPUT_PKG}.{COMMANDS[name]}")


def _transform_help(name: str, argv: list[str]) -> int:
    module = importlib.import_module(f"{TRANSFORMS_PKG}.{TRANSFORM_COMMANDS[name]}")
    parser = build_parser(module, prog=f"{PROG} {name}")
    selected_model = _extract_selected_model(argv)
    if selected_model:
        from .transform.core import runtime as transform_runtime

        normalized = transform_runtime.normalize_model_name(selected_model)
        if normalized in transform_runtime.provider_choices_for_family("vlm"):
            catalog = transform_runtime.render_model_catalog(normalized)
            parser.epilog = catalog if not parser.epilog else f"{parser.epilog}\n\n{catalog}"
    return parse_help(parser, ["-h"])


def _dispatch_help(argv: list[str]) -> int:
    """Route `--help` to the most specific provider/transform/command help."""
    selected_model = _extract_selected_model(argv)
    if selected_model is not None and argv and argv[0] in {"--model", "--provider"}:
        _print_model_help(selected_model)
        return 0

    target = help_target(argv, [*COMMANDS, *TRANSFORM_COMMANDS])
    if target is None:
        _print_usage()
        return 0
    name, sub = target
    if name in TRANSFORM_COMMANDS:
        return _transform_help(name, argv)
    return parse_help(build_parser(_output_module(name), prog=f"{PROG} {name}"), [*sub, "-h"])


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
    steps, command_name, command_args = split_rest(rest, transforms, COMMANDS, prog=PROG, max_transforms=1)
    if command_name is None:
        _print_usage()
        raise SystemExit("Missing output command.")

    command_module = importlib.import_module(f"{OUTPUT_PKG}.{COMMANDS[command_name]}")
    parsed_command = build_parser(command_module, prog=f"{PROG} {command_name}").parse_args(command_args)

    cleanup: list[str] = []
    try:
        dataset = router.ingest(Path(parsed.src), from_hint=parsed.from_fmt)
        dataset = run_chain(dataset, steps, cleanup=cleanup)
        return result_to_exit_code(command_module.run(dataset, parsed_command))
    finally:
        remove_dirs(cleanup)


if __name__ == "__main__":
    raise SystemExit(main())
