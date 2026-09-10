"""Gen branch: run generation actions, then materialize the generated image."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from typing import Optional

from ..branch_cli import build_parser, parse_help, result_to_exit_code

ACTIONS_PKG = f"{__package__}.transform.commands"
OUTPUT_PKG = f"{__package__}.output.commands"
PROG = "cvsuite gen"


def _iter_actions() -> dict[str, str]:
    pkg = importlib.import_module(ACTIONS_PKG)
    out: dict[str, str] = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.ispkg:
            continue
        out[mod.name.replace("_", "-")] = mod.name
    return out


def _iter_outputs() -> dict[str, str]:
    pkg = importlib.import_module(OUTPUT_PKG)
    out: dict[str, str] = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.ispkg:
            continue
        out[mod.name.replace("_", "-")] = mod.name
    return out


ACTION_COMMANDS = _iter_actions()
OUTPUT_COMMANDS = _iter_outputs()


def _print_usage() -> None:
    actions = ", ".join(sorted(ACTION_COMMANDS)) or "none"
    outputs = ", ".join(sorted(OUTPUT_COMMANDS)) or "none"
    print(f"usage: {PROG} <action> [action args] <output-command> [args]")
    print(f"actions: {actions}")
    print(f"output commands: {outputs}")


def _split_output(rest: list[str]) -> tuple[list[str], str | None, list[str]]:
    if not rest:
        return [], None, []
    output_index = next((idx for idx, token in enumerate(rest) if token in OUTPUT_COMMANDS), None)
    if output_index is None:
        return rest, None, []
    return rest[:output_index], rest[output_index], rest[output_index + 1:]


def _help_index(tokens: list[str]) -> Optional[int]:
    return next((idx for idx, token in enumerate(tokens) if token in {"-h", "--help"}), None)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        _print_usage()
        return 0

    first = argv[0]
    rest = argv[1:]

    if first in ACTION_COMMANDS:
        action_args, output_name, output_args = _split_output(rest)
        help_idx = _help_index(rest)
        output_idx = next((idx for idx, token in enumerate(rest) if token in OUTPUT_COMMANDS), None)
        if help_idx is not None:
            if output_idx is not None and help_idx > output_idx and output_name is not None:
                output_module = importlib.import_module(f"{OUTPUT_PKG}.{OUTPUT_COMMANDS[output_name]}")
                return parse_help(build_parser(output_module, prog=f"{PROG} {output_name}"), output_args)
            action_module = importlib.import_module(f"{ACTIONS_PKG}.{ACTION_COMMANDS[first]}")
            return parse_help(build_parser(action_module, prog=f"{PROG} {first}"), action_args)

        action_module = importlib.import_module(f"{ACTIONS_PKG}.{ACTION_COMMANDS[first]}")
        action_parser = build_parser(action_module, prog=f"{PROG} {first}")
        parsed_action, remaining = action_parser.parse_known_args(rest)
        if not remaining:
            _print_usage()
            raise SystemExit("Missing output command.")
        output_name = remaining[0]
        output_args = remaining[1:]
        if output_name not in OUTPUT_COMMANDS:
            _print_usage()
            raise SystemExit(f"Unknown output command: {output_name!r}")

        output_module = importlib.import_module(f"{OUTPUT_PKG}.{OUTPUT_COMMANDS[output_name]}")
        output_parser = build_parser(output_module, prog=f"{PROG} {output_name}")
        parsed_output = output_parser.parse_args(output_args)

        dataset = action_module.run(parsed_action)
        result = output_module.run(dataset, parsed_output)
        return result_to_exit_code(result)

    if first in OUTPUT_COMMANDS and any(token in {"-h", "--help"} for token in rest):
        output_module = importlib.import_module(f"{OUTPUT_PKG}.{OUTPUT_COMMANDS[first]}")
        return parse_help(build_parser(output_module, prog=f"{PROG} {first}"), rest)

    _print_usage()
    raise SystemExit(f"Unknown action or output command: {first}")


if __name__ == "__main__":
    raise SystemExit(main())
