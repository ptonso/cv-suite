from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

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

class Step(NamedTuple):
    name: str
    args: argparse.Namespace
    module: Any


class _ProbeError(Exception):
    pass


class _ProbeParser(argparse.ArgumentParser):
    """A parser whose failures raise, so that a candidate stage can be tried without printing or exiting."""

    def error(self, message: str):
        raise _ProbeError(message)


def build_parser(module: Any, prog: str, parser_class: type[argparse.ArgumentParser] = argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = parser_class(
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
    transforms: dict[str, Callable[[], Any]],
    commands: dict[str, Any],
    *,
    prog: str = "cvsuite",
    max_transforms: int | None = None,
) -> tuple[list[Step], str | None, list[str]]:
    """Cut `rest` into transform stages and the output command.

    A stage ends at the next token naming a transform or the output command, provided the stage's own
    parser accepts the tokens in between. Returns the steps, the output command name and its raw arguments;
    the name is None when `rest` holds no output command.
    """
    steps: list[Step] = []
    i = 0
    while i < len(rest):
        token = rest[i]
        if token in commands:
            _check_steps(steps, max_transforms)
            return steps, token, rest[i + 1 :]
        if token not in transforms:
            raise SystemExit(f"Unrecognized token {token!r}: expected a transform or an output command.")
        module = transforms[token]()
        cuts = [j for j in range(i + 1, len(rest)) if rest[j] in transforms or rest[j] in commands] + [len(rest)]
        first_error = None
        for cut in cuts:
            parser = build_parser(module, prog=f"{prog} {token}", parser_class=_ProbeParser)
            try:
                steps.append(Step(token, parser.parse_args(rest[i + 1 : cut]), module))
            except _ProbeError as exc:
                first_error = first_error or str(exc)
                continue
            i = cut
            break
        else:
            raise SystemExit(f"Cannot parse the arguments of transform {token!r} (token {i + 1} of the command): {first_error}")
    _check_steps(steps, max_transforms)
    return steps, None, []


def _check_steps(steps: list[Step], max_transforms: int | None) -> None:
    if max_transforms is not None and len(steps) > max_transforms:
        raise SystemExit(f"This branch accepts at most {max_transforms} transform(s), got {len(steps)}: {', '.join(s.name for s in steps)}.")
    model_backed = [index for index, step in enumerate(steps) if getattr(step.module, "FM", False)]
    if len(model_backed) > 1:
        raise SystemExit(f"A chain holds at most one model-backed transform, got {len(model_backed)}: {', '.join(steps[i].name for i in model_backed)}.")
    if model_backed and model_backed[0] != 0:
        raise SystemExit(f"The model-backed transform {steps[model_backed[0]].name!r} must come first in the chain.")


def _plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return str(value)


def run_chain(dataset: VisionDataset, steps: list[Step], *, cleanup: list[str]) -> VisionDataset:
    """Run the transforms in order, record each as a step, and collect temporary folders in `cleanup`."""
    _collect_cleanup(dataset, cleanup)
    for step in steps:
        dataset = step.module.run(dataset, step.args)
        _collect_cleanup(dataset, cleanup)
        meta = dict(dataset.meta)
        info = meta.pop("step_info", None)
        args = {k: _plain(v) for k, v in vars(step.args).items() if k not in {"target_dir", "source_path"} and not k.startswith("_")}
        meta["steps"] = [*meta.get("steps", []), {"name": step.name, "args": args, "info": info}]
        dataset.meta = meta
    dataset.meta = {**dataset.meta, "cleanup_dirs": list(cleanup)}
    return dataset


def _collect_cleanup(dataset: VisionDataset, cleanup: list[str]) -> None:
    for item in dataset.meta.get("cleanup_dirs", []):
        if str(item) not in cleanup:
            cleanup.append(str(item))
