"""Filter namespace with subcommands for training logistic configs and applying filters."""

from __future__ import annotations

import argparse

from . import apply, logistic_opt


def attach(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="filter_command", metavar="FILTER_OP", required=True)

    commands = {
        "train-logistic": logistic_opt,
        "apply": apply,
    }
    for name, module in commands.items():
        help_text = (module.__doc__ or "").strip()
        subparser = subparsers.add_parser(name, help=help_text, description=help_text)
        attach = getattr(module, "attach", None)
        if attach is not None:
            attach(subparser)
        subparser.set_defaults(_filter_run=getattr(module, "run"))


def run(ds, args: argparse.Namespace):
    func = getattr(args, "_filter_run", None)
    if func is None:
        raise ValueError("Missing filter subcommand.")
    return func(ds, args)
