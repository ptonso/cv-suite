"""Ops transform namespace with subcommands such as rebox, make-negatives, and crop-dets."""

from __future__ import annotations

import argparse

from . import crop_dets, inverse_crop_dets, make_negatives, rebox


def attach(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="ops_command", metavar="OP", required=True)

    commands = {
        "rebox": rebox,
        "make-negatives": make_negatives,
        "crop-dets": crop_dets,
        "inverse-crop-dets": inverse_crop_dets,
    }
    for name, module in commands.items():
        help_text = (module.__doc__ or "").strip()
        subparser = subparsers.add_parser(name, help=help_text, description=help_text)
        attach = getattr(module, "attach", None)
        if attach is not None:
            attach(subparser)
        subparser.set_defaults(_ops_run=getattr(module, "run"))


def run(ds, args: argparse.Namespace):
    func = getattr(args, "_ops_run", None)
    if func is None:
        raise ValueError("Missing ops subcommand.")
    return func(ds, args)
