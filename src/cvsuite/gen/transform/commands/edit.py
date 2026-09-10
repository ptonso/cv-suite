"""Run a text-guided image edit over a label-style source dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core import runtime

SUPPORTED_INPUTS = ("auto", "yolo", "labelme", "coco", "images", "unstructured")


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("src", type=Path, help="Source image or dataset to edit.")
    parser.add_argument("--from", dest="from_fmt", choices=SUPPORTED_INPUTS, default="auto", help="Input format hint.")
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help=(
            "Repeatable raw prompt string or YAML/JSON prompt file path used for image editing. "
            "Prompt files may be a list of strings, a mapping of id to prompt, "
            "or a list of single-entry id-to-prompt mappings."
        ),
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of output images to generate per prompt/source combination.",
    )
    runtime.attach_runtime_args(parser, family="edit")
    parser.epilog = runtime.format_available_family_providers("edit")


def run(args: argparse.Namespace):
    from cvsuite.gen import core as gen_core
    from ...core import router

    fmt = router.detect_format(args.src, from_hint=args.from_fmt)
    dataset = router.ingest_edit_source(args.src, from_hint=args.from_fmt)
    meta = dict(dataset.meta)
    meta["gen_mode"] = "edit"
    meta["gen_source"] = str(Path(args.src))
    meta["gen_source_format"] = str(fmt)
    meta["source_format"] = str(fmt)
    meta["source_records"] = len(dataset.records)
    meta["source_boxes_total"] = sum(len(record.boxes) for record in dataset.records)
    meta["source_polygons_total"] = sum(len(record.polys) for record in dataset.records)
    meta["source_keypoints_total"] = sum(len(record.kpts) for record in dataset.records)
    dataset.meta = meta
    prompts = gen_core.parse_prompt_items(getattr(args, "prompt", []), caller_cwd=Path.cwd())
    raw_num_images = getattr(args, "num_images", 1)
    num_images = 1 if raw_num_images is None else int(raw_num_images)
    if num_images < 1:
        raise SystemExit("`--num-images` must be >= 1.")
    expanded = gen_core.expand_edit_dataset(dataset, prompts, num_images=num_images)
    return runtime.invoke_model(expanded, args, prompt=prompts)
