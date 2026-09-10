"""Run a text-to-image generation request."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core import runtime


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help=(
            "Repeatable raw prompt string or YAML/JSON prompt file path used for image generation. "
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
    runtime.attach_runtime_args(parser, family="create")
    parser.epilog = (
        runtime.format_available_family_providers("create")
        + "\n\n"
        "Examples:\n"
        "  cvsuite gen create --provider flux --prompt \"a red fox in snow\" to-dst ./out.png\n"
        "  cvsuite gen create --provider openrouter --model-id google/gemini-2.5-flash-image --prompt prompts.yaml --prompt \"night city skyline\" to-dst ./out-dir"
    )


def run(args: argparse.Namespace):
    from cvsuite.gen import core

    prompts = core.parse_prompt_items(getattr(args, "prompt", []), caller_cwd=Path.cwd())
    raw_num_images = getattr(args, "num_images", 1)
    num_images = 1 if raw_num_images is None else int(raw_num_images)
    if num_images < 1:
        raise SystemExit("`--num-images` must be >= 1.")
    dataset = core.build_create_dataset(prompts, root=Path.cwd(), num_images=num_images)
    return runtime.invoke_model(dataset, args, prompt=prompts)
