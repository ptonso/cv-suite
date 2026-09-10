"""Run FM-backed classification over the dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvsuite.classify.core import io as classify_io
from cvsuite.common.core import FMRequest
from cvsuite.common.fm.core import FMRunner
from cvsuite.common.fm.core import prompt_utils
from cvsuite.common.fm.providers.registry import format_available_providers

DEFAULT_PROVIDER = "clip"


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", "--model", dest="provider", required=True, help="Classification provider id (for example: clip or siglip2).")
    parser.add_argument(
        "--prompt",
        default="",
        help="Label list or JSON/YAML prompt payload. Simple label lists are expanded with --template-prompt.",
    )
    parser.add_argument(
        "--template-prompt",
        default=prompt_utils.DEFAULT_CLASS_TEMPLATE_PROMPT,
        help=(
            "Template used for label-list prompts. "
            f"Must contain {prompt_utils.CLASS_TEMPLATE_TOKEN!r}; "
            f"example: 'a photo with a {prompt_utils.CLASS_TEMPLATE_TOKEN}'."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="cpu", help="Execution device preference.")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--precision", default="fp32", help="Numerical precision hint.")
    parser.add_argument("--config", type=Path, default=None, help="Optional model-specific YAML config.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore resumable FM state and force a fresh model pass.")
    parser.epilog = (
        format_available_providers("classify")
        + "\n\n"
        "Prompt YAML examples:\n"
        "  # Simple label list\n"
        "  - partition_a\n"
        "  - partition_b\n"
        "  - partition_c\n"
        f"  # Each label is injected into --template-prompt using {prompt_utils.CLASS_TEMPLATE_TOKEN}\n"
        f"  # Default template: {prompt_utils.DEFAULT_CLASS_TEMPLATE_PROMPT}\n\n"
        "  # Explicit prompt map\n"
        "  partition_a:\n"
        "    - partition a\n"
        "    - alpha partition\n"
        "  partition_b:\n"
        "    - partition b\n"
        "  partition_c:\n"
        "    - partition c\n\n"
        "Template prompt examples:\n"
        f"  --template-prompt \"a photo of a {prompt_utils.CLASS_TEMPLATE_TOKEN}\"\n"
        f"  --template-prompt \"a photo with a {prompt_utils.CLASS_TEMPLATE_TOKEN}\"\n\n"
        "Notes:\n"
        "  - Simple label lists use --template-prompt.\n"
        "  - Explicit prompt maps are used verbatim."
    )


def run(dataset, args: argparse.Namespace):
    meta = dict(dataset.meta)
    meta.pop(classify_io.CLASSIFY_INGEST_MODE_META, None)
    dataset.meta = meta
    dataset.fm_request = FMRequest(
        task="classify",
        provider=args.provider,
        prompt=args.prompt,
        config_path=args.config,
        device=args.device,
        precision=args.precision,
        batch_size=args.batch,
        model_args={"template_prompt": args.template_prompt},
        meta={"provider_family": "classify"},
    )
    return FMRunner.from_dataset(dataset).run(dataset, no_resume=bool(getattr(args, "no_resume", False)))
