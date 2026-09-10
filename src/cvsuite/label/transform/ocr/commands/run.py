"""Run OCR models over label datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvsuite.common.core import FMRequest
from cvsuite.common.fm.core import FMRunner
from cvsuite.common.fm.providers.registry import format_available_providers

DEFAULT_PROVIDER = "paddleocr"


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", "--model", dest="provider", required=True, help="OCR provider id. `paddleOCR` is accepted as an alias.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="cpu", help="Execution device preference.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for OCR inference.")
    parser.add_argument("--precision", default="fp32", help="Numerical precision hint (fp32/fp16/bf16).")
    parser.add_argument("--config", type=Path, default=None, help="Optional model-specific YAML config.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any resumable FM run state and force a fresh model pass.")
    parser.epilog = format_available_providers("ocr")


def normalize_provider(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "paddleocr":
        return DEFAULT_PROVIDER
    raise ValueError(f"Unsupported OCR provider for `cvsuite label ocr`: {provider!r}")


def run(dataset, args: argparse.Namespace):
    provider_name = normalize_provider(args.provider)
    dataset.fm_request = FMRequest(
        task="ocr",
        provider=provider_name,
        prompt="",
        config_path=args.config,
        device=args.device,
        precision=args.precision,
        batch_size=args.batch,
        model_args={},
        meta={"provider_family": "ocr"},
    )
    return FMRunner.from_dataset(dataset).run(dataset, no_resume=bool(getattr(args, "no_resume", False)))
