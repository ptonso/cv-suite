from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvsuite.common.core import FMRequest
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import FMRunner
from cvsuite.common.fm.providers.registry import (
    allowed_model_ids_for_provider,
    default_model_id_for_provider,
    format_available_providers,
    provider_choices_for_family,
    resolve_provider_spec,
)

_SIZE_ERROR_MARKERS = (
    "size",
    "width",
    "height",
    "resolution",
    "divisible",
    "multiple of",
)


def _iter_exception_chain(exc: BaseException | None):
    seen: set[int] = set()
    pending: list[BaseException] = []
    if exc is not None:
        pending.append(exc)
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        yield current
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)


def _describe_hf_access_error(exc: BaseException, model_id: str) -> str | None:
    gated_repo = False
    forbidden = False
    for current in _iter_exception_chain(exc):
        status_code = getattr(getattr(current, "response", None), "status_code", None)
        message = str(current).strip()
        lowered = message.lower()
        if status_code == 403 or "403 forbidden" in lowered or "403 client error" in lowered:
            forbidden = True
        if "public gated repositories" in lowered or "gated repo" in lowered or "gated repos" in lowered:
            gated_repo = True

    if gated_repo:
        return (
            f"Access to Hugging Face model {model_id!r} was denied. "
            "The resolved token is missing permission for public gated repositories. "
            "Update HF_TOKEN/HUGGINGFACE_HUB_TOKEN permissions or use a non-gated checkpoint."
        )
    if forbidden:
        return (
            f"Access to Hugging Face model {model_id!r} was denied with HTTP 403. "
            "Check the token permissions and confirm that the account can access this repository."
        )
    return None


def parse_model_args(raw_items: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in raw_items:
        text = str(item or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise SystemExit(f"Invalid --model-arg {item!r}. Expected key=value.")
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --model-arg {item!r}. Expected key=value.")
        try:
            out[key] = json.loads(value)
        except Exception:
            out[key] = value
    return out


def format_available_family_providers(family: str) -> str:
    return format_available_providers(family)


def attach_runtime_args(parser: argparse.ArgumentParser, *, family: str | None = None) -> None:
    parser.add_argument("--provider", "--model", dest="provider", required=True, help="Generation provider id.")
    parser.add_argument("--model-id", default=None, help="Optional backend checkpoint id forwarded into model_args.")
    parser.add_argument("--width", type=int, default=None, help="Requested output image width in pixels.")
    parser.add_argument("--height", type=int, default=None, help="Requested output image height in pixels.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu", "cuda"], default="auto", help="Execution device preference.")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size.")
    parser.add_argument(
        "--max-gpu-memory",
        default=None,
        help="Optional per-visible-GPU memory cap such as 14GiB or 12000MiB. Requires --device auto.",
    )
    parser.add_argument("--precision", default="fp32", help="Numerical precision hint.")
    parser.add_argument("--config", type=Path, default=None, help="Optional model-specific YAML config.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore resumable FM state and force a fresh model pass.")
    parser.add_argument(
        "--model-arg",
        action="append",
        default=[],
        metavar="key=value",
        help="Extra model argument forwarded into FMRequest.model_args.",
    )


def _validate_api_runtime_args(args: argparse.Namespace) -> None:
    if str(getattr(args, "provider", "")).strip().lower() != "openrouter":
        return
    if getattr(args, "max_gpu_memory", None):
        raise SystemExit("`--max-gpu-memory` is not supported for API provider `openrouter`.")
    if getattr(args, "config", None) is not None:
        raise SystemExit("`--config` is not supported for API provider `openrouter` in v1.")
    if str(getattr(args, "device", "auto")) != "auto":
        raise SystemExit("`--device` is only supported with its default `auto` value for API provider `openrouter`.")
    if str(getattr(args, "precision", "fp32")) != "fp32":
        raise SystemExit("`--precision` is only supported with its default `fp32` value for API provider `openrouter`.")


def invoke_model(dataset: VisionDataset, args: argparse.Namespace, *, prompt: str | list[str]) -> VisionDataset:
    family = str((dataset.meta or {}).get("gen_mode") or "create")
    provider_name = str(getattr(args, "provider", "") or "").strip()
    if not provider_name:
        raise SystemExit("`--provider` is required for `cvsuite gen`.")
    max_gpu_memory = str(getattr(args, "max_gpu_memory", "") or "").strip() or None
    if max_gpu_memory is not None and str(getattr(args, "device", "auto")) != "auto":
        raise SystemExit("`--max-gpu-memory` requires `--device auto`.")
    _validate_api_runtime_args(args)

    model_args = parse_model_args(list(getattr(args, "model_arg", []) or []))
    for key in ("width", "height"):
        value = getattr(args, key, None)
        if value is None:
            continue
        value = int(value)
        if value < 1:
            raise SystemExit(f"`--{key}` must be >= 1.")
        model_args[key] = value
    model_id = str(getattr(args, "model_id", "") or "").strip()
    if not model_id:
        model_id = default_model_id_for_provider(provider_name, family=family)
    if provider_name == "openrouter":
        allowed_ids = allowed_model_ids_for_provider(provider_name, family=family)
        if model_id not in allowed_ids:
            raise SystemExit(
                f"`--model-id {model_id}` is not supported for provider `openrouter` family `{family}`. "
                f"Allowed values: {', '.join(allowed_ids)}."
            )
    if model_id:
        model_args["model_id"] = model_id

    prompt_payload = json.dumps(list(prompt), ensure_ascii=False) if isinstance(prompt, list) else str(prompt or "")
    resolve_provider_spec(provider_name, family=family)
    dataset.fm_request = FMRequest(
        task="gen",
        provider=provider_name,
        prompt=prompt_payload,
        config_path=getattr(args, "config", None),
        device=str(getattr(args, "device", "auto")),
        precision=str(getattr(args, "precision", "fp32")),
        max_gpu_memory=max_gpu_memory,
        batch_size=int(getattr(args, "batch", 1)),
        model_args=model_args,
        meta={"provider_family": family},
    )
    runner = FMRunner.from_dataset(dataset)
    try:
        return runner.run(dataset, no_resume=bool(getattr(args, "no_resume", False)))
    except Exception as exc:
        hf_error = _describe_hf_access_error(exc, str(model_args.get("model_id") or provider_name))
        if hf_error is not None:
            raise RuntimeError(hf_error) from exc
        text = str(exc).lower()
        if "width" in model_args or "height" in model_args:
            if any(marker in text for marker in _SIZE_ERROR_MARKERS) or isinstance(exc, (RuntimeError, ValueError)):
                raise RuntimeError(
                    f"Provider {provider_name!r} rejected requested size "
                    f"width={model_args.get('width')} height={model_args.get('height')}: {exc}"
                ) from exc
        if ("width" in model_args or "height" in model_args) and any(marker in text for marker in _SIZE_ERROR_MARKERS):
            raise RuntimeError(
                f"Provider {provider_name!r} rejected requested size "
                f"width={model_args.get('width')} height={model_args.get('height')}: {exc}"
            ) from exc
        raise
