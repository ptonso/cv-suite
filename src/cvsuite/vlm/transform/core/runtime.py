from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from cvsuite.common.core import FMRequest, VQA
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import FMRunner
from cvsuite.common.fm.providers.registry import (
    allowed_model_ids_for_provider,
    default_model_id_for_provider,
    format_available_providers,
    provider_choices_for_family,
    resolve_provider_spec,
)

from ...core import io

MODEL_RUNTIME_IDS = {"minicpm-v": "minicpm_v"}
SUPPORTED_LOCAL_PROVIDERS = ("qwen", "paligemma", "llava", "blip", "cogvlm", "internvl", "minicpm-v")
SUPPORTED_MODELS = SUPPORTED_LOCAL_PROVIDERS
HF_API_BASE = "https://huggingface.co/api/models"
HF_TIMEOUT_SECS = 4.0
_SIZE_RE = re.compile(r"-(\d+(?:\.\d+)?)B-", re.IGNORECASE)
_NATURAL_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ModelCatalogSpec:
    author: str
    search: str
    prefixes: tuple[str, ...]
    include_substrings: tuple[str, ...]
    exclude_substrings: tuple[str, ...]
    default_model_id: str
    max_ids: int | None = None


MODEL_CATALOG_SPECS = {
    "qwen": ModelCatalogSpec(
        author="Qwen",
        search="Qwen",
        prefixes=("Qwen/Qwen2.5-VL-", "Qwen/Qwen3-VL-"),
        include_substrings=("-Instruct",),
        exclude_substrings=("-AWQ", "-GPTQ", "-GGUF"),
        default_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    ),
    "paligemma": ModelCatalogSpec(
        author="google",
        search="paligemma",
        prefixes=("google/paligemma-",),
        include_substrings=(),
        exclude_substrings=("-gguf",),
        default_model_id="google/paligemma-3b-ft-ocrvqa-448",
    ),
    "llava": ModelCatalogSpec(
        author="llava-hf",
        search="llava",
        prefixes=("llava-hf/",),
        include_substrings=("-hf",),
        exclude_substrings=("-int4", "video", "critic"),
        default_model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        max_ids=12,
    ),
    "blip": ModelCatalogSpec(
        author="Salesforce",
        search="blip-vqa",
        prefixes=("Salesforce/blip-vqa-",),
        include_substrings=(),
        exclude_substrings=("-onnx",),
        default_model_id="Salesforce/blip-vqa-base",
        max_ids=8,
    ),
    "cogvlm": ModelCatalogSpec(
        author="zai-org",
        search="cogvlm",
        prefixes=("zai-org/cogvlm",),
        include_substrings=(),
        exclude_substrings=("-int4", "-gguf", "video", "grounding"),
        default_model_id="zai-org/cogvlm2-llama3-chat-19B",
        max_ids=8,
    ),
    "internvl": ModelCatalogSpec(
        author="OpenGVLab",
        search="InternVL2_5",
        prefixes=("OpenGVLab/InternVL2_5-",),
        include_substrings=(),
        exclude_substrings=("-awq", "-gptq", "-gguf"),
        default_model_id="OpenGVLab/InternVL2_5-1B",
        max_ids=10,
    ),
    "minicpm-v": ModelCatalogSpec(
        author="openbmb",
        search="MiniCPM",
        prefixes=("openbmb/MiniCPM-V-", "openbmb/MiniCPM-Llama3-V-"),
        include_substrings=(),
        exclude_substrings=("-int4", "-gguf", "-awq", "-gptq", "MiniCPM-o"),
        default_model_id="openbmb/MiniCPM-V-4",
        max_ids=10,
    ),
}


class ModelCatalogResolutionError(RuntimeError):
    pass


def normalize_model_name(model: str | None) -> str:
    return str(model or "").strip().lower().replace("_", "-")


def default_model_checkpoint(model: str | None) -> str:
    return MODEL_CATALOG_SPECS[normalize_model_name(model)].default_model_id


def runtime_model_name(model: str | None) -> str:
    normalized = normalize_model_name(model)
    return MODEL_RUNTIME_IDS.get(normalized, normalized)


def _model_namespace(spec: ModelCatalogSpec) -> str | None:
    namespaces = {prefix.split("/", 1)[0] for prefix in spec.prefixes if "/" in prefix}
    if len(namespaces) != 1:
        return None
    return next(iter(namespaces))


def _bare_model_prefixes(spec: ModelCatalogSpec) -> tuple[str, ...]:
    bare_prefixes = []
    for prefix in spec.prefixes:
        leaf_prefix = prefix.split("/", 1)[1] if "/" in prefix else prefix
        if leaf_prefix:
            bare_prefixes.append(leaf_prefix)
    if bare_prefixes:
        return tuple(dict.fromkeys(bare_prefixes))

    default_leaf = spec.default_model_id.split("/", 1)[1] if "/" in spec.default_model_id else spec.default_model_id
    family = default_leaf.split("-", 1)[0].strip()
    return (f"{family}-",) if family else ()


def normalize_model_checkpoint(model: str | None, model_id: str | None) -> str:
    raw_model_id = str(model_id or "").strip()
    if not raw_model_id or "/" in raw_model_id:
        return raw_model_id

    spec = MODEL_CATALOG_SPECS[normalize_model_name(model)]
    namespace = _model_namespace(spec)
    if namespace is None:
        return raw_model_id

    bare_prefixes = _bare_model_prefixes(spec)
    if not any(raw_model_id.startswith(prefix) for prefix in bare_prefixes):
        return raw_model_id
    return f"{namespace}/{raw_model_id}"


def _natural_key(text: str) -> tuple[object, ...]:
    parts = _NATURAL_RE.split(text.lower())
    out: list[object] = []
    for part in parts:
        if not part:
            continue
        out.append(int(part) if part.isdigit() else part)
    return tuple(out)


def _model_sort_key(model_id: str) -> tuple[object, ...]:
    size_match = _SIZE_RE.search(model_id)
    if size_match is not None:
        return (0, float(size_match.group(1)), *_natural_key(model_id))
    return (1, *_natural_key(model_id))


def _catalog_url(spec: ModelCatalogSpec) -> str:
    params = {
        "author": spec.author,
        "search": spec.search,
        "sort": "lastModified",
        "direction": "-1",
        "limit": "100",
    }
    return f"{HF_API_BASE}?{urlencode(params)}"


def _fetch_hf_model_ids(spec: ModelCatalogSpec) -> tuple[str, ...]:
    request = Request(
        _catalog_url(spec),
        headers={
            "Accept": "application/json",
            "User-Agent": "cvsuite/0.0.1",
        },
    )
    with urlopen(request, timeout=HF_TIMEOUT_SECS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    ids: list[str] = []
    for item in payload:
        model_id = str(item.get("id") or "").strip()
        if not any(model_id.startswith(prefix) for prefix in spec.prefixes):
            continue
        if any(token not in model_id for token in spec.include_substrings):
            continue
        lowered = model_id.lower()
        if any(token.lower() in lowered for token in spec.exclude_substrings):
            continue
        ids.append(model_id)

    deduped = sorted(set(ids), key=_model_sort_key)
    if spec.max_ids is not None:
        deduped = deduped[: spec.max_ids]
    if not deduped:
        raise RuntimeError(f"No matching models returned from Hugging Face for search={spec.search!r}.")
    return tuple(deduped)


@lru_cache(maxsize=len(SUPPORTED_LOCAL_PROVIDERS))
def resolve_supported_model_catalog(model: str) -> tuple[tuple[str, ...], str]:
    normalized = normalize_model_name(model)
    spec = MODEL_CATALOG_SPECS[normalized]
    try:
        return _fetch_hf_model_ids(spec), "live Hugging Face query"
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelCatalogResolutionError(f"Failed to resolve {normalized} model ids from Hugging Face: {exc}") from exc


def render_model_catalog(model: str | None = None) -> str:
    selected = None if model is None else normalize_model_name(model)
    model_names = None if selected is None else (selected,)
    return format_available_providers("vlm", provider_names=model_names, prefer_aliases=True)


def render_model_help_hint() -> str:
    providers = ", ".join(provider_choices_for_family("vlm"))
    return "\n".join(
        [
            f"Providers: {providers}",
            "Precision values: fp32, fp16, bf16, nf4",
            "With `--device auto`, supported Hugging Face VLM backends use `device_map=\"auto\"` and infer GPU/CPU budgets for CPU RAM then disk fallback.",
            "Use `--max-gpu-memory <VALUE>` with `--device auto` to override the inferred per-visible-GPU budget.",
            "Managed `--device auto` currently supports fp32, fp16, and bf16. NF4 currently requires `--device cuda`.",
            "When managed `--device auto` is active, the runtime auto-enables the expandable-segments allocator hint unless you already configured one.",
            "NF4 auto-installs bitsandbytes into the managed model venv on first use.",
        ]
    )


def _append_epilog(parser: argparse.ArgumentParser, text: str) -> None:
    existing = (parser.epilog or "").rstrip()
    parser.epilog = text if not existing else f"{existing}\n\n{text}"


def attach_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", "--model", dest="provider", required=True, help="VLM provider id.")
    parser.add_argument(
        "--model-id",
        default=None,
        help=(
            "Optional backend checkpoint id. "
            "Defaults depend on --provider; inspect known ids with "
            "'cvsuite vlm --provider <provider> -h'."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "gpu", "cuda"], default="auto", help="Execution device preference.")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size.")
    parser.add_argument(
        "--max-gpu-memory",
        default=None,
        help=(
            "Optional per-visible-GPU memory cap such as 14GiB or 12000MiB. "
            "Requires --device auto and overrides the inferred per-visible-GPU budget for managed CPU RAM then disk fallback."
        ),
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        help=(
            "Numerical precision hint. "
            "Supported values: fp32, fp16, bf16, nf4. "
            "NF4 is available for VLM wrappers and auto-installs bitsandbytes into the managed model venv on first use. "
            "Managed --device auto currently supports fp32, fp16, and bf16; use --device cuda for NF4."
        ),
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional model-specific YAML config.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional model cache/weights directory.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any resumable FM run state and force a fresh model pass.")
    parser.add_argument(
        "--model-arg",
        action="append",
        default=[],
        metavar="key=value",
        help="Extra model argument forwarded into FMRequest.model_args.",
    )
    _append_epilog(parser, render_model_help_hint())


def _validate_api_runtime_args(args: argparse.Namespace) -> None:
    if str(getattr(args, "provider", "")).strip().lower() != "openrouter":
        return
    if getattr(args, "max_gpu_memory", None):
        raise SystemExit("`--max-gpu-memory` is not supported for API provider `openrouter`.")
    if getattr(args, "config", None) is not None:
        raise SystemExit("`--config` is not supported for API provider `openrouter` in v1.")
    if getattr(args, "weights", None) is not None:
        raise SystemExit("`--weights` is not supported for API provider `openrouter`.")
    if str(getattr(args, "device", "auto")) != "auto":
        raise SystemExit("`--device` is only supported with its default `auto` value for API provider `openrouter`.")
    if str(getattr(args, "precision", "fp32")) != "fp32":
        raise SystemExit("`--precision` is only supported with its default `fp32` value for API provider `openrouter`.")


def invoke_model(dataset: VisionDataset, args: argparse.Namespace, *, prompt: str = "") -> VisionDataset:
    max_gpu_memory = str(getattr(args, "max_gpu_memory", "") or "").strip() or None
    if max_gpu_memory is not None and str(args.device) != "auto":
        raise SystemExit("`--max-gpu-memory` requires `--device auto`.")
    provider_arg = str(getattr(args, "provider", "") or "").strip()
    if not provider_arg:
        raise SystemExit("`--provider` is required for `cvsuite vlm`.")
    _validate_api_runtime_args(args)
    model_args = io.parse_model_args(args.model_arg)
    spec = resolve_provider_spec(provider_arg, family="vlm")
    canonical_provider = spec.provider_name
    raw_model_id = str(getattr(args, "model_id", "") or "").strip()
    if canonical_provider == "openrouter":
        model_id = raw_model_id or default_model_id_for_provider(canonical_provider, family="vlm")
        allowed_ids = allowed_model_ids_for_provider(canonical_provider, family="vlm")
        if model_id not in allowed_ids:
            raise SystemExit(
                f"`--model-id {model_id}` is not supported for provider `openrouter` family `vlm`. "
                f"Allowed values: {', '.join(allowed_ids)}."
            )
    else:
        model_id = normalize_model_checkpoint(provider_arg, raw_model_id or default_model_id_for_provider(canonical_provider, family="vlm"))
    if model_id:
        model_args["model_id"] = model_id
    dataset.fm_request = FMRequest(
        task="vlm",
        provider=runtime_model_name(canonical_provider),
        prompt=str(prompt or ""),
        config_path=args.config,
        device=str(args.device),
        precision=str(args.precision),
        max_gpu_memory=max_gpu_memory,
        batch_size=int(args.batch),
        weights_dir=args.weights,
        model_args=model_args,
        meta={"provider_family": "vlm"},
    )
    return FMRunner.from_dataset(dataset).run(dataset, no_resume=bool(getattr(args, "no_resume", False)))


def has_pending_vqas(dataset: VisionDataset) -> bool:
    for record in dataset.records:
        for qa in record.vqas:
            if (qa.question or "").strip() and not (qa.answer and qa.model):
                return True
    return False


def seed_caption(dataset: VisionDataset) -> None:
    if any(record.vqas for record in dataset.records):
        raise SystemExit("caption expects an image-only dataset with no existing questions.")
    for record in dataset.records:
        record.vqas.append(VQA(question=io.CAPTION_PROMPT, answer="", meta={"source": "caption"}))


def seed_ask(dataset: VisionDataset, prompt: str) -> None:
    questions = io.load_prompt_spec(prompt)
    for record in dataset.records:
        pending = {
            (
                qa.question.strip(),
                None if not isinstance(qa.meta, dict) else qa.meta.get("label"),
                None if not isinstance(qa.meta, dict) else qa.meta.get("source"),
            )
            for qa in record.vqas
            if (qa.question or "").strip() and not (qa.answer and qa.model)
        }
        for question, label in questions:
            key = (question, label, "ask")
            if key in pending:
                continue
            meta = {"source": "ask"}
            if label is not None:
                meta["label"] = label
            record.vqas.append(VQA(question=question, answer="", meta=meta))
            pending.add(key)
