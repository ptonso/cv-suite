from __future__ import annotations

import importlib
import inspect
import json
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Generic, Mapping, Sequence, TypeVar

from PIL import Image
import yaml

from cvsuite.common.core import FMRequest, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.providers.base import BaseFMModel, RuntimeContext

TOptions = TypeVar("TOptions")
TJob = TypeVar("TJob")
TRuntime = TypeVar("TRuntime")

GEN_PROMPT_ATTR = "gen_prompt"
GEN_PROMPT_INDEX_ATTR = "gen_prompt_index"
GEN_SAMPLE_INDEX_ATTR = "gen_sample_index"
GEN_OUTPUT_KEY_ATTR = "gen_output_key"
GEN_SOURCE_RECORD_IDX_ATTR = "gen_source_record_idx"
GEN_SOURCE_IMAGE_ATTR = "gen_source_image"
GEN_SOURCE_KEY_ATTR = "gen_source_key"

__all__ = [
    "BaseGenerationModel",
    "GEN_OUTPUT_KEY_ATTR",
    "GEN_PROMPT_ATTR",
    "GEN_PROMPT_INDEX_ATTR",
    "GEN_SAMPLE_INDEX_ATTR",
    "GEN_SOURCE_IMAGE_ATTR",
    "GEN_SOURCE_KEY_ATTR",
    "GEN_SOURCE_RECORD_IDX_ATTR",
    "GeneratedImage",
    "GenerationRuntime",
    "build_generation_runtime_extra",
    "generation_record_triplet",
    "load_generation_runtime",
    "materialize_generated_output",
    "normalize_generated_output",
    "parse_prompt_payload",
]


@dataclass(frozen=True)
class GeneratedImage:
    image: Image.Image | None = None
    path: Path | None = None
    format: str = "PNG"


@dataclass
class GenerationRuntime:
    mode: str
    backend_spec: str
    config: dict[str, Any]
    backend: Any
    outputs_dir: Path


class BaseGenerationModel(BaseFMModel[TOptions, TJob, TRuntime], Generic[TOptions, TJob, TRuntime]):
    supported_generation_modes = frozenset({"create", "edit"})

    def validate_completed_job_ids(
        self,
        dataset: VisionDataset,
        jobs: Sequence[TJob],
        ctx: RuntimeContext[TOptions],
        completed_job_ids: set[str],
    ) -> set[str]:
        if not completed_job_ids:
            return completed_job_ids
        valid: set[str] = set()
        outputs_dir = ctx.work_dir / "images"
        for job in jobs:
            job_id = self.job_id(job, dataset, ctx)
            if job_id not in completed_job_ids:
                continue
            output_key = str(getattr(job, "output_key", "") or "").strip()
            if output_key and not (outputs_dir / f"{output_key}.png").exists():
                continue
            valid.add(job_id)
        return valid

    def validate_generation_request(self, dataset: VisionDataset, request: FMRequest) -> str:
        if str(request.task or "") != "gen":
            raise ValueError(f"{self.model_name} expects FMRequest.task='gen', got {request.task!r}.")
        mode = str(dataset.meta.get("gen_mode") or "create")
        if mode not in self.supported_generation_modes:
            supported = ", ".join(sorted(self.supported_generation_modes))
            raise ValueError(f"Unsupported generation mode {mode!r}. Expected one of: {supported}.")
        return mode

    def build_generated_dataset(
        self,
        dataset: VisionDataset,
        *,
        image_path: Path,
        width: int,
        height: int,
        extra_meta: Mapping[str, object] | None = None,
    ) -> VisionDataset:
        meta = dict(dataset.meta)
        meta["gen_mode"] = str(meta.get("gen_mode") or "result")
        if extra_meta:
            meta.update(dict(extra_meta))
        return VisionDataset(
            records=[Record(image=ImageRecord(path=image_path, width=width, height=height), split="train")],
            root=image_path.parent,
            meta=meta,
            fm_request=dataset.fm_request,
        )

    def build_generation_fm_meta(
        self,
        dataset: VisionDataset,
        *,
        config_path: Path | None,
        weights_dir: Path | None,
        request: FMRequest,
        batch_size: int,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        mode = str(dataset.meta.get("gen_mode") or "create")
        meta: dict[str, object] = {
            "task": "gen",
            "provider": self.model_name,
            "mode": mode,
            "device": request.device,
            "precision": request.precision,
            "batch_size": batch_size,
            "weights": str(weights_dir) if weights_dir else None,
            "config": str(config_path) if config_path else None,
        }
        if extra:
            meta.update(dict(extra))
        return meta


def parse_prompt_payload(raw_prompt: str) -> list[str]:
    text = str(raw_prompt or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return [text]
    if isinstance(payload, list):
        prompts: list[str] = []
        for item in payload:
            if not isinstance(item, str):
                raise ValueError("Gen prompt payload must be a JSON list of strings.")
            prompt = item.strip()
            if not prompt:
                raise ValueError("Gen prompt payload must not contain empty prompt strings.")
            prompts.append(prompt)
        return prompts
    if isinstance(payload, str):
        prompt = payload.strip()
        if not prompt:
            raise ValueError("Gen prompt payload must not be empty.")
        return [prompt]
    raise ValueError("Gen prompt payload must be a raw string or a JSON list of strings.")

def generation_record_triplet(record_idx: int, record: Record) -> tuple[str, int, str]:
    raw_prompt = record.attributes.get(GEN_PROMPT_ATTR)
    prompt = str(raw_prompt or "").strip()
    if not prompt:
        raise ValueError(f"Generation record {record_idx} is missing {GEN_PROMPT_ATTR!r}.")
    raw_prompt_index = record.attributes.get(GEN_PROMPT_INDEX_ATTR, record_idx)
    try:
        prompt_index = int(raw_prompt_index)
    except Exception as exc:
        raise ValueError(
            f"Generation record {record_idx} has invalid {GEN_PROMPT_INDEX_ATTR!r}: {raw_prompt_index!r}"
        ) from exc
    output_key = str(record.attributes.get(GEN_OUTPUT_KEY_ATTR) or "").strip()
    if not output_key:
        raise ValueError(f"Generation record {record_idx} is missing {GEN_OUTPUT_KEY_ATTR!r}.")
    return prompt, prompt_index, output_key


def _options_to_dict(options: Any) -> dict[str, Any]:
    if options is None:
        return {}
    if isinstance(options, Mapping):
        return dict(options)
    if is_dataclass(options):
        return dict(asdict(options))
    if hasattr(options, "__dict__"):
        return {str(key): value for key, value in vars(options).items() if not str(key).startswith("_")}
    raise TypeError(f"Generation options must be dict-like or dataclass-backed, got {type(options).__name__}.")


def _load_generation_config(ctx: RuntimeContext[Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if ctx.config_path is not None and ctx.config_path.exists():
        data = yaml.safe_load(ctx.config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("Generation config YAML must contain a mapping/object at the top level.")
        config.update(data)
    config.update(_options_to_dict(ctx.options))
    return config


def _load_object(spec: str) -> Any:
    text = str(spec or "").strip()
    if not text:
        raise ValueError("Generation backend spec must be a non-empty string.")
    module_name: str
    attr_name: str
    if ":" in text:
        module_name, attr_name = text.split(":", 1)
    else:
        module_name, attr_name = text.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(f"Generation backend target {text!r} is missing attribute {attr_name!r}.") from exc


def _call_factory(factory: Any, **kwargs: Any) -> Any:
    if inspect.isclass(factory):
        target = factory
    elif callable(factory):
        target = factory
    else:
        return factory
    signature = inspect.signature(target)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs:
        return target(**kwargs)
    accepted: dict[str, Any] = {}
    for name in signature.parameters:
        if name in kwargs:
            accepted[name] = kwargs[name]
    return target(**accepted)


def _resolve_backend(
    *,
    backend_aliases: Mapping[str, str],
    config: dict[str, Any],
    dataset: VisionDataset,
    ctx: RuntimeContext[Any],
    mode: str,
    model: Any | None = None,
) -> tuple[str, Any]:
    backend_spec = str(config.get("backend") or "").strip()
    if not backend_spec:
        model_id = str(config.get("model_id") or "").strip().lower()
        if model_id in backend_aliases:
            backend_spec = model_id
    if not backend_spec:
        raise RuntimeError(
            "No generation backend is configured. "
            "Provide `--model-arg backend=module:object`, use a base-specific debug alias such as `prompt-card`, "
            "or set `backend:` in the generation config YAML."
        )
    resolved_spec = backend_aliases.get(backend_spec.lower(), backend_spec)
    target = _load_object(resolved_spec)
    backend = _call_factory(target, config=config, dataset=dataset, ctx=ctx, mode=mode, model=model)
    if not hasattr(backend, "generate_batch"):
        raise TypeError(
            f"Generation backend {resolved_spec!r} must resolve to an object exposing `generate_batch(...)`."
        )
    return backend_spec, backend


def load_generation_runtime(
    *,
    backend_aliases: Mapping[str, str],
    dataset: VisionDataset,
    ctx: RuntimeContext[Any],
    mode: str,
    model: Any | None = None,
) -> GenerationRuntime:
    config = _load_generation_config(ctx)
    backend_spec, backend = _resolve_backend(
        backend_aliases=backend_aliases,
        config=config,
        dataset=dataset,
        ctx=ctx,
        mode=mode,
        model=model,
    )
    outputs_dir = ctx.work_dir / "images"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return GenerationRuntime(
        mode=mode,
        backend_spec=backend_spec,
        config=config,
        backend=backend,
        outputs_dir=outputs_dir,
    )


def normalize_generated_output(output: Any) -> GeneratedImage:
    if isinstance(output, GeneratedImage):
        return output
    if isinstance(output, Image.Image):
        return GeneratedImage(image=output)
    if isinstance(output, (str, Path)):
        return GeneratedImage(path=Path(output))
    raise TypeError(
        "Generation backend outputs must be PIL images, filesystem paths, or GeneratedImage instances."
    )


def materialize_generated_output(result: GeneratedImage, dst: Path) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if result.image is not None:
        image = result.image
        image.save(dst, format=result.format or "PNG")
        return image.size
    if result.path is None:
        raise ValueError("GeneratedImage must include either `image` or `path`.")
    src = Path(result.path)
    if not src.exists():
        raise FileNotFoundError(f"Generation backend returned a missing image path: {src}")
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    with Image.open(dst) as image:
        return image.size


def build_generation_runtime_extra(
    dataset: VisionDataset,
    runtime: GenerationRuntime,
    ctx: RuntimeContext[Any],
) -> dict[str, object]:
    prompts = parse_prompt_payload(ctx.prompt)
    extra: dict[str, object] = {
        "backend": runtime.backend_spec,
        "prompt_count": int(dataset.meta.get("gen_prompt_count", len(prompts))),
        "source_count": int(dataset.meta.get("gen_source_count", len(dataset.records) if runtime.mode == "edit" else 0)),
        "fanout_mode": str(dataset.meta.get("gen_fanout_mode") or ("cartesian" if runtime.mode == "edit" else "prompt-list")),
    }
    if "model_id" in runtime.config and str(runtime.config.get("model_id") or "").strip():
        extra["backend_model_id"] = str(runtime.config["model_id"])
    backend_extra = getattr(runtime.backend, "fm_meta_extra", None)
    if isinstance(backend_extra, Mapping):
        extra.update(dict(backend_extra))
    return extra
