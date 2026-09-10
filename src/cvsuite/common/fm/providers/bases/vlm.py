from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Generic, Sequence, TypeVar

import yaml
from PIL import Image

from cvsuite.common.core import VQA
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BaseFMModel, RuntimeContext

torch = utils.torch

TConfig = TypeVar("TConfig")
TOptions = TypeVar("TOptions")
TRuntime = TypeVar("TRuntime")

__all__ = [
    "BaseVLMModel",
    "HFLoadPlacement",
    "VLMJob",
    "append_answer",
    "build_hf_placement_meta",
    "build_vlm_fm_meta",
    "coerce_generated_text",
    "collect_vlm_jobs",
    "ensure_image_token",
    "load_dataclass_config",
    "load_rgb_image",
    "move_inputs",
    "resolve_device_map",
    "resolve_hf_load_placement",
    "resolve_offload_folder_path",
    "vlm_job_id",
]


@dataclass(frozen=True)
class HFLoadPlacement:
    device_map: str | dict[str, str | int] | None
    max_memory: dict[int | str, int | str] | None = None
    offload_folder: str | None = None
    offload_cap_enabled: bool = False
    managed_device_map_active: bool = False
    memory_budget_source: str = "none"


@dataclass(frozen=True)
class VLMJob:
    record_idx: int
    qa_idx: int | None
    question: str
    label: str | None
    image_path: Path


_ONE_GIB = 1024 ** 3
_MIN_GPU_RESERVE_BYTES = 1 * _ONE_GIB
_MIN_CPU_RESERVE_BYTES = 2 * _ONE_GIB
_GPU_RESERVE_FRACTION = 0.10
_CPU_RESERVE_FRACTION = 0.20


def load_dataclass_config(path: Path | None, config_cls: type[TConfig], overrides: Any | None = None) -> TConfig:
    cfg = config_cls()
    if path and path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict):
            values = {field.name: data.get(field.name, getattr(cfg, field.name)) for field in fields(config_cls)}
            cfg = config_cls(**values)
    if overrides is None:
        return cfg
    values = {field.name: getattr(cfg, field.name) for field in fields(config_cls)}
    for field in fields(type(overrides)):
        value = getattr(overrides, field.name)
        if value is not None:
            values[field.name] = value
    return config_cls(**values)


def parse_questions(raw: str) -> list[tuple[str, str | None]]:
    txt = (raw or "").strip()
    if not txt:
        return []
    try:
        data = json.loads(txt)
    except Exception:
        return [(txt, None)]
    if isinstance(data, list):
        return [(str(item).strip(), None) for item in data if str(item).strip()]
    if isinstance(data, dict):
        out: list[tuple[str, str | None]] = []
        for key, value in data.items():
            label = str(key).strip() or None
            if isinstance(value, list):
                for item in value:
                    question = str(item).strip()
                    if question:
                        out.append((question, label))
            else:
                question = str(value).strip()
                if question:
                    out.append((question, label))
                elif label:
                    out.append((label, label))
        return out
    return [(txt, None)]


def collect_vlm_jobs(dataset: VisionDataset, raw_prompt: str, caller_cwd: Path) -> tuple[list[VLMJob], list[dict[str, str]]]:
    raw_jobs: list[tuple[int, int | None, str, str | None]] = []
    for record_idx, record in enumerate(dataset.records):
        for qa_idx, qa in enumerate(record.vqas):
            question = (qa.question or "").strip()
            if not question:
                continue
            if qa.answer and qa.model:
                continue
            label = None
            if isinstance(qa.meta, dict):
                raw_label = qa.meta.get("label")
                if raw_label is not None and str(raw_label).strip():
                    label = str(raw_label).strip()
            raw_jobs.append((record_idx, qa_idx, question, label))

    if not raw_jobs:
        raw_jobs = [
            (record_idx, None, question, label)
            for record_idx, _record in enumerate(dataset.records)
            for question, label in parse_questions(raw_prompt)
        ]

    jobs: list[VLMJob] = []
    seen_questions: set[tuple[str, str | None]] = set()
    question_meta: list[dict[str, str]] = []
    for record_idx, qa_idx, question, label in raw_jobs:
        key = (question, label)
        if key not in seen_questions:
            seen_questions.add(key)
            question_meta.append({"question": question, "label": label} if label else {"question": question})
        jobs.append(
            VLMJob(
                record_idx=record_idx,
                qa_idx=qa_idx,
                question=question,
                label=label,
                image_path=utils.resolve_path(
                    dataset.records[record_idx].image.path,
                    dataset.root,
                    caller_cwd,
                    allow_root_basename=True,
                ),
            )
        )

    if not jobs:
        raise ValueError("A prompt/question is required in the dataset FM request for this VLM wrapper.")
    return jobs, question_meta


def vlm_job_id(job: VLMJob) -> str:
    if job.qa_idx is not None:
        return f"record:{job.record_idx}:qa:{job.qa_idx}"
    digest = hashlib.sha1(f"{job.question}\0{job.label or ''}".encode("utf-8")).hexdigest()[:12]
    return f"record:{job.record_idx}:prompt:{digest}"


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def append_answer(dataset: VisionDataset, job: VLMJob, answer: str, model_id: str) -> int:
    record = dataset.records[job.record_idx]
    value = str(answer or "").strip()
    if job.qa_idx is not None:
        existing = record.vqas[job.qa_idx]
        meta = dict(existing.meta or {})
        if job.label and "label" not in meta:
            meta["label"] = job.label
        existing.answer = value
        existing.score = None
        existing.model = model_id
        existing.meta = meta
    else:
        meta = {"label": job.label} if job.label else {}
        record.vqas.append(
            VQA(
                question=job.question,
                answer=value,
                score=None,
                model=model_id,
                meta=meta,
            )
        )
    utils.append_once(record.attributes.setdefault("fm_tasks", []), "vlm")
    return job.record_idx


def build_vlm_fm_meta(
    *,
    provider_id: str,
    hf_model_id: str,
    question_meta: list[dict[str, str]],
    ctx,
    precision: utils.ResolvedPrecision | None = None,
    placement: HFLoadPlacement | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    requested_max_gpu_memory = str(ctx.request.max_gpu_memory or "").strip() or None
    placement_meta = build_hf_placement_meta(ctx, placement)
    meta: dict[str, object] = {
        "task": "vlm",
        "provider": provider_id,
        "hf_model": hf_model_id,
        "device": str(ctx.request.device),
        "precision": ctx.request.precision,
        "batch_size": ctx.batch_size,
        "weights": str(ctx.weights_dir) if ctx.weights_dir else None,
        "config": str(ctx.config_path) if ctx.config_path else None,
        "questions": list(question_meta),
    }
    meta.update(placement_meta)
    meta["max_gpu_memory"] = requested_max_gpu_memory
    if precision is not None:
        meta.update(utils.build_precision_meta(precision))
    if extra:
        meta.update(extra)
    return meta


def ensure_image_token(prompt: str) -> str:
    stripped = prompt.lstrip()
    if stripped.startswith("<image>"):
        return prompt
    if not prompt:
        return "<image>"
    return f"<image>\n{prompt}"


def resolve_device_map(pref: str) -> str | dict[str, str]:
    choice = utils.normalize_device_pref(pref)
    if choice == "auto":
        return "auto" if torch.cuda.is_available() else {"": "cpu"}
    if choice == "cuda":
        utils.require_cuda_available(pref)
        return {"": "cuda"}
    return {"": "cpu"}


def resolve_offload_folder_path(stage_dir: Path | None, model_cache: Path) -> Path:
    base_dir = stage_dir if stage_dir is not None else (model_cache / "stage")
    offload_dir = base_dir / "offload"
    offload_dir.mkdir(parents=True, exist_ok=True)
    return offload_dir


def _clamp_budget_bytes(available_bytes: int, reserve_bytes: int) -> int:
    return max(1, int(available_bytes) - int(reserve_bytes))


def _infer_auto_max_memory() -> dict[int | str, int]:
    gpu_count = int(torch.cuda.device_count())
    if gpu_count < 1:
        raise RuntimeError("`--device auto` requires at least one visible CUDA device for managed Hugging Face placement.")

    max_memory: dict[int | str, int] = {}
    for index in range(gpu_count):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except Exception:
            total_bytes = int(torch.cuda.get_device_properties(index).total_memory)
            free_bytes = total_bytes
        reserve_bytes = max(int(total_bytes * _GPU_RESERVE_FRACTION), _MIN_GPU_RESERVE_BYTES)
        max_memory[index] = _clamp_budget_bytes(free_bytes, reserve_bytes)

    cpu_available = utils.available_cpu_memory_bytes()
    cpu_reserve = max(int(cpu_available * _CPU_RESERVE_FRACTION), _MIN_CPU_RESERVE_BYTES)
    max_memory["cpu"] = _clamp_budget_bytes(cpu_available, cpu_reserve)
    return max_memory


def _serialize_max_memory_map(max_memory: dict[int | str, int | str] | None) -> dict[str, int | str] | None:
    if max_memory is None:
        return None
    serialized: dict[str, int | str] = {}
    for key, value in max_memory.items():
        if isinstance(value, int):
            serialized[str(key)] = int(value)
        else:
            serialized[str(key)] = str(value)
    return serialized


def build_hf_placement_meta(ctx, placement: HFLoadPlacement | None) -> dict[str, object]:
    managed_device_map_active = bool(placement and placement.managed_device_map_active)
    effective_max_memory = _serialize_max_memory_map(placement.max_memory if placement is not None else None)
    offload_folder = placement.offload_folder if managed_device_map_active and placement is not None else None
    return {
        "managed_device_map_active": managed_device_map_active,
        "memory_budget_source": placement.memory_budget_source if placement is not None else "none",
        "effective_max_memory": effective_max_memory,
        "offload_cap_enabled": managed_device_map_active,
        "offload_scope": "stage" if managed_device_map_active else None,
        "offload_folder": offload_folder,
        "allocator_hint_auto_applied": os.environ.get("VT_AUTO_ALLOCATOR_HINT_APPLIED") == "1"
        if managed_device_map_active
        else False,
    }


def resolve_hf_load_placement(
    request,
    *,
    device: torch.device,
    stage_dir: Path | None,
    model_cache: Path,
    include_default_device_map: bool = True,
    default_device_map: str | dict[str, str | int] | None = None,
    default_offload_folder: bool = False,
) -> HFLoadPlacement:
    requested_max_gpu_memory = str(getattr(request, "max_gpu_memory", "") or "").strip() or None
    auto_requested = utils.normalize_device_pref(request.device) == "auto"
    if requested_max_gpu_memory is not None and not auto_requested:
        raise ValueError("`--max-gpu-memory` requires `--device auto`.")
    if requested_max_gpu_memory is not None and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("`--max-gpu-memory` requires CUDA and `--device auto`.")
    if auto_requested and device.type == "cuda" and torch.cuda.is_available():
        if requested_max_gpu_memory is not None:
            gpu_count = int(torch.cuda.device_count())
            if gpu_count < 1:
                raise RuntimeError("`--max-gpu-memory` requires at least one visible CUDA device.")
            max_memory: dict[int | str, int | str] = {index: requested_max_gpu_memory for index in range(gpu_count)}
            max_memory["cpu"] = utils.available_cpu_memory_bytes()
            return HFLoadPlacement(
                device_map="auto",
                max_memory=max_memory,
                offload_folder=str(resolve_offload_folder_path(stage_dir, model_cache)),
                offload_cap_enabled=True,
                managed_device_map_active=True,
                memory_budget_source="explicit",
            )

        return HFLoadPlacement(
            device_map="auto",
            max_memory=_infer_auto_max_memory(),
            offload_folder=str(resolve_offload_folder_path(stage_dir, model_cache)),
            offload_cap_enabled=True,
            managed_device_map_active=True,
            memory_budget_source="inferred",
        )

    device_map = default_device_map if default_device_map is not None else resolve_device_map(request.device)
    if not include_default_device_map:
        device_map = None
    offload_folder = None
    if default_offload_folder and device_map is not None:
        offload_folder = str(resolve_offload_folder_path(stage_dir, model_cache))
    managed_device_map_active = device_map == "auto"
    return HFLoadPlacement(
        device_map=device_map,
        offload_folder=offload_folder,
        offload_cap_enabled=managed_device_map_active,
        managed_device_map_active=managed_device_map_active,
        memory_budget_source="none",
    )


def move_inputs(inputs: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def coerce_generated_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        return coerce_generated_text(value[0]) if value else ""
    if isinstance(value, list):
        return "".join(coerce_generated_text(item) for item in value)
    if hasattr(value, "__iter__") and not isinstance(value, (bytes, bytearray, dict)):
        return "".join(str(item) for item in value)
    return str(value)


class BaseVLMModel(BaseFMModel[TOptions, VLMJob, TRuntime], Generic[TOptions, TRuntime]):
    warning_prefix: str | None = None

    def __init__(self) -> None:
        super().__init__()
        self.question_meta: list[dict[str, str]] = []

    @staticmethod
    def load_dataclass_config(path: Path | None, config_cls: type[TConfig], overrides: Any | None = None) -> TConfig:
        return load_dataclass_config(path, config_cls, overrides)

    @staticmethod
    def resolve_device_map(pref: str) -> str | dict[str, str]:
        return resolve_device_map(pref)

    @staticmethod
    def ensure_image_token(prompt: str) -> str:
        return ensure_image_token(prompt)

    @staticmethod
    def load_rgb_image(path: Path) -> Image.Image:
        return load_rgb_image(path)

    def _warning_prefix(self) -> str:
        return str(self.warning_prefix or self.model_name)

    def build_jobs(
        self,
        dataset: VisionDataset,
        ctx: RuntimeContext[TOptions],
    ) -> list[VLMJob]:
        jobs, question_meta = collect_vlm_jobs(dataset, ctx.prompt, ctx.caller_cwd)
        self.question_meta = question_meta
        return jobs

    def job_id(
        self,
        job: VLMJob,
        dataset: VisionDataset,
        ctx: RuntimeContext[TOptions],
    ) -> str:
        return vlm_job_id(job)

    def load_batch_images(
        self,
        dataset: VisionDataset,
        batch: Sequence[VLMJob],
    ) -> tuple[list[VLMJob], list[Image.Image], list[str]]:
        ready_jobs: list[VLMJob] = []
        images: list[Image.Image] = []
        warnings: list[str] = []
        for job in batch:
            if not job.image_path.exists():
                warnings.append(
                    f"[{self._warning_prefix()}] missing image idx={job.record_idx} "
                    f"src={dataset.records[job.record_idx].image.path} resolved={job.image_path}"
                )
                continue
            try:
                images.append(self.load_rgb_image(job.image_path))
            except Exception as exc:
                warnings.append(
                    f"[{self._warning_prefix()}] failed to load image idx={job.record_idx} path={job.image_path}: {exc}"
                )
                continue
            ready_jobs.append(job)
        return ready_jobs, images, warnings

    @staticmethod
    def move_vlm_inputs(inputs: dict[str, Any], device, dtype):
        return move_inputs(inputs, device, dtype)

    @staticmethod
    def coerce_answer_text(value: Any) -> str:
        return coerce_generated_text(value).strip()

    def append_answers(
        self,
        dataset: VisionDataset,
        jobs: Sequence[VLMJob],
        answers: Sequence[Any],
    ) -> list[int]:
        modified: list[int] = []
        for job, answer in zip(jobs, answers):
            modified.append(append_answer(dataset, job, self.coerce_answer_text(answer), self.model_name))
        return modified

    def build_vlm_meta(
        self,
        *,
        runtime: TRuntime,
        ctx: RuntimeContext[TOptions],
        hf_model_id: str,
        precision,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return build_vlm_fm_meta(
            provider_id=self.model_name,
            hf_model_id=hf_model_id,
            question_meta=self.question_meta,
            ctx=ctx,
            precision=precision,
            placement=self._last_hf_load_placement,
            extra=extra,
        )
