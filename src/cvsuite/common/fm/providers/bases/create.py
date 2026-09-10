from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Sequence, TypeVar

from PIL import Image, ImageColor, ImageDraw, ImageFont

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext

from .gen import (
    BaseGenerationModel,
    GenerationRuntime,
    GEN_PROMPT_ATTR,
    build_generation_runtime_extra,
    generation_record_triplet,
    load_generation_runtime,
    materialize_generated_output,
    normalize_generated_output,
)

TOptions = TypeVar("TOptions")

DEFAULT_CREATE_BACKEND_ALIASES = {
    "debug": "cvsuite.common.fm.providers.bases.create:PromptCardCreateBackend",
    "prompt-card": "cvsuite.common.fm.providers.bases.create:PromptCardCreateBackend",
    "prompt_card": "cvsuite.common.fm.providers.bases.create:PromptCardCreateBackend",
    "test": "cvsuite.common.fm.providers.bases.create:PromptCardCreateBackend",
}


@dataclass(frozen=True)
class CreateGenerationOptions:
    backend: str | None = None
    model_id: str | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    max_sequence_length: int | None = None
    true_cfg_scale: float | None = None


@dataclass(frozen=True)
class CreateGenerationJob:
    record_idx: int
    prompt: str
    prompt_index: int
    output_key: str


def _int_config(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(1, parsed)


def _color_from_prompt(prompt: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return tuple(int(digest[offset : offset + 2], 16) for offset in (0, 2, 4))


class PromptCardCreateBackend:
    """Small local backend used for smoke tests and debugging create-style gen pipelines."""

    def __init__(self, config: dict[str, Any] | None = None, **_kwargs: Any) -> None:
        self.config = dict(config or {})

    def generate_batch(self, *, jobs: Sequence[CreateGenerationJob], **_kwargs: Any):
        return [self._render_prompt(job.prompt) for job in jobs]

    def _render_prompt(self, prompt: str) -> Image.Image:
        width = _int_config(self.config, "width", 512)
        height = _int_config(self.config, "height", 512)
        background = self.config.get("background")
        if isinstance(background, str) and background.strip():
            bg = ImageColor.getrgb(background.strip())
        else:
            bg = _color_from_prompt(prompt)
        canvas = Image.new("RGB", (width, height), color=bg)
        _draw_prompt_banner(canvas, prompt, caption="prompt-card create")
        return canvas


def _draw_prompt_banner(image: Image.Image, prompt: str, *, caption: str) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    margin = 10
    banner_height = min(max(48, height // 4), max(60, height - 2 * margin))
    top = max(0, height - banner_height)
    draw.rectangle([(0, top), (width, height)], fill=(0, 0, 0))
    wrapped = textwrap.wrap(prompt, width=max(12, width // 12)) or [prompt]
    lines = [caption] + wrapped[:4]
    y = top + margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill=(255, 255, 255))
        y += 14


class BaseCreateGenerationModel(
    BaseGenerationModel[TOptions, CreateGenerationJob, GenerationRuntime],
    Generic[TOptions],
):
    supported_generation_modes = frozenset({"create"})
    backend_aliases = DEFAULT_CREATE_BACKEND_ALIASES

    def build_jobs(self, dataset: VisionDataset, ctx: RuntimeContext[TOptions]) -> list[CreateGenerationJob]:
        mode = self.validate_generation_request(dataset, ctx.request)
        if mode != "create":
            raise ValueError(f"{self.model_name} only supports generation mode 'create', got {mode!r}.")
        jobs: list[CreateGenerationJob] = []
        for record_idx, record in enumerate(dataset.records):
            prompt, prompt_index, output_key = generation_record_triplet(record_idx, record)
            jobs.append(
                CreateGenerationJob(
                    record_idx=record_idx,
                    prompt=prompt,
                    prompt_index=prompt_index,
                    output_key=output_key,
                )
            )
        return jobs

    def job_id(self, job: CreateGenerationJob, dataset: VisionDataset, ctx: RuntimeContext[TOptions]) -> str:
        return f"record:{job.record_idx}"

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[TOptions]) -> GenerationRuntime:
        return load_generation_runtime(
            backend_aliases=self.backend_aliases,
            dataset=dataset,
            ctx=ctx,
            mode="create",
            model=self,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: Sequence[CreateGenerationJob],
        runtime: GenerationRuntime,
        ctx: RuntimeContext[TOptions],
    ) -> BatchResult:
        outputs = runtime.backend.generate_batch(
            mode="create",
            jobs=batch,
            dataset=dataset,
            ctx=ctx,
            config=runtime.config,
        )
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"Generation backend {runtime.backend_spec!r} returned {len(outputs)} outputs for {len(batch)} jobs."
            )

        dataset.root = runtime.outputs_dir
        modified: list[int] = []
        for job, raw_output in zip(batch, outputs):
            result = normalize_generated_output(raw_output)
            dst = runtime.outputs_dir / f"{job.output_key}.png"
            width, height = materialize_generated_output(result, dst)
            record = dataset.records[job.record_idx]
            record.image.path = Path(dst.name)
            record.image.width = int(width)
            record.image.height = int(height)
            utils.append_once(record.attributes.setdefault("fm_tasks", []), "gen")
            modified.append(job.record_idx)
        return BatchResult(modified_record_indices=modified)

    def finalize_dataset(self, dataset: VisionDataset, runtime: GenerationRuntime, ctx: RuntimeContext[TOptions]) -> None:
        if runtime.outputs_dir.exists():
            dataset.root = runtime.outputs_dir

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: GenerationRuntime,
        ctx: RuntimeContext[TOptions],
    ) -> dict[str, object]:
        return self.build_generation_fm_meta(
            dataset,
            config_path=ctx.config_path,
            weights_dir=ctx.weights_dir,
            request=ctx.request,
            batch_size=ctx.batch_size,
            extra=build_generation_runtime_extra(dataset, runtime, ctx),
        )
