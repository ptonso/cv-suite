from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Sequence, TypeVar

from PIL import Image, ImageDraw, ImageFont

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext

from .gen import (
    BaseGenerationModel,
    GEN_SOURCE_IMAGE_ATTR,
    GenerationRuntime,
    build_generation_runtime_extra,
    generation_record_triplet,
    load_generation_runtime,
    materialize_generated_output,
    normalize_generated_output,
)

TOptions = TypeVar("TOptions")

DEFAULT_EDIT_BACKEND_ALIASES = {
    "debug": "cvsuite.common.fm.providers.bases.edit:PromptCardEditBackend",
    "prompt-card": "cvsuite.common.fm.providers.bases.edit:PromptCardEditBackend",
    "prompt_card": "cvsuite.common.fm.providers.bases.edit:PromptCardEditBackend",
    "test": "cvsuite.common.fm.providers.bases.edit:PromptCardEditBackend",
}


@dataclass(frozen=True)
class EditGenerationOptions:
    backend: str | None = None
    model_id: str | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    image_guidance_scale: float | None = None
    negative_prompt: str | None = None
    strength: float | None = None
    max_sequence_length: int | None = None
    true_cfg_scale: float | None = None
    designer_model_id: str | None = None


@dataclass(frozen=True)
class EditGenerationJob:
    record_idx: int
    prompt: str
    prompt_index: int
    output_key: str
    source_image_path: Path


class PromptCardEditBackend:
    """Small local backend used for smoke tests and debugging edit-style gen pipelines."""

    def __init__(self, config: dict[str, Any] | None = None, **_kwargs: Any) -> None:
        self.config = dict(config or {})

    def generate_batch(self, *, jobs: Sequence[EditGenerationJob], **_kwargs: Any):
        return [self._render_edit(job.source_image_path, job.prompt) for job in jobs]

    def _render_edit(self, source_path: Path, prompt: str) -> Image.Image:
        with Image.open(source_path) as source:
            canvas = source.convert("RGB")
        _draw_prompt_banner(canvas, prompt, caption="prompt-card edit")
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


class BaseEditGenerationModel(
    BaseGenerationModel[TOptions, EditGenerationJob, GenerationRuntime],
    Generic[TOptions],
):
    allow_root_basename = True
    supported_generation_modes = frozenset({"edit"})
    backend_aliases = DEFAULT_EDIT_BACKEND_ALIASES

    def build_jobs(self, dataset: VisionDataset, ctx: RuntimeContext[TOptions]) -> list[EditGenerationJob]:
        mode = self.validate_generation_request(dataset, ctx.request)
        if mode != "edit":
            raise ValueError(f"{self.model_name} only supports generation mode 'edit', got {mode!r}.")
        jobs: list[EditGenerationJob] = []
        for record_idx, record in enumerate(dataset.records):
            prompt, prompt_index, output_key = generation_record_triplet(record_idx, record)
            source_value = record.attributes.get(GEN_SOURCE_IMAGE_ATTR)
            source_ref = Path(str(source_value)) if str(source_value or "").strip() else record.image.path
            source_image_path = utils.resolve_path(
                source_ref,
                dataset.root,
                ctx.caller_cwd,
                allow_root_basename=self.allow_root_basename,
            )
            jobs.append(
                EditGenerationJob(
                    record_idx=record_idx,
                    prompt=prompt,
                    prompt_index=prompt_index,
                    output_key=output_key,
                    source_image_path=source_image_path,
                )
            )
        return jobs

    def job_id(self, job: EditGenerationJob, dataset: VisionDataset, ctx: RuntimeContext[TOptions]) -> str:
        return f"record:{job.record_idx}"

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[TOptions]) -> GenerationRuntime:
        return load_generation_runtime(
            backend_aliases=self.backend_aliases,
            dataset=dataset,
            ctx=ctx,
            mode="edit",
            model=self,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: Sequence[EditGenerationJob],
        runtime: GenerationRuntime,
        ctx: RuntimeContext[TOptions],
    ) -> BatchResult:
        warnings: list[str] = []
        ready_jobs: list[EditGenerationJob] = []
        for job in batch:
            if not job.source_image_path.exists():
                warnings.append(
                    f"[gen] missing source image idx={job.record_idx} path={job.source_image_path}"
                )
                continue
            ready_jobs.append(job)
        if not ready_jobs:
            return BatchResult(modified_record_indices=[], warnings=warnings)

        outputs = runtime.backend.generate_batch(
            mode="edit",
            jobs=ready_jobs,
            dataset=dataset,
            ctx=ctx,
            config=runtime.config,
        )
        if len(outputs) != len(ready_jobs):
            raise RuntimeError(
                f"Generation backend {runtime.backend_spec!r} returned {len(outputs)} outputs for {len(ready_jobs)} jobs."
            )

        dataset.root = runtime.outputs_dir
        modified: list[int] = []
        for job, raw_output in zip(ready_jobs, outputs):
            result = normalize_generated_output(raw_output)
            dst = runtime.outputs_dir / f"{job.output_key}.png"
            width, height = materialize_generated_output(result, dst)
            record = dataset.records[job.record_idx]
            record.image.path = Path(dst.name)
            record.image.width = int(width)
            record.image.height = int(height)
            utils.append_once(record.attributes.setdefault("fm_tasks", []), "gen")
            modified.append(job.record_idx)
        return BatchResult(modified_record_indices=modified, warnings=warnings)

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
