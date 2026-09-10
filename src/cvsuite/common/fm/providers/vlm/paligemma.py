from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases.vlm import BaseVLMModel, VLMJob

DEFAULT_MODEL_ID = "google/paligemma-3b-ft-ocrvqa-448"
PARAMS = "3B"


@dataclass(frozen=True)
class PGOptions:
    model_id: str | None = None
    max_new_tokens: int | None = None
    use_fast: bool | None = None


@dataclass(frozen=True)
class PGConfig:
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 64
    use_fast: bool = True


PGJob = VLMJob


@dataclass
class PGRuntime:
    cfg: PGConfig
    device: torch.device
    dtype: torch.dtype
    precision: utils.ResolvedPrecision
    model: PaliGemmaForConditionalGeneration
    processor: AutoProcessor
    model_device: torch.device
    autocast_ctx: object
    cfg_path: Path | None


class PaliGemmaModel(BaseVLMModel[PGOptions, PGRuntime]):
    description = "PaliGemma VLM wrapper: runs VQA prompts over a VisionDataset JSON."
    options_cls = PGOptions
    supports_nf4_precision = True
    supports_managed_auto_device = True

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[PGOptions]) -> PGRuntime:
        cfg = self.load_dataclass_config(ctx.config_path, PGConfig, ctx.options)
        source = utils.materialize_hf_model_source(
            cfg.model_id,
            hub_dir=ctx.hub_dir,
            stage_dir=getattr(ctx, "stage_dir", None),
            caller_cwd=getattr(ctx, "caller_cwd", Path.cwd()),
        )
        device = utils.select_device(ctx.request.device)
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        dtype = precision.compute_dtype
        default_device_map = self.resolve_device_map(ctx.request.device)

        qcfg = None
        load_kwargs, _placement = self.build_hf_load_kwargs(
            ctx,
            device,
            precision,
            dtype_key="torch_dtype",
            default_device_map=default_device_map,
        )
        try:
            model = self.load_hf_pretrained_model(
                PaliGemmaForConditionalGeneration.from_pretrained,
                precision,
                source.load_arg,
                **utils.build_hf_source_kwargs(
                    source,
                    quantization_config=qcfg,
                    **load_kwargs,
                ),
            )
            processor = AutoProcessor.from_pretrained(
                source.load_arg,
                **utils.build_hf_source_kwargs(
                    source,
                    use_fast=cfg.use_fast,
                ),
            )
        except OSError as exc:
            detail = utils.describe_hf_access_error(exc, cfg.model_id)
            if detail is not None:
                raise RuntimeError(detail) from exc
            raise
        model.eval()
        model_device = next(model.parameters()).device
        autocast_ctx = utils.maybe_autocast(model_device, precision)
        return PGRuntime(
            cfg=cfg,
            device=device,
            dtype=dtype,
            precision=precision,
            model=model,
            processor=processor,
            model_device=model_device,
            autocast_ctx=autocast_ctx,
            cfg_path=ctx.config_path,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: list[PGJob],
        runtime: PGRuntime,
        ctx: RuntimeContext[PGOptions],
    ) -> BatchResult:
        batch_jobs, images, warnings = self.load_batch_images(dataset, batch)
        if not batch_jobs:
            return BatchResult(modified_record_indices=[], warnings=warnings)

        text_prompts = [self.ensure_image_token(job.question) for job in batch_jobs]
        inputs = runtime.processor(
            images=images,
            text=text_prompts,
            padding=True,
            return_tensors="pt",
        )
        inputs = self.move_vlm_inputs(inputs, runtime.model_device, runtime.dtype)
        gen_kwargs = {"max_new_tokens": runtime.cfg.max_new_tokens}
        if "cache_implementation" in inspect.signature(runtime.model.generate).parameters:
            gen_kwargs["cache_implementation"] = "static"
        with torch.inference_mode(), runtime.autocast_ctx:
            output_ids = runtime.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_ids = output_ids[:, input_len:]
        texts = runtime.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return BatchResult(
            modified_record_indices=self.append_answers(dataset, batch_jobs, texts),
            warnings=warnings,
        )

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: PGRuntime,
        ctx: RuntimeContext[PGOptions],
    ) -> dict[str, object]:
        return self.build_vlm_meta(
            runtime=runtime,
            ctx=ctx,
            hf_model_id=runtime.cfg.model_id,
            precision=runtime.precision,
            extra={
                "config": str(runtime.cfg_path) if runtime.cfg_path else None,
                "max_new_tokens": runtime.cfg.max_new_tokens,
            },
        )


MODEL = PaliGemmaModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
