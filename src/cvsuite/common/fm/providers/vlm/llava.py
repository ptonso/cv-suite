from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases.vlm import BaseVLMModel, VLMJob

DEFAULT_MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
PARAMS = "500M"


@dataclass(frozen=True)
class LlavaOptions:
    model_id: str | None = None
    max_new_tokens: int | None = None
    trust_remote_code: bool | None = None


@dataclass(frozen=True)
class LlavaConfig:
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 64
    trust_remote_code: bool = False


@dataclass
class LlavaRuntime:
    cfg: LlavaConfig
    device: torch.device
    dtype: torch.dtype
    precision: utils.ResolvedPrecision
    model: AutoModelForImageTextToText
    processor: AutoProcessor
    model_device: torch.device
    autocast_ctx: object


class LlavaModel(BaseVLMModel[LlavaOptions, LlavaRuntime]):
    description = "LLaVA VLM wrapper: runs VQA prompts over a VisionDataset JSON."
    options_cls = LlavaOptions
    supports_nf4_precision = True
    supports_managed_auto_device = True

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[LlavaOptions]) -> LlavaRuntime:
        cfg = self.load_dataclass_config(ctx.config_path, LlavaConfig, ctx.options)
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

        processor = AutoProcessor.from_pretrained(
            source.load_arg,
            **utils.build_hf_source_kwargs(
                source,
                trust_remote_code=cfg.trust_remote_code,
            ),
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
        load_kwargs, _placement = self.build_hf_load_kwargs(
            ctx,
            device,
            precision,
            dtype_key="torch_dtype",
            default_device_map=default_device_map,
        )

        model = self.load_hf_pretrained_model(
            AutoModelForImageTextToText.from_pretrained,
            precision,
            source.load_arg,
            **utils.build_hf_source_kwargs(
                source,
                trust_remote_code=cfg.trust_remote_code,
                **load_kwargs,
            ),
        )
        model.eval()
        model_device = next(model.parameters()).device
        autocast_ctx = utils.maybe_autocast(model_device, precision)
        return LlavaRuntime(
            cfg=cfg,
            device=device,
            dtype=dtype,
            precision=precision,
            model=model,
            processor=processor,
            model_device=model_device,
            autocast_ctx=autocast_ctx,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: list[VLMJob],
        runtime: LlavaRuntime,
        ctx: RuntimeContext[LlavaOptions],
    ) -> BatchResult:
        ready_jobs, images, warnings = self.load_batch_images(dataset, batch)
        if not ready_jobs:
            return BatchResult(modified_record_indices=[], warnings=warnings)

        prompts = [
            runtime.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": job.question}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for job in ready_jobs
        ]

        inputs = runtime.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        inputs = self.move_vlm_inputs(inputs, runtime.model_device, runtime.dtype)
        generate_kwargs = {"max_new_tokens": runtime.cfg.max_new_tokens}
        tokenizer = getattr(runtime.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
            generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

        with torch.inference_mode(), runtime.autocast_ctx:
            output_ids = runtime.model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_ids = output_ids[:, input_len:]
        texts = runtime.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return BatchResult(
            modified_record_indices=self.append_answers(dataset, ready_jobs, texts),
            warnings=warnings,
        )

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: LlavaRuntime,
        ctx: RuntimeContext[LlavaOptions],
    ) -> dict[str, object]:
        return self.build_vlm_meta(
            runtime=runtime,
            ctx=ctx,
            hf_model_id=runtime.cfg.model_id,
            precision=runtime.precision,
            extra={"max_new_tokens": runtime.cfg.max_new_tokens},
        )


MODEL = LlavaModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
