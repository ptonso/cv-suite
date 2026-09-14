from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases.vlm import BaseVLMModel, VLMJob

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PARAMS = "3B"
SUPPORTED_QWEN_MODEL_TYPES = frozenset({"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"})


@dataclass(frozen=True)
class QwenOptions:
    model_id: str | None = None
    max_new_tokens: int | None = None
    trust_remote_code: bool | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None


@dataclass(frozen=True)
class QwenVLConfig:
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 64
    trust_remote_code: bool = False
    min_pixels: int = 16 * 28 * 28
    max_pixels: int = 2560 * 28 * 28


QwenJob = VLMJob


@dataclass
class QwenRuntime:
    cfg: QwenVLConfig
    device: torch.device
    dtype: torch.dtype
    precision: utils.ResolvedPrecision
    model: AutoModelForImageTextToText
    processor: AutoProcessor
    model_device: torch.device
    autocast_ctx: object
    model_type: str


class QwenModel(BaseVLMModel[QwenOptions, QwenRuntime]):
    description = "Qwen VLM wrapper: runs VQA prompts over a VisionDataset JSON."
    options_cls = QwenOptions
    supports_nf4_precision = True
    supports_managed_auto_device = True

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[QwenOptions]) -> QwenRuntime:
        cfg = self.load_dataclass_config(ctx.config_path, QwenVLConfig, ctx.options)
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
        config = AutoConfig.from_pretrained(
            source.load_arg,
            **utils.build_hf_source_kwargs(
                source,
                trust_remote_code=cfg.trust_remote_code,
            ),
        )
        model_type = str(getattr(config, "model_type", "") or "").strip()
        if model_type not in SUPPORTED_QWEN_MODEL_TYPES:
            supported = ", ".join(sorted(SUPPORTED_QWEN_MODEL_TYPES))
            raise ValueError(
                f"Qwen VLM wrapper does not support checkpoint {cfg.model_id!r} with model_type={model_type!r}. "
                f"Expected one of: {supported}."
            )
        processor = AutoProcessor.from_pretrained(
            source.load_arg,
            min_pixels=int(cfg.min_pixels),
            max_pixels=int(cfg.max_pixels),
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
            default_offload_folder=True,
        )
        model = self.load_hf_pretrained_model(
            AutoModelForImageTextToText.from_pretrained,
            precision,
            source.load_arg,
            **utils.build_hf_source_kwargs(
                source,
                config=config,
                trust_remote_code=cfg.trust_remote_code,
                **load_kwargs,
            ),
        )
        model.eval()
        model_device = next(model.parameters()).device
        autocast_ctx = utils.maybe_autocast(model_device, precision)
        return QwenRuntime(
            cfg=cfg,
            device=device,
            dtype=dtype,
            precision=precision,
            model=model,
            processor=processor,
            model_device=model_device,
            autocast_ctx=autocast_ctx,
            model_type=model_type,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: list[QwenJob],
        runtime: QwenRuntime,
        ctx: RuntimeContext[QwenOptions],
    ) -> BatchResult:
        ready_jobs, images, warnings = self.load_batch_images(dataset, batch)
        if not ready_jobs:
            return BatchResult(modified_record_indices=[], warnings=warnings)

        text_prompts = []
        for job in ready_jobs:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": job.question},
                    ],
                }
            ]
            text_prompts.append(
                runtime.processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = runtime.processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = self.move_vlm_inputs(inputs, runtime.model_device, runtime.dtype)
        with torch.inference_mode(), runtime.autocast_ctx:
            output_ids = runtime.model.generate(**inputs, max_new_tokens=runtime.cfg.max_new_tokens)

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
        runtime: QwenRuntime,
        ctx: RuntimeContext[QwenOptions],
    ) -> dict[str, object]:
        return self.build_vlm_meta(
            runtime=runtime,
            ctx=ctx,
            hf_model_id=runtime.cfg.model_id,
            precision=runtime.precision,
            extra={
                "max_new_tokens": runtime.cfg.max_new_tokens,
                "hf_model_type": runtime.model_type,
                "min_pixels": runtime.cfg.min_pixels,
                "max_pixels": runtime.cfg.max_pixels,
            },
        )


MODEL = QwenModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
