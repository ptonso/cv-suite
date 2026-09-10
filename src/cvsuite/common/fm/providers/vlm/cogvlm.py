from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases.vlm import BaseVLMModel, VLMJob

DEFAULT_MODEL_ID = "zai-org/cogvlm2-llama3-chat-19B"
PARAMS = "19B"


@dataclass(frozen=True)
class CogVLMOptions:
    model_id: str | None = None
    max_new_tokens: int | None = None
    trust_remote_code: bool | None = None
    template_version: str | None = None
    use_fast: bool | None = None


@dataclass(frozen=True)
class CogVLMConfig:
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 64
    trust_remote_code: bool = True
    template_version: str = "chat"
    use_fast: bool = False


@dataclass
class CogVLMRuntime:
    cfg: CogVLMConfig
    device: torch.device
    dtype: torch.dtype
    precision: utils.ResolvedPrecision
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    model_device: torch.device
    autocast_ctx: object


class CogVLMModel(BaseVLMModel[CogVLMOptions, CogVLMRuntime]):
    description = "CogVLM wrapper: runs image-chat prompts over a VisionDataset JSON."
    options_cls = CogVLMOptions
    supports_nf4_precision = True
    supports_managed_auto_device = True

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[CogVLMOptions]) -> CogVLMRuntime:
        cfg = self.load_dataclass_config(ctx.config_path, CogVLMConfig, ctx.options)
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

        tokenizer = AutoTokenizer.from_pretrained(
            source.load_arg,
            **utils.build_hf_source_kwargs(
                source,
                trust_remote_code=cfg.trust_remote_code,
                use_fast=cfg.use_fast,
            ),
        )
        load_kwargs, _placement = self.build_hf_load_kwargs(
            ctx,
            device,
            precision,
            dtype_key="torch_dtype",
            default_device_map=default_device_map,
        )
        model = self.load_hf_pretrained_model(
            AutoModelForCausalLM.from_pretrained,
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
        return CogVLMRuntime(
            cfg=cfg,
            device=device,
            dtype=dtype,
            precision=precision,
            model=model,
            tokenizer=tokenizer,
            model_device=model_device,
            autocast_ctx=autocast_ctx,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: list[VLMJob],
        runtime: CogVLMRuntime,
        ctx: RuntimeContext[CogVLMOptions],
    ) -> BatchResult:
        ready_jobs, images, warnings = self.load_batch_images(dataset, batch)
        modified: list[int] = []

        for job, image in zip(ready_jobs, images):
            input_by_model = runtime.model.build_conversation_input_ids(
                runtime.tokenizer,
                query=job.question,
                history=[],
                images=[image],
                template_version=runtime.cfg.template_version,
            )
            attention_mask = input_by_model.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_by_model["input_ids"])
            inputs = {
                "input_ids": input_by_model["input_ids"].unsqueeze(0).to(runtime.model_device),
                "attention_mask": attention_mask.unsqueeze(0).to(runtime.model_device),
            }
            token_type_ids = input_by_model.get("token_type_ids")
            if token_type_ids is not None:
                inputs["token_type_ids"] = token_type_ids.unsqueeze(0).to(runtime.model_device)

            image_tensors = input_by_model.get("images")
            if image_tensors is not None:
                if torch.is_tensor(image_tensors):
                    image_tensors = [image_tensors]
                converted = []
                for tensor in image_tensors:
                    value = tensor.to(runtime.model_device)
                    if value.is_floating_point():
                        value = value.to(runtime.dtype)
                    converted.append(value)
                inputs["images"] = [converted]

            generate_kwargs = {
                "max_new_tokens": runtime.cfg.max_new_tokens,
                "do_sample": False,
            }
            if runtime.tokenizer.eos_token_id is not None:
                generate_kwargs["pad_token_id"] = runtime.tokenizer.eos_token_id

            with torch.inference_mode(), runtime.autocast_ctx:
                output_ids = runtime.model.generate(**inputs, **generate_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            answer = runtime.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
            modified.extend(self.append_answers(dataset, [job], [answer]))

        return BatchResult(modified_record_indices=modified, warnings=warnings)

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: CogVLMRuntime,
        ctx: RuntimeContext[CogVLMOptions],
    ) -> dict[str, object]:
        return self.build_vlm_meta(
            runtime=runtime,
            ctx=ctx,
            hf_model_id=runtime.cfg.model_id,
            precision=runtime.precision,
            extra={
                "max_new_tokens": runtime.cfg.max_new_tokens,
                "template_version": runtime.cfg.template_version,
            },
        )


MODEL = CogVLMModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
