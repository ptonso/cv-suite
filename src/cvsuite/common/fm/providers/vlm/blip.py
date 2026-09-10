from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoProcessor, BlipConfig as HFBlipConfig, BlipForQuestionAnswering

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases.vlm import BaseVLMModel, VLMJob

DEFAULT_MODEL_ID = "Salesforce/blip-vqa-base"
PARAMS = "446M"
BLIP_IGNORED_UNEXPECTED_KEYS = [
    r"text_encoder\.embeddings\.position_ids",
    r"text_decoder\.bert\.embeddings\.position_ids",
]


@dataclass(frozen=True)
class BlipOptions:
    model_id: str | None = None
    max_new_tokens: int | None = None


@dataclass(frozen=True)
class BlipConfig:
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 32


@dataclass
class BlipRuntime:
    cfg: BlipConfig
    device: torch.device
    dtype: torch.dtype
    precision: utils.ResolvedPrecision
    model: BlipForQuestionAnswering
    processor: AutoProcessor
    model_device: torch.device
    autocast_ctx: object


class BlipModel(BaseVLMModel[BlipOptions, BlipRuntime]):
    description = "BLIP VLM wrapper: runs VQA prompts over a VisionDataset JSON."
    options_cls = BlipOptions
    supports_nf4_precision = True
    supports_managed_auto_device = True

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[BlipOptions]) -> BlipRuntime:
        cfg = self.load_dataclass_config(ctx.config_path, BlipConfig, ctx.options)
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
        hf_config = HFBlipConfig.from_pretrained(source.load_arg, **utils.build_hf_source_kwargs(source))
        # BLIP VQA checkpoints keep decoder/output weights untied even though the
        # upstream config still asks Transformers to tie them on load.
        hf_config.tie_word_embeddings = False
        if getattr(hf_config, "text_config", None) is not None:
            hf_config.text_config.tie_word_embeddings = False
        ignored_unexpected = list(BlipForQuestionAnswering._keys_to_ignore_on_load_unexpected or [])
        for pattern in BLIP_IGNORED_UNEXPECTED_KEYS:
            if pattern not in ignored_unexpected:
                ignored_unexpected.append(pattern)
        BlipForQuestionAnswering._keys_to_ignore_on_load_unexpected = ignored_unexpected
        load_kwargs, _placement = self.build_hf_load_kwargs(
            ctx,
            device,
            precision,
            dtype_key="torch_dtype",
            default_device_map=default_device_map,
        )

        model = self.load_hf_pretrained_model(
            BlipForQuestionAnswering.from_pretrained,
            precision,
            source.load_arg,
            **utils.build_hf_source_kwargs(
                source,
                config=hf_config,
                **load_kwargs,
            ),
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(source.load_arg, **utils.build_hf_source_kwargs(source))
        model_device = next(model.parameters()).device
        autocast_ctx = utils.maybe_autocast(model_device, precision)
        return BlipRuntime(
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
        runtime: BlipRuntime,
        ctx: RuntimeContext[BlipOptions],
    ) -> BatchResult:
        ready_jobs, images, warnings = self.load_batch_images(dataset, batch)
        if not ready_jobs:
            return BatchResult(modified_record_indices=[], warnings=warnings)

        questions = [job.question for job in ready_jobs]
        inputs = runtime.processor(images=images, text=questions, padding=True, return_tensors="pt")
        inputs = self.move_vlm_inputs(inputs, runtime.model_device, runtime.dtype)
        with torch.inference_mode(), runtime.autocast_ctx:
            output_ids = runtime.model.generate(**inputs, max_new_tokens=runtime.cfg.max_new_tokens)

        texts = runtime.processor.batch_decode(output_ids, skip_special_tokens=True)
        return BatchResult(
            modified_record_indices=self.append_answers(dataset, ready_jobs, texts),
            warnings=warnings,
        )

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: BlipRuntime,
        ctx: RuntimeContext[BlipOptions],
    ) -> dict[str, object]:
        return self.build_vlm_meta(
            runtime=runtime,
            ctx=ctx,
            hf_model_id=runtime.cfg.model_id,
            precision=runtime.precision,
            extra={"max_new_tokens": runtime.cfg.max_new_tokens},
        )


MODEL = BlipModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
