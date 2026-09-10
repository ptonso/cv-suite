from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.utils import logging as transformers_logging

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import prompt_utils
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseClassificationModel, RecordImageJob

DEFAULT_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PARAMS = "986M"


@dataclass(frozen=True)
class ClipOptions:
    threshold: float | None = 0.6
    template_prompt: str = prompt_utils.DEFAULT_CLASS_TEMPLATE_PROMPT


ClipJob = RecordImageJob


@dataclass
class ClipRuntime:
    device: torch.device
    dtype: torch.dtype
    processor: CLIPProcessor
    model: CLIPModel
    text_feats: torch.Tensor
    logit_scale: torch.Tensor
    labels: List[str]
    prompt_map: Dict[str, List[str]]
    prompt_to_label: List[str]
    prompts: List[str]


def _pool_output(outputs):
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(outputs, tuple) and len(outputs) > 1:
        return outputs[1]
    raise RuntimeError("CLIP model outputs did not contain a pooled embedding.")

def _build_prompt_batch(prompt_map: Dict[str, List[str]]) -> Tuple[List[str], List[str], List[str]]:
    labels: List[str] = []
    prompt_to_label: List[str] = []
    prompts: List[str] = []
    for label, label_prompts in prompt_map.items():
        clean_label = str(label).strip()
        if not clean_label:
            continue
        labels.append(clean_label)
        for prompt in label_prompts:
            clean_prompt = str(prompt).strip()
            if not clean_prompt:
                continue
            prompt_to_label.append(clean_label)
            prompts.append(clean_prompt)
    if not labels or not prompts:
        raise ValueError("At least one label and prompt are required for CLIP classification.")
    return labels, prompt_to_label, prompts


def _encode_text(
    prompts: List[str],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_outputs = model.text_model(**text_inputs)
        feats = model.text_projection(_pool_output(text_outputs))
    feats = feats.to(device=device, dtype=dtype)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    return feats, logit_scale


def _predict_images(
    rec_batch: Sequence[Tuple[int, Image.Image]],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    dtype: torch.dtype,
    text_feats: torch.Tensor,
    logit_scale: torch.Tensor,
    labels: Sequence[str],
    prompt_to_label: Sequence[str],
) -> torch.Tensor:
    images = [img for _, img in rec_batch]
    inputs = processor(images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
    img_feats = model.visual_projection(_pool_output(vision_outputs))
    img_feats = img_feats.to(device=device, dtype=dtype)
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    prompt_logits = logit_scale * img_feats @ text_feats.t()
    label_logits: List[torch.Tensor] = []
    for label in labels:
        indices = [idx for idx, label_name in enumerate(prompt_to_label) if label_name == label]
        if not indices:
            raise ValueError(f"Label {label!r} has no prompts.")
        label_logits.append(prompt_logits[:, indices].max(dim=1).values)
    return torch.stack(label_logits, dim=-1).softmax(dim=-1)


def _load_model(
    hub_dir: Path | None,
    stage_dir: Path | None,
    caller_cwd: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[CLIPModel, CLIPProcessor]:
    source = utils.materialize_hf_model_source(
        DEFAULT_MODEL_ID,
        hub_dir=hub_dir,
        stage_dir=stage_dir,
        caller_cwd=caller_cwd,
    )
    previous_verbosity = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity_error()
        processor = CLIPProcessor.from_pretrained(
            source.load_arg,
            **utils.build_hf_source_kwargs(
                source,
                use_fast=False,
            ),
        )
        model = CLIPModel.from_pretrained(
            source.load_arg,
            **utils.build_hf_source_kwargs(
                source,
                torch_dtype=dtype,
            ),
        )
    finally:
        transformers_logging.set_verbosity(previous_verbosity)
    model = model.to(device)
    model.eval()
    return model, processor


class ClipModel(BaseClassificationModel[ClipOptions, ClipRuntime]):
    description = "CLIP wrapper: consumes in.json, writes out.json with zero-shot classifications."
    options_cls = ClipOptions
    allow_root_basename = True

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[ClipOptions]) -> ClipRuntime:
        prompt_map = prompt_utils.build_prompt_map(ctx.prompt, template_prompt=ctx.options.template_prompt)
        labels, prompt_to_label, prompts = _build_prompt_batch(prompt_map)
        device = utils.select_device(ctx.request.device)
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        dtype = precision.compute_dtype
        model, processor = _load_model(
            ctx.hub_dir,
            ctx.stage_dir,
            ctx.caller_cwd,
            device,
            dtype,
        )
        text_feats, logit_scale = _encode_text(prompts, processor, model, device, dtype)
        return ClipRuntime(
            device=device,
            dtype=dtype,
            processor=processor,
            model=model,
            text_feats=text_feats,
            logit_scale=logit_scale,
            labels=labels,
            prompt_map=prompt_map,
            prompt_to_label=prompt_to_label,
            prompts=prompts,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: Sequence[ClipJob],
        runtime: ClipRuntime,
        ctx: RuntimeContext[ClipOptions],
    ) -> BatchResult:
        loaded, warnings = self.collect_loaded_records(dataset, batch, self.load_rgb_image)
        images: List[Tuple[int, object]] = [(job.record_idx, image) for job, image in loaded]
        modified: List[int] = []
        if not images:
            return BatchResult(modified_record_indices=modified, warnings=warnings)

        probs = _predict_images(
            images,
            runtime.processor,
            runtime.model,
            runtime.device,
            runtime.dtype,
            runtime.text_feats,
            runtime.logit_scale,
            runtime.labels,
            runtime.prompt_to_label,
        )
        for (job, _), prob_vec in zip(loaded, probs):
            top_idx = int(torch.argmax(prob_vec).item())
            top_label = runtime.labels[top_idx]
            top_score = float(prob_vec[top_idx].item())
            modified.append(
                self.store_classification_result(
                    dataset,
                    job,
                    ctx=ctx,
                    label=top_label,
                    score=top_score,
                    probs={lbl: float(prob_vec[i].item()) for i, lbl in enumerate(runtime.labels)},
                    meta=self.build_classification_record_meta(
                        ctx,
                        extra={
                            "hf_model": DEFAULT_MODEL_ID,
                            "candidate_prompts": runtime.prompt_map,
                            "template_prompt": ctx.options.template_prompt,
                        },
                    ),
                    candidate_labels_default=runtime.labels,
                )
            )
        return BatchResult(modified_record_indices=modified, warnings=warnings)

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: ClipRuntime,
        ctx: RuntimeContext[ClipOptions],
    ) -> dict[str, object]:
        return self.build_classification_fm_meta(
            ctx,
            candidate_labels=runtime.labels,
            extra={
                "hf_model": DEFAULT_MODEL_ID,
                "weights": str(ctx.weights_dir) if ctx.weights_dir else None,
                "candidate_prompts": runtime.prompt_map,
                "template_prompt": ctx.options.template_prompt,
            },
        )


MODEL = ClipModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
