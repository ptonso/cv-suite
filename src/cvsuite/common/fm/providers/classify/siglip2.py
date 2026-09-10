from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import yaml
from transformers import AutoConfig, AutoModel, AutoProcessor

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import prompt_utils
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseClassificationModel, RecordImageJob

DEFAULT_MODEL_ID = "google/siglip2-so400m-patch16-512"
PARAMS = "400M"


@dataclass(frozen=True)
class Siglip2Options:
    threshold: float | None = 0.6
    template_prompt: str = prompt_utils.DEFAULT_CLASS_TEMPLATE_PROMPT


@dataclass(frozen=True)
class Siglip2Config:
    model_id: str = DEFAULT_MODEL_ID


Siglip2Job = RecordImageJob


@dataclass
class Siglip2Runtime:
    device: torch.device
    autocast_ctx: object
    model: torch.nn.Module
    processor: AutoProcessor
    cfg: Siglip2Config
    cfg_path: Path | None
    labels: List[str]
    prompt_map: Dict[str, List[str]]
    prompt_to_label: List[str]
    prompts: List[str]


def _load_config(path: Path | None) -> Siglip2Config:
    cfg = Siglip2Config()
    if path and path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict):
            values = {f.name: data.get(f.name, getattr(cfg, f.name)) for f in fields(Siglip2Config)}
            cfg = Siglip2Config(**values)
    return cfg


def _build_prompt_batch(prompt_map: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    prompt_to_label: List[str] = []
    prompts: List[str] = []
    for label, label_prompts in prompt_map.items():
        clean_label = str(label).strip()
        if not clean_label:
            continue
        for prompt in label_prompts:
            clean_prompt = str(prompt).strip()
            if not clean_prompt:
                continue
            prompt_to_label.append(clean_label)
            prompts.append(clean_prompt)
    if not prompts:
        raise ValueError("At least one prompt is required for SigLIP2 classification.")
    return prompt_to_label, prompts


def _prediction_from_prompt_logits(
    prompt_logits: torch.Tensor,
    labels: Sequence[str],
    prompt_to_label: Sequence[str],
) -> Tuple[str, float, Dict[str, float]]:
    label_logits: List[torch.Tensor] = []
    for label in labels:
        indices = [idx for idx, label_name in enumerate(prompt_to_label) if label_name == label]
        if not indices:
            raise ValueError(f"Label {label!r} has no prompts.")
        label_logits.append(prompt_logits[indices].max())

    # SigLIP is sigmoid-trained, so its raw per-pair probabilities are tiny in
    # absolute terms (~1e-4) and not comparable to a CLIP-style threshold. For
    # single-label classification over the prompt set, normalize across labels
    # (same as the CLIP provider) so `--threshold` and the confidence stats mean
    # the same thing regardless of backend.
    probabilities = torch.softmax(torch.stack(label_logits), dim=-1)
    best_index = int(torch.argmax(probabilities).item())
    scores = {label: float(probabilities[idx].item()) for idx, label in enumerate(labels)}
    return labels[best_index], scores[labels[best_index]], scores


def _move_inputs(inputs: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device=device) if torch.is_tensor(value) else value
    return moved


def _load_model_bundle(
    cfg: Siglip2Config,
    hub_dir: Path | None,
    stage_dir: Path | None,
    caller_cwd: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, AutoProcessor]:
    source = utils.materialize_hf_model_source(
        cfg.model_id,
        hub_dir=hub_dir,
        stage_dir=stage_dir,
        caller_cwd=caller_cwd,
    )

    try:
        config = AutoConfig.from_pretrained(source.load_arg, **utils.build_hf_source_kwargs(source))
        if config.model_type not in {"siglip", "siglip2"}:
            raise RuntimeError(f"Unsupported config model_type for {cfg.model_id}: {config.model_type}")
        model = AutoModel.from_pretrained(
            source.load_arg,
            **utils.build_hf_source_kwargs(
                source,
                config=config,
            ),
        )
        processor = AutoProcessor.from_pretrained(source.load_arg, **utils.build_hf_source_kwargs(source))
    except OSError as exc:
        raise RuntimeError(
            "Unable to load the SigLIP2 model. If this is the first run, an internet "
            "connection is required once so the weights can be cached locally."
        ) from exc

    model = model.to(device).eval()
    return model, processor


class Siglip2Model(BaseClassificationModel[Siglip2Options, Siglip2Runtime]):
    description = "SigLIP2 wrapper: consumes in.json, writes out.json with one-shot classifications."
    options_cls = Siglip2Options
    allow_root_basename = True

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[Siglip2Options]) -> Siglip2Runtime:
        cfg = _load_config(ctx.config_path)
        prompt_map = prompt_utils.build_prompt_map(ctx.prompt, template_prompt=ctx.options.template_prompt)
        labels = list(prompt_map.keys())
        if not labels:
            raise ValueError("At least one label must be provided via the dataset FM request for SigLIP2 classification.")
        prompt_to_label, prompts = _build_prompt_batch(prompt_map)
        device = utils.select_device(ctx.request.device)
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        model, processor = _load_model_bundle(
            cfg,
            ctx.hub_dir,
            ctx.stage_dir,
            ctx.caller_cwd,
            device,
        )
        autocast_ctx = utils.maybe_autocast_for_module(device, precision, model)
        return Siglip2Runtime(
            device=device,
            autocast_ctx=autocast_ctx,
            model=model,
            processor=processor,
            cfg=cfg,
            cfg_path=ctx.config_path,
            labels=labels,
            prompt_map=prompt_map,
            prompt_to_label=prompt_to_label,
            prompts=prompts,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: Sequence[Siglip2Job],
        runtime: Siglip2Runtime,
        ctx: RuntimeContext[Siglip2Options],
    ) -> BatchResult:
        loaded, warnings = self.collect_loaded_records(dataset, batch, self.load_rgb_image)
        images = [image for _job, image in loaded]
        modified: List[int] = []
        if not images:
            return BatchResult(modified_record_indices=modified, warnings=warnings)

        inputs = runtime.processor(
            text=runtime.prompts,
            images=images,
            return_tensors="pt",
            padding="max_length",
        )
        inputs = _move_inputs(inputs, runtime.device)

        with torch.inference_mode(), runtime.autocast_ctx:
            prompt_logits_batch = runtime.model(**inputs).logits_per_image.detach().cpu().to(torch.float32)

        for (job, _image), prompt_logits in zip(loaded, prompt_logits_batch):
            top_label, top_score, scores = _prediction_from_prompt_logits(
                prompt_logits=prompt_logits,
                labels=runtime.labels,
                prompt_to_label=runtime.prompt_to_label,
            )
            extra_meta: dict[str, object] = {
                "hf_model": runtime.cfg.model_id,
                "candidate_prompts": runtime.prompt_map,
                "template_prompt": ctx.options.template_prompt,
            }
            if runtime.cfg_path is not None:
                extra_meta["config"] = str(runtime.cfg_path)
            modified.append(
                self.store_classification_result(
                    dataset,
                    job,
                    ctx=ctx,
                    label=top_label,
                    score=top_score,
                    probs=scores,
                    meta=self.build_classification_record_meta(
                        ctx,
                        candidate_labels=runtime.labels,
                        extra=extra_meta,
                    ),
                )
            )
        return BatchResult(modified_record_indices=modified, warnings=warnings)

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: Siglip2Runtime,
        ctx: RuntimeContext[Siglip2Options],
    ) -> dict[str, object]:
        extra: dict[str, object] = {
            "hf_model": runtime.cfg.model_id,
            "weights": str(ctx.weights_dir) if ctx.weights_dir else None,
            "candidate_prompts": runtime.prompt_map,
            "template_prompt": ctx.options.template_prompt,
        }
        if runtime.cfg_path is not None:
            extra["config"] = str(runtime.cfg_path)
        return self.build_classification_fm_meta(
            ctx,
            candidate_labels=runtime.labels,
            extra=extra,
        )


MODEL = Siglip2Model()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
