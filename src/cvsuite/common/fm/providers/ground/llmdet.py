from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseGroundModel, RecordImageJob

DEFAULT_MODEL_ID = "iSEE-Laboratory/llmdet_base"
ALLOWED_MODEL_IDS = (
    "iSEE-Laboratory/llmdet_tiny",
    DEFAULT_MODEL_ID,
    "iSEE-Laboratory/llmdet_large",
)
PARAMS = "Swin-B default"


@dataclass(frozen=True)
class LLMDetOptions:
    model_id: str = ""


@dataclass
class LLMDetConfig:
    model_id: str = DEFAULT_MODEL_ID
    box_threshold: float = 0.4
    text_threshold: float = 0.3
    force_cpu: bool = False
    revision: str | None = None


LLMDetJob = RecordImageJob


@dataclass
class LLMDetRuntime:
    cfg: LLMDetConfig
    model_id: str
    device: torch.device
    dtype: torch.dtype
    autocast_ctx: object
    processor: AutoProcessor
    model: AutoModelForZeroShotObjectDetection


def _load_config(path: Optional[Path]) -> LLMDetConfig:
    cfg = LLMDetConfig()
    if path and path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict):
            for f in fields(LLMDetConfig):
                if f.name in data:
                    setattr(cfg, f.name, data[f.name])
    return cfg


def _resolve_model_id(cfg: LLMDetConfig, options: LLMDetOptions) -> str:
    model_id = str(options.model_id or cfg.model_id or DEFAULT_MODEL_ID).strip()
    if model_id not in ALLOWED_MODEL_IDS:
        supported = ", ".join(ALLOWED_MODEL_IDS)
        raise ValueError(f"Unsupported LLMDet model_id {model_id!r}. Supported model ids: {supported}.")
    return model_id


def _label_token_spans(input_ids: torch.Tensor, separator_id: int, special_ids: set[int]) -> list[list[int]]:
    """Token index spans for each candidate label in a merged "a. b. c." prompt sequence."""
    spans: list[list[int]] = []
    current: list[int] = []
    for idx, tok in enumerate(input_ids.tolist()):
        if tok == separator_id or tok in special_ids:
            if current:
                spans.append(current)
                current = []
        else:
            current.append(idx)
    if current:
        spans.append(current)
    return spans


def _to_device(inputs: dict[str, object], device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif key == "pixel_values":
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _setup_model(
    model_id: str,
    cfg: LLMDetConfig,
    hub_dir: Optional[Path],
    stage_dir: Optional[Path],
    caller_cwd: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
    source = utils.materialize_hf_model_source(
        model_id,
        hub_dir=hub_dir,
        stage_dir=stage_dir,
        caller_cwd=caller_cwd,
        revision=cfg.revision,
    )
    processor = AutoProcessor.from_pretrained(source.load_arg, **utils.build_hf_source_kwargs(source))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        source.load_arg,
        **utils.build_hf_source_kwargs(
            source,
            dtype=dtype,
            use_safetensors=True,
        ),
    ).to(device)
    model.eval()
    return processor, model


class LLMDetModel(BaseGroundModel[LLMDetOptions, LLMDetRuntime]):
    description = "LLMDet Swin open-vocabulary detector over VisionDataset JSON."
    options_cls = LLMDetOptions

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[LLMDetOptions]) -> LLMDetRuntime:
        cfg = _load_config(ctx.config_path)
        model_id = _resolve_model_id(cfg, ctx.options)
        device = utils.select_device(ctx.request.device, cfg.force_cpu)
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        dtype = precision.compute_dtype
        autocast_ctx = utils.maybe_autocast(device, precision)
        processor, model = _setup_model(
            model_id,
            cfg,
            ctx.hub_dir,
            getattr(ctx, "stage_dir", None),
            getattr(ctx, "caller_cwd", Path.cwd()),
            device,
            dtype,
        )
        return LLMDetRuntime(
            cfg=cfg,
            model_id=model_id,
            device=device,
            dtype=dtype,
            autocast_ctx=autocast_ctx,
            processor=processor,
            model=model,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: list[LLMDetJob],
        runtime: LLMDetRuntime,
        ctx: RuntimeContext[LLMDetOptions],
    ) -> BatchResult:
        warnings: list[str] = []
        modified: list[int] = []
        class_to_id = {name: idx for idx, name in enumerate(dataset.classes)}

        for job in batch:
            rec = dataset.records[job.record_idx]
            if not job.image_path.exists():
                warnings.append(self.missing_image_warning(dataset, job))
                continue

            try:
                with Image.open(job.image_path) as im:
                    image = im.convert("RGB")
            except Exception as exc:
                warnings.append(self.image_load_warning(dataset, job, exc))
                continue

            next_gid = self.next_group_id(rec)
            prompts = self.ground_prompts_for_record(rec)
            cls_ids = [self.ensure_class_id(dataset, prompt, class_to_id) for prompt in prompts]
            for prompt in prompts:
                if "." in prompt:
                    raise ValueError(f"LLMDet ground prompts must not contain '.': {prompt!r}")

            if prompts:
                inputs = runtime.processor(images=image, text=[prompts], return_tensors="pt")
                inputs = _to_device(dict(inputs), runtime.device, runtime.dtype)

                with torch.no_grad(), runtime.autocast_ctx:
                    outputs = runtime.model(**inputs)

                results = runtime.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=float(runtime.cfg.box_threshold),
                    text_threshold=float(runtime.cfg.text_threshold),
                    target_sizes=[image.size[::-1]],
                )

                tokenizer = runtime.processor.tokenizer
                spans = _label_token_spans(
                    inputs["input_ids"][0],
                    tokenizer.convert_tokens_to_ids("."),
                    set(tokenizer.all_special_ids),
                )
                if len(spans) != len(prompts):
                    raise ValueError(
                        f"Tokenized prompt produced {len(spans)} label spans for {len(prompts)} prompts."
                    )

                # Attribute each kept box to the label whose token span scores highest,
                # mirroring the keep mask used by post_process_grounded_object_detection.
                probs = torch.sigmoid(outputs.logits[0])
                keep = probs.max(dim=-1).values > float(runtime.cfg.box_threshold)
                span_scores = torch.stack([probs[:, span].max(dim=-1).values for span in spans], dim=-1)
                label_indices = span_scores.argmax(dim=-1)[keep].cpu().tolist()

                boxes = results[0]["boxes"].detach().cpu().numpy()
                scores = results[0]["scores"].detach().cpu().numpy()
                if len(label_indices) != len(scores):
                    raise RuntimeError(
                        f"Box/label attribution mismatch: {len(label_indices)} kept queries vs {len(scores)} boxes."
                    )
                width = float(rec.image.width) or float(image.width)
                height = float(rec.image.height) or float(image.height)
                for idx in range(len(scores)):
                    prompt = prompts[label_indices[idx]]
                    gid = next_gid
                    next_gid += 1
                    self.append_ground_box(
                        rec,
                        box_xyxy=boxes[idx],
                        width=width,
                        height=height,
                        cls_id=cls_ids[label_indices[idx]],
                        label=prompt,
                        score=float(scores[idx]),
                        group_id=gid,
                        prompt=prompt,
                    )

            tasks = rec.attributes.setdefault("fm_tasks", [])
            if "llmdet" not in tasks:
                tasks.append("llmdet")
            self.finalize_record_task(rec, set_det_when_boxes=True)
            modified.append(job.record_idx)

        self.finalize_dataset_task(
            dataset,
            has_boxes_any=any(rec.boxes for rec in dataset.records),
            has_masks_any=False,
            set_det_when_boxes=True,
        )
        return BatchResult(modified_record_indices=modified, warnings=warnings)

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: LLMDetRuntime,
        ctx: RuntimeContext[LLMDetOptions],
    ) -> dict[str, object]:
        return self.build_ground_fm_meta(
            dataset,
            ctx,
            task_name="llmdet",
            extra={
                "device": str(runtime.device),
                "hf_model": runtime.model_id,
                "config": str(ctx.config_path) if ctx.config_path else None,
            },
        )


MODEL = LLMDetModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
