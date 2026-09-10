from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import yaml
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseGroundModel, RecordImageJob

DEFAULT_MODEL_ID = "gdino"
PARAMS = "composite"


@dataclass(frozen=True)
class GDINOOptions:
    pass


@dataclass
class GDINOConfig:
    grounding_model: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.4
    text_threshold: float = 0.3
    force_cpu: bool = False


GDINOJob = RecordImageJob


@dataclass
class GDINORuntime:
    cfg: GDINOConfig
    device: torch.device
    dtype: torch.dtype
    autocast_ctx: object
    processor: AutoProcessor
    gdino: AutoModelForZeroShotObjectDetection


def _load_config(path: Optional[Path]) -> GDINOConfig:
    cfg = GDINOConfig()
    if path and path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict):
            for f in fields(GDINOConfig):
                if f.name in data:
                    setattr(cfg, f.name, data[f.name])
    return cfg


def _setup_model(
    cfg: GDINOConfig,
    hub_dir: Optional[Path],
    stage_dir: Optional[Path],
    caller_cwd: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
    source = utils.materialize_hf_model_source(
        cfg.grounding_model,
        hub_dir=hub_dir,
        stage_dir=stage_dir,
        caller_cwd=caller_cwd,
    )
    processor = AutoProcessor.from_pretrained(source.load_arg, **utils.build_hf_source_kwargs(source))
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
        source.load_arg,
        **utils.build_hf_source_kwargs(
            source,
            dtype=dtype,
        ),
    ).to(device)
    gdino.eval()
    return processor, gdino


class GDINOModel(BaseGroundModel[GDINOOptions, GDINORuntime]):
    description = "Grounding-DINO detector: grounded bounding boxes over VisionDataset JSON."
    options_cls = GDINOOptions

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[GDINOOptions]) -> GDINORuntime:
        cfg = _load_config(ctx.config_path)
        device = utils.select_device(ctx.request.device, cfg.force_cpu)
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        dtype = precision.compute_dtype
        autocast_ctx = utils.maybe_autocast(device, precision)
        processor, gdino = _setup_model(
            cfg,
            ctx.hub_dir,
            getattr(ctx, "stage_dir", None),
            getattr(ctx, "caller_cwd", Path.cwd()),
            device,
            dtype,
        )
        return GDINORuntime(
            cfg=cfg,
            device=device,
            dtype=dtype,
            autocast_ctx=autocast_ctx,
            processor=processor,
            gdino=gdino,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: List[GDINOJob],
        runtime: GDINORuntime,
        ctx: RuntimeContext[GDINOOptions],
    ) -> BatchResult:
        warnings: List[str] = []
        modified: List[int] = []
        class_to_id = {name: idx for idx, name in enumerate(dataset.classes)}

        for job in batch:
            rec = dataset.records[job.record_idx]
            if not job.image_path.exists():
                warnings.append(f"[gdino] missing image src={rec.image.path} resolved={job.image_path}")
                continue

            try:
                with Image.open(job.image_path) as im:
                    image = im.convert("RGB")
            except Exception as exc:
                warnings.append(f"[gdino] failed to load image src={rec.image.path} resolved={job.image_path}: {exc}")
                continue

            try:
                next_gid = self.next_group_id(rec)
                pixel_inputs = runtime.processor(images=image, return_tensors="pt")
                pixel_values = pixel_inputs["pixel_values"].to(device=runtime.device, dtype=runtime.dtype)

                for prompt in self.ground_prompts_for_record(rec):
                    cls_id = self.ensure_class_id(dataset, prompt, class_to_id)

                    text_prompt = utils.normalize_prompt_text(prompt)
                    text_inputs = runtime.processor(text=text_prompt, return_tensors="pt")
                    inputs = {**text_inputs, "pixel_values": pixel_values}
                    inputs = {k: v.to(device=runtime.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                    with torch.no_grad(), runtime.autocast_ctx:
                        outputs = runtime.gdino(**inputs)

                    results = runtime.processor.post_process_grounded_object_detection(
                        outputs,
                        inputs["input_ids"],
                        threshold=float(runtime.cfg.box_threshold),
                        text_threshold=float(runtime.cfg.text_threshold),
                        target_sizes=[image.size[::-1]],
                    )
                    if not results or results[0].get("boxes") is None:
                        continue

                    boxes = results[0]["boxes"].detach().cpu().numpy()
                    scores = results[0]["scores"].detach().cpu().numpy()

                    W = float(rec.image.width) or float(image.width)
                    H = float(rec.image.height) or float(image.height)
                    for idx in range(len(scores)):
                        gid = next_gid
                        next_gid += 1
                        self.append_ground_box(
                            rec,
                            box_xyxy=boxes[idx],
                            width=W,
                            height=H,
                            cls_id=cls_id,
                            label=prompt,
                            score=float(scores[idx]),
                            group_id=gid,
                            prompt=prompt,
                        )

                tasks = rec.attributes.setdefault("fm_tasks", [])
                if "grounded-dino" not in tasks:
                    tasks.append("grounded-dino")
                self.finalize_record_task(rec, set_det_when_boxes=True)
                modified.append(job.record_idx)
            except Exception as exc:
                warnings.append(f"[gdino] failed to process image src={rec.image.path} resolved={job.image_path}: {exc}")
                continue

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
        runtime: GDINORuntime,
        ctx: RuntimeContext[GDINOOptions],
    ) -> dict[str, object]:
        return self.build_ground_fm_meta(
            dataset,
            ctx,
            task_name="grounded-dino",
            extra={
                "device": str(runtime.device),
                "grounding_model": runtime.cfg.grounding_model,
                "config": str(ctx.config_path) if ctx.config_path else None,
            },
        )


MODEL = GDINOModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
