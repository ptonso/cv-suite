from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseGroundModel, RecordImageJob

try:  # Optional SAM2 dependency; masks are skipped if unavailable.
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:  # pragma: no cover
    build_sam2 = None
    SAM2ImagePredictor = None

DEFAULT_MODEL_ID = "gsam"
PARAMS = "composite"


@dataclass(frozen=True)
class GSAMOptions:
    pass


@dataclass
class GSAMConfig:
    grounding_model: str = "IDEA-Research/grounding-dino-tiny"
    sam2_checkpoint: str = "checkpoints/sam2.1/sam2.1_hiera_small.pt"
    sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    box_threshold: float = 0.4
    text_threshold: float = 0.3
    force_cpu: bool = True


GSAMJob = RecordImageJob


@dataclass
class GSAMRuntime:
    cfg: GSAMConfig
    device: torch.device
    dtype: torch.dtype
    autocast_ctx: object
    processor: AutoProcessor
    gdino: AutoModelForZeroShotObjectDetection
    sam_predictor: Optional[SAM2ImagePredictor]
    resolved_cfg_path: Optional[Path]
    resolved_ckpt_path: Optional[Path]


def _load_config(path: Optional[Path]) -> GSAMConfig:
    cfg = GSAMConfig()
    if path and path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict):
            for f in fields(GSAMConfig):
                if f.name in data:
                    setattr(cfg, f.name, data[f.name])
    return cfg


def _download_sam2_checkpoint(
    model_id: str,
    filename: str,
    hub_dir: Optional[Path],
    stage_dir: Optional[Path],
    caller_cwd: Path,
) -> Optional[Path]:
    try:
        return utils.materialize_hf_file(
            model_id,
            filename,
            hub_dir=hub_dir,
            stage_dir=stage_dir,
            caller_cwd=caller_cwd,
        )
    except Exception as exc:  # pragma: no cover - best-effort download
        print(f"[gsam] failed to download {filename} from {model_id}: {exc}")
        return None


def _setup_models(
    cfg: GSAMConfig,
    hub_dir: Optional[Path],
    weights_dir: Optional[Path],
    stage_dir: Optional[Path],
    caller_cwd: Path,
    config_path: Optional[Path],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection, Optional[SAM2ImagePredictor], Optional[Path], Optional[Path]]:
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

    sam_predictor: Optional[SAM2ImagePredictor] = None
    resolved_cfg_path: Optional[Path] = None
    resolved_ckpt_path: Optional[Path] = None
    if SAM2ImagePredictor is not None and build_sam2 is not None:
        sam2_pkg = None
        try:
            mod = importlib.import_module("sam2")
            sam2_pkg = Path(mod.__file__).resolve().parent
        except Exception:
            sam2_pkg = None

        model_cache_env = os.environ.get("FM_MODEL_CACHE")
        model_cache = Path(model_cache_env) if model_cache_env else None
        if stage_dir:
            sam2_root_base = stage_dir
        elif weights_dir:
            sam2_root_base = weights_dir
        elif model_cache:
            sam2_root_base = model_cache
        else:
            sam2_root_base = caller_cwd
        sam2_root = sam2_root_base / "sam2"
        sam2_root.mkdir(parents=True, exist_ok=True)
        print(f"[gsam] SAM2 root: {sam2_root}")
        legacy_root = (model_cache / "sam2") if model_cache and model_cache != sam2_root_base else None

        def _first_existing(paths: List[Path]) -> Optional[Path]:
            for p in paths:
                if p.exists():
                    return p
            return None

        cfg_rel = Path(cfg.sam2_model_config)
        cfg_candidates: List[Path] = []
        if cfg_rel.is_absolute():
            cfg_candidates.append(cfg_rel)
        if sam2_pkg:
            cfg_candidates.append(sam2_pkg / cfg.sam2_model_config)
            cfg_candidates.append(sam2_pkg / "configs" / "sam2.1" / cfg_rel.name)
        if not cfg_rel.is_absolute():
            cfg_candidates.append(sam2_root / cfg_rel)
            cfg_candidates.append(sam2_root / "configs" / "sam2.1" / cfg_rel.name)
            if legacy_root and legacy_root != sam2_root:
                cfg_candidates.append(legacy_root / cfg_rel)
                cfg_candidates.append(legacy_root / "configs" / "sam2.1" / cfg_rel.name)
            base = caller_cwd
            if config_path:
                base = config_path.parent
            cfg_candidates.append(base / cfg_rel)

        resolved_cfg_path = _first_existing(cfg_candidates)
        cfg_path = resolved_cfg_path or (cfg_candidates[0] if cfg_candidates else None)

        ckpt_rel = Path(cfg.sam2_checkpoint)
        ckpt_candidates: List[Path] = []
        if ckpt_rel.is_absolute():
            ckpt_candidates.append(ckpt_rel)
        else:
            ckpt_candidates.append(sam2_root / ckpt_rel)
            ckpt_candidates.append(sam2_root / ckpt_rel.name)
            if legacy_root and legacy_root != sam2_root:
                ckpt_candidates.append(legacy_root / ckpt_rel)
                ckpt_candidates.append(legacy_root / ckpt_rel.name)
            base = caller_cwd
            if config_path:
                base = config_path.parent
            ckpt_candidates.append(base / ckpt_rel)

        resolved_ckpt_path = _first_existing(ckpt_candidates)
        ckpt_path = resolved_ckpt_path or (ckpt_candidates[0] if ckpt_candidates else None)
        if ckpt_path and resolved_ckpt_path is None:
            downloaded = _download_sam2_checkpoint(
                "facebook/sam2.1-hiera-small",
                ckpt_path.name,
                hub_dir,
                stage_dir,
                caller_cwd,
            )
            if downloaded and downloaded.exists():
                resolved_ckpt_path = downloaded

        cfg_display = resolved_cfg_path.resolve() if resolved_cfg_path else "missing"
        ckpt_display = resolved_ckpt_path.resolve() if resolved_ckpt_path else "missing"
        print(f"[gsam] SAM2 config requested: {cfg.sam2_model_config}")
        print(f"[gsam] SAM2 config resolved: {cfg_display}")
        print(f"[gsam] SAM2 checkpoint requested: {cfg.sam2_checkpoint}")
        print(f"[gsam] SAM2 checkpoint resolved: {ckpt_display}")

        if resolved_cfg_path and resolved_ckpt_path:
            config_name = cfg.sam2_model_config
            if sam2_pkg:
                try:
                    rel = resolved_cfg_path.relative_to(sam2_pkg)
                    config_name = rel.as_posix()
                except ValueError:
                    if (sam2_pkg / cfg.sam2_model_config).exists():
                        config_name = cfg.sam2_model_config
                    else:
                        config_name = resolved_cfg_path.name
            print(f"[gsam] SAM2 config name: {config_name}")
            print(f"Loaded SAM2 config: {resolved_cfg_path}")
            print(f"Loaded SAM2 checkpoint: {resolved_ckpt_path}")
            sam_model = build_sam2(config_name, str(resolved_ckpt_path), device=str(device))
            sam_predictor = SAM2ImagePredictor(sam_model)
        else:
            print(
                "[gsam] SAM2 assets missing; "
                f"config={bool(resolved_cfg_path)} ckpt={bool(resolved_ckpt_path)} "
                f"(config: {cfg_path}, ckpt: {ckpt_path})"
            )

    if sam_predictor is None:
        print("Could not load sam2 model.")

    return processor, gdino, sam_predictor, resolved_cfg_path, resolved_ckpt_path


class GSAMModel(BaseGroundModel[GSAMOptions, GSAMRuntime]):
    description = "Grounded-SAM2 wrapper: grounded detections/segmentations over VisionDataset JSON."
    options_cls = GSAMOptions

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[GSAMOptions]) -> GSAMRuntime:
        cfg = _load_config(ctx.config_path)
        device = utils.select_device(ctx.request.device, cfg.force_cpu)
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        dtype = precision.compute_dtype
        autocast_ctx = utils.maybe_autocast(device, precision)
        processor, gdino, sam_predictor, resolved_cfg_path, resolved_ckpt_path = _setup_models(
            cfg,
            ctx.hub_dir,
            ctx.weights_dir,
            getattr(ctx, "stage_dir", None),
            getattr(ctx, "caller_cwd", Path.cwd()),
            ctx.config_path,
            device,
            dtype,
        )
        return GSAMRuntime(
            cfg=cfg,
            device=device,
            dtype=dtype,
            autocast_ctx=autocast_ctx,
            processor=processor,
            gdino=gdino,
            sam_predictor=sam_predictor,
            resolved_cfg_path=resolved_cfg_path,
            resolved_ckpt_path=resolved_ckpt_path,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: List[GSAMJob],
        runtime: GSAMRuntime,
        ctx: RuntimeContext[GSAMOptions],
    ) -> BatchResult:
        warnings: List[str] = []
        modified: List[int] = []
        class_to_id = {name: idx for idx, name in enumerate(dataset.classes)}
        has_masks_any = bool(getattr(dataset.task, "value", dataset.task) == "seg")

        for job in batch:
            rec = dataset.records[job.record_idx]
            if not job.image_path.exists():
                warnings.append(f"[gsam] missing image src={rec.image.path} resolved={job.image_path}")
                continue

            try:
                with Image.open(job.image_path) as im:
                    image = im.convert("RGB")
            except Exception as exc:
                warnings.append(f"[gsam] failed to load image src={rec.image.path} resolved={job.image_path}: {exc}")
                continue

            try:
                next_gid = self.next_group_id(rec)
                pixel_inputs = runtime.processor(images=image, return_tensors="pt")
                pixel_values = pixel_inputs["pixel_values"].to(device=runtime.device, dtype=runtime.dtype)

                if runtime.sam_predictor is not None:
                    runtime.sam_predictor.set_image(np.array(image.convert("RGB")))

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

                    masks = None
                    if runtime.sam_predictor is not None and boxes.size > 0:
                        masks_pred, _, _ = runtime.sam_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=boxes,
                            multimask_output=False,
                        )
                        if masks_pred.ndim == 4:
                            masks_pred = masks_pred.squeeze(1)
                        masks = masks_pred

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

                        if masks is not None and idx < len(masks):
                            polys = utils.mask_to_polygons(masks[idx])
                            if self.append_ground_polygons(
                                rec,
                                polys=polys,
                                width=W,
                                height=H,
                                cls_id=cls_id,
                                label=prompt,
                                score=float(scores[idx]),
                                group_id=gid,
                                prompt=prompt,
                            ):
                                has_masks_any = True

                tasks = rec.attributes.setdefault("fm_tasks", [])
                if "grounded-sam2" not in tasks:
                    tasks.append("grounded-sam2")
                self.finalize_record_task(rec, set_det_when_boxes=False)
                modified.append(job.record_idx)
            except Exception as exc:
                warnings.append(f"[gsam] failed to process image src={rec.image.path} resolved={job.image_path}: {exc}")
                continue

        self.finalize_dataset_task(dataset, has_boxes_any=any(rec.boxes for rec in dataset.records), has_masks_any=has_masks_any, set_det_when_boxes=False)
        return BatchResult(modified_record_indices=modified, warnings=warnings)

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: GSAMRuntime,
        ctx: RuntimeContext[GSAMOptions],
    ) -> dict[str, object]:
        return self.build_ground_fm_meta(
            dataset,
            ctx,
            task_name="grounded-sam2",
            extra={
                "device": str(runtime.device),
                "config": str(ctx.config_path) if ctx.config_path else None,
                "config_resolved": str(runtime.resolved_cfg_path) if runtime.resolved_cfg_path else None,
                "checkpoint": str(runtime.resolved_ckpt_path) if runtime.resolved_ckpt_path else None,
            },
        )


MODEL = GSAMModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
