from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseGroundModel, RecordImageJob

DEFAULT_MODEL_ID = "yolo_e"
PARAMS = "26l"
MODEL_ALIASES = ("yoloe", "yolo-e")


@dataclass(frozen=True)
class YoloEOptions:
    reference_dataset_path: str = ""
    confidence_threshold: float = 0.25


@dataclass
class YoloEConfig:
    model_id: str = "yoloe-26l-seg"


YoloEJob = RecordImageJob


@dataclass
class YoloERuntime:
    cfg: YoloEConfig
    model: object
    device_arg: object
    mode: str  # "text" | "reference"
    class_names: List[str]
    confidence: float
    refer_image: Optional[Path]
    visual_prompts: Optional[dict]
    vp_predictor: Optional[type]


def _load_config(path: Optional[Path]) -> YoloEConfig:
    cfg = YoloEConfig()
    if path and path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict):
            for f in fields(YoloEConfig):
                if f.name in data:
                    setattr(cfg, f.name, data[f.name])
    return cfg


def _fm_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _legacy_weight_roots(ctx: RuntimeContext[YoloEOptions]) -> List[Path]:
    roots: List[Path] = []
    for raw in (Path.cwd(), _fm_root(), getattr(ctx, "caller_cwd", None)):
        if raw is None:
            continue
        path = Path(raw)
        if path not in roots:
            roots.append(path)
    return roots


def _require_weights_dir(ctx: RuntimeContext[YoloEOptions]) -> Path:
    weights_dir = getattr(ctx, "weights_dir", None)
    if weights_dir is None:
        raise ValueError("YOLO-E requires a weights_dir so checkpoint downloads are durable.")
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def _weight_filename(model_id: str) -> str:
    name = Path(str(model_id or "").strip()).name
    if not name:
        raise ValueError("YOLO-E model_id must not be empty.")
    return name if name.endswith(".pt") else f"{name}.pt"


def _adopt_legacy_weight(filename: str, target: Path, ctx: RuntimeContext[YoloEOptions]) -> None:
    if target.exists():
        return
    for root in _legacy_weight_roots(ctx):
        source = root / filename
        try:
            if source.resolve() == target.resolve():
                continue
        except OSError:
            pass
        if not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            source.replace(target)
        except OSError:
            shutil.copy2(source, target)
        return


@contextmanager
def _ultralytics_weights_cwd(weights_dir: Path):
    previous = Path.cwd()
    os.chdir(weights_dir)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_weights(cfg: YoloEConfig, ctx: RuntimeContext[YoloEOptions]) -> str:
    weights_dir = _require_weights_dir(ctx)
    filename = _weight_filename(cfg.model_id)
    target = weights_dir / filename
    _adopt_legacy_weight(filename, target, ctx)
    return str(target)


def _device_arg(device: torch.device) -> object:
    if device.type == "cpu":
        return "cpu"
    return device.index if device.index is not None else 0


def _reference_label(record, classes: List[str], box) -> str:
    label = getattr(box, "label", None)
    if isinstance(label, str) and label.strip():
        return label.strip()
    cls_idx = int(getattr(box, "cls", -1))
    if 0 <= cls_idx < len(classes):
        return str(classes[cls_idx])
    return str(cls_idx)


def _build_visual_prompts(reference: VisionDataset) -> tuple[Path, dict, List[str]]:
    record = reference.records[0]
    if not record.boxes:
        raise ValueError("Reference image has no bounding-box labels for visual prompting.")
    width = float(record.image.width)
    height = float(record.image.height)
    if width <= 0 or height <= 0:
        raise ValueError("Reference image is missing valid width/height.")

    class_names: List[str] = []
    name_to_id: dict[str, int] = {}
    bboxes: List[List[float]] = []
    cls_ids: List[int] = []
    for box in record.boxes:
        label = _reference_label(record, list(reference.classes), box)
        if label not in name_to_id:
            name_to_id[label] = len(class_names)
            class_names.append(label)
        half_w = float(box.w) * width / 2.0
        half_h = float(box.h) * height / 2.0
        cx = float(box.cx) * width
        cy = float(box.cy) * height
        bboxes.append([cx - half_w, cy - half_h, cx + half_w, cy + half_h])
        cls_ids.append(name_to_id[label])

    visual_prompts = dict(
        bboxes=np.array(bboxes, dtype=np.float32),
        cls=np.array(cls_ids, dtype=np.int64),
    )
    return record.image.path, visual_prompts, class_names


class YoloEModel(BaseGroundModel[YoloEOptions, YoloERuntime]):
    description = "YOLO-E wrapper: open-vocabulary grounded detections/segmentations over VisionDataset JSON, via text prompts OR a single reference image+labels."
    options_cls = YoloEOptions

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[YoloEOptions]) -> YoloERuntime:
        cfg = _load_config(ctx.config_path)
        device = utils.select_device(ctx.request.device)

        reference_path = str(ctx.options.reference_dataset_path or "").strip()
        prompts = self.dataset_ground_prompts(dataset)
        if reference_path and prompts:
            raise ValueError("YOLO-E accepts either text prompts or a reference image, not both.")
        if not reference_path and not prompts:
            raise ValueError("YOLO-E requires either text prompts or a reference image.")

        from ultralytics import YOLOE

        weights_dir = _require_weights_dir(ctx)
        for filename in ("mobileclip_blt.ts", "mobileclip2_b.ts"):
            _adopt_legacy_weight(filename, weights_dir / filename, ctx)
        with _ultralytics_weights_cwd(weights_dir):
            model = YOLOE(_resolve_weights(cfg, ctx))

        refer_image: Optional[Path] = None
        visual_prompts: Optional[dict] = None
        vp_predictor: Optional[type] = None
        if reference_path:
            mode = "reference"
            reference = VisionDataset.from_json(Path(reference_path))
            if len(reference.records) != 1:
                raise ValueError(
                    f"YOLO-E reference mode expects exactly one reference image; found {len(reference.records)}."
                )
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

            refer_image, visual_prompts, class_names = _build_visual_prompts(reference)
            vp_predictor = YOLOEVPSegPredictor
        else:
            mode = "text"
            class_names = list(prompts)
            with _ultralytics_weights_cwd(weights_dir):
                model.set_classes(class_names, model.get_text_pe(class_names))

        return YoloERuntime(
            cfg=cfg,
            model=model,
            device_arg=_device_arg(device),
            mode=mode,
            class_names=class_names,
            confidence=float(ctx.options.confidence_threshold),
            refer_image=refer_image,
            visual_prompts=visual_prompts,
            vp_predictor=vp_predictor,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: List[YoloEJob],
        runtime: YoloERuntime,
        ctx: RuntimeContext[YoloEOptions],
    ) -> BatchResult:
        warnings: List[str] = []
        modified: List[int] = []
        class_to_id = {name: idx for idx, name in enumerate(dataset.classes)}
        has_masks_any = bool(getattr(dataset.task, "value", dataset.task) == "seg")

        for job in batch:
            rec = dataset.records[job.record_idx]
            if not job.image_path.exists():
                warnings.append(self.missing_image_warning(dataset, job))
                continue

            try:
                predict_kwargs = dict(
                    source=str(job.image_path),
                    conf=runtime.confidence,
                    device=runtime.device_arg,
                    verbose=False,
                )
                if runtime.mode == "reference":
                    predict_kwargs.update(
                        refer_image=str(runtime.refer_image),
                        visual_prompts=runtime.visual_prompts,
                        predictor=runtime.vp_predictor,
                    )
                results = runtime.model.predict(**predict_kwargs)
            except Exception as exc:
                warnings.append(f"[yolo_e] failed to predict idx={job.record_idx} path={job.image_path}: {exc}")
                continue

            if not results:
                continue
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                modified.append(job.record_idx)
                continue

            height, width = (float(v) for v in result.orig_shape)
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            cls_idx = boxes.cls.detach().cpu().numpy().astype(int)
            masks = getattr(result, "masks", None)
            mask_polys = masks.xy if masks is not None else None

            next_gid = self.next_group_id(rec)
            for i in range(len(xyxy)):
                cid = int(cls_idx[i])
                label = runtime.class_names[cid] if 0 <= cid < len(runtime.class_names) else str(cid)
                cls_id = self.ensure_class_id(dataset, label, class_to_id)
                gid = next_gid
                next_gid += 1
                self.append_ground_box(
                    rec,
                    box_xyxy=xyxy[i],
                    width=width,
                    height=height,
                    cls_id=cls_id,
                    label=label,
                    score=float(confs[i]),
                    group_id=gid,
                    prompt=label,
                )
                if mask_polys is not None and i < len(mask_polys) and len(mask_polys[i]) >= 3:
                    poly = [(float(x), float(y)) for x, y in mask_polys[i]]
                    if self.append_ground_polygons(
                        rec,
                        polys=[poly],
                        width=width,
                        height=height,
                        cls_id=cls_id,
                        label=label,
                        score=float(confs[i]),
                        group_id=gid,
                        prompt=label,
                    ):
                        has_masks_any = True

            tasks = rec.attributes.setdefault("fm_tasks", [])
            if "yolo-e" not in tasks:
                tasks.append("yolo-e")
            self.finalize_record_task(rec, set_det_when_boxes=False)
            modified.append(job.record_idx)

        self.finalize_dataset_task(
            dataset,
            has_boxes_any=any(rec.boxes for rec in dataset.records),
            has_masks_any=has_masks_any,
            set_det_when_boxes=False,
        )
        return BatchResult(modified_record_indices=modified, warnings=warnings)

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: YoloERuntime,
        ctx: RuntimeContext[YoloEOptions],
    ) -> dict[str, object]:
        return self.build_ground_fm_meta(
            dataset,
            ctx,
            task_name="yolo-e",
            extra={
                "mode": runtime.mode,
                "model_id": runtime.cfg.model_id,
                "device": str(runtime.device_arg),
                "reference_classes": list(runtime.class_names) if runtime.mode == "reference" else [],
            },
        )


MODEL = YoloEModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
