from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseClassificationModel, RecordImageJob

DEFAULT_MODEL_ID = "DuarteBarbosa/deep-image-orientation-detection"
PARAMS = "unknown"
HF_CHECKPOINT = "orientation_model_v2_0.9882.pth"
IMAGE_SIZE = 384
RESIZE_SIZE = IMAGE_SIZE + 32
NUM_CLASSES = 4
CLASS_LABELS = (
    "correct",
    "rotate_90_clockwise",
    "rotate_180",
    "rotate_90_counter_clockwise",
)
CLASS_DESCRIPTIONS = {
    0: "Image is correctly oriented (0°).",
    1: "Image needs to be rotated 90° Clockwise to be correct.",
    2: "Image needs to be rotated 180° to be correct.",
    3: "Image needs to be rotated 90° Counter-Clockwise to be correct.",
}
LABEL_DESCRIPTIONS = {CLASS_LABELS[idx]: CLASS_DESCRIPTIONS[idx] for idx in range(NUM_CLASSES)}
CLASS_TO_CCW_DEGREES = {
    0: 0,
    1: 270,
    2: 180,
    3: 90,
}


@dataclass(frozen=True)
class DeepOrientationOptions:
    threshold: float | None = None


DeepOrientationJob = RecordImageJob


@dataclass
class DeepOrientationRuntime:
    device: torch.device
    autocast_ctx: object
    model: torch.nn.Module
    transform: transforms.Compose


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_image_safely(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGB", "L"}:
            return image.convert("RGB")

        rgba_image = image.convert("RGBA")
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))
        background.paste(rgba_image, mask=rgba_image)
        rgba_image.close()
        return background


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = str(key)
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod.") :]
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        normalized[clean_key] = value
    return normalized


def _extract_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        candidate = payload.get("state_dict") if "state_dict" in payload else payload
        if isinstance(candidate, dict):
            return _normalize_state_dict_keys(candidate)
    raise RuntimeError("Orientation checkpoint did not contain a loadable state_dict.")


def _download_checkpoint(hub_dir: Path | None, stage_dir: Path | None, caller_cwd: Path) -> Path:
    try:
        return utils.materialize_hf_file(
            DEFAULT_MODEL_ID,
            HF_CHECKPOINT,
            hub_dir=hub_dir,
            stage_dir=stage_dir,
            caller_cwd=caller_cwd,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the deep orientation checkpoint. If this is the first run, "
            "an internet connection is required once so the weights can be cached locally."
        ) from exc


def _build_model() -> torch.nn.Module:
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def _load_model(hub_dir: Path | None, stage_dir: Path | None, caller_cwd: Path, device: torch.device) -> torch.nn.Module:
    checkpoint_path = _download_checkpoint(hub_dir, stage_dir, caller_cwd)
    state_dict = _extract_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = _build_model()
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model


def _classification_meta(
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    top_idx: int,
) -> dict[str, object]:
    return {
        "model": "deep_orientation",
        "hf_model": DEFAULT_MODEL_ID,
        "checkpoint": HF_CHECKPOINT,
        "device": str(device),
        "precision": precision,
        "batch_size": batch_size,
        "candidate_labels": list(CLASS_LABELS),
        "candidate_descriptions": dict(LABEL_DESCRIPTIONS),
        "class_index": top_idx,
        "class_label": CLASS_LABELS[top_idx],
        "class_description": CLASS_DESCRIPTIONS[top_idx],
        "correction_degrees_ccw": CLASS_TO_CCW_DEGREES[top_idx],
        "correction_degrees_cw": (360 - CLASS_TO_CCW_DEGREES[top_idx]) % 360,
    }


class DeepOrientationModel(BaseClassificationModel[DeepOrientationOptions, DeepOrientationRuntime]):
    description = "Deep image orientation wrapper: consumes in.json, writes out.json with 4-way orientation predictions."
    options_cls = DeepOrientationOptions
    allow_root_basename = True

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[DeepOrientationOptions]) -> DeepOrientationRuntime:
        device = utils.select_device(ctx.request.device)
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        model = _load_model(
            ctx.hub_dir,
            ctx.stage_dir,
            ctx.caller_cwd,
            device,
        )
        autocast_ctx = utils.maybe_autocast_for_module(device, precision, model)
        return DeepOrientationRuntime(
            device=device,
            autocast_ctx=autocast_ctx,
            model=model,
            transform=_build_transform(),
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: List[DeepOrientationJob],
        runtime: DeepOrientationRuntime,
        ctx: RuntimeContext[DeepOrientationOptions],
    ) -> BatchResult:
        loaded, warnings = self.collect_loaded_records(dataset, batch, _load_image_safely)
        tensors: List[Tuple[DeepOrientationJob, torch.Tensor]] = []
        modified: List[int] = []
        for job, image in loaded:
            try:
                tensors.append((job, runtime.transform(image)))
            finally:
                image.close()
        if not tensors:
            return BatchResult(modified_record_indices=modified, warnings=warnings)

        batch_tensor = torch.stack([tensor for _, tensor in tensors]).to(device=runtime.device)
        with torch.inference_mode(), runtime.autocast_ctx:
            logits = runtime.model(batch_tensor)
        probs_batch = torch.softmax(logits.detach().cpu().to(torch.float32), dim=-1)

        for (job, _), probs in zip(tensors, probs_batch):
            top_idx = int(torch.argmax(probs).item())
            top_label = CLASS_LABELS[top_idx]
            top_score = float(probs[top_idx].item())
            modified.append(
                self.store_classification_result(
                    dataset,
                    job,
                    ctx=ctx,
                    label=top_label,
                    score=top_score,
                    probs={label: float(probs[label_idx].item()) for label_idx, label in enumerate(CLASS_LABELS)},
                    meta=_classification_meta(
                        device=runtime.device,
                        precision=ctx.request.precision,
                        batch_size=ctx.batch_size,
                        top_idx=top_idx,
                    ),
                )
            )

        return BatchResult(modified_record_indices=modified, warnings=warnings)

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: DeepOrientationRuntime,
        ctx: RuntimeContext[DeepOrientationOptions],
    ) -> dict[str, object]:
        return self.build_classification_fm_meta(
            ctx,
            candidate_labels=list(CLASS_LABELS),
            extra={
                "hf_model": DEFAULT_MODEL_ID,
                "checkpoint": HF_CHECKPOINT,
                "device": str(runtime.device),
                "weights": str(ctx.weights_dir) if ctx.weights_dir else None,
                "candidate_descriptions": dict(LABEL_DESCRIPTIONS),
            },
        )


MODEL = DeepOrientationModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
