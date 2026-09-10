from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import os
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from PIL import Image

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseGroundModel, RecordImageJob

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPT_DIR_STR = str(_SCRIPT_DIR)
while _SCRIPT_DIR_STR in sys.path:
    sys.path.remove(_SCRIPT_DIR_STR)

_SAM3_IMPORT_ERROR: Exception | None = None
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except Exception as exc:  # pragma: no cover - optional dependency
    build_sam3_image_model = None
    Sam3Processor = None
    _SAM3_IMPORT_ERROR = exc

DEFAULT_MODEL_ID = "sam3"
PARAMS = "unknown"
_SAM3_FUSED_ADDMM_PATCHED = False


@dataclass(frozen=True)
class Sam3Options:
    confidence_threshold: float = 0.5


Sam3Job = RecordImageJob


@dataclass
class Sam3Runtime:
    device: torch.device
    autocast_ctx: object
    processor: Sam3Processor
    managed_auto_fallback: str | None = None


def _wrap_sam3_fused_addmm_act(addmm_act_func):
    def _wrapped(activation, linear, mat1):
        output = addmm_act_func(activation, linear, mat1)
        weight = getattr(linear, "weight", None)
        target_dtype = None
        if isinstance(weight, torch.Tensor) and weight.is_floating_point() and weight.device.type != "meta":
            target_dtype = weight.dtype
        elif isinstance(mat1, torch.Tensor) and mat1.is_floating_point() and mat1.device.type != "meta":
            target_dtype = mat1.dtype
        if (
            target_dtype is not None
            and isinstance(output, torch.Tensor)
            and output.is_floating_point()
            and output.dtype != target_dtype
        ):
            output = output.to(dtype=target_dtype)
        return output

    setattr(_wrapped, "_cvsuite_preserves_output_dtype", True)
    return _wrapped


def _patch_sam3_fused_addmm_dtype_compat() -> None:
    global _SAM3_FUSED_ADDMM_PATCHED
    if _SAM3_FUSED_ADDMM_PATCHED:
        return
    fused_mod = importlib.import_module("sam3.perflib.fused")
    vitdet_mod = importlib.import_module("sam3.model.vitdet")
    current = getattr(fused_mod, "addmm_act", None)
    if not callable(current):
        return
    if getattr(current, "_cvsuite_preserves_output_dtype", False):
        _SAM3_FUSED_ADDMM_PATCHED = True
        return
    wrapped = _wrap_sam3_fused_addmm_act(current)
    fused_mod.addmm_act = wrapped
    vitdet_mod.addmm_act = wrapped
    _SAM3_FUSED_ADDMM_PATCHED = True


def _cuda_arch_list() -> set[str]:
    arch_flags = getattr(torch._C, "_cuda_getArchFlags", lambda: "")()
    arch_list = {flag for flag in arch_flags.split() if flag.startswith("sm_")}
    return arch_list or set(torch.cuda.get_arch_list())


def _accelerate_infer_auto_device_map(model: Any, *, max_memory: dict[int | str, int | str] | None) -> dict[str, str | int]:
    from accelerate import infer_auto_device_map

    return infer_auto_device_map(model, max_memory=max_memory)


def _accelerate_dispatch_model(
    model: Any,
    *,
    device_map: dict[str, str | int],
    offload_dir: str | None,
) -> Any:
    from accelerate import dispatch_model

    return dispatch_model(model, device_map=device_map, offload_dir=offload_dir)


def _device_map_targets(device_map: dict[str, str | int]) -> set[str]:
    targets: set[str] = set()
    for raw in device_map.values():
        if isinstance(raw, int):
            targets.add(f"cuda:{raw}")
        else:
            targets.add(str(raw))
    return targets


@contextmanager
def _patch_sam3_cpu_build_precompute():
    position_encoding_mod = importlib.import_module("sam3.model.position_encoding")
    decoder_mod = importlib.import_module("sam3.model.decoder")

    position_encoding_cls = position_encoding_mod.PositionEmbeddingSine
    transformer_decoder_cls = decoder_mod.TransformerDecoder

    original_position_init = position_encoding_cls.__init__
    original_decoder_init = transformer_decoder_cls.__init__

    def _patched_position_init(self, *args, **kwargs):
        if kwargs.get("precompute_resolution") is not None:
            kwargs = dict(kwargs)
            kwargs["precompute_resolution"] = None
        return original_position_init(self, *args, **kwargs)

    def _patched_decoder_init(self, *args, **kwargs):
        if kwargs.get("resolution") is not None and kwargs.get("stride") is not None:
            kwargs = dict(kwargs)
            kwargs["resolution"] = None
            kwargs["stride"] = None
        return original_decoder_init(self, *args, **kwargs)

    position_encoding_cls.__init__ = _patched_position_init
    transformer_decoder_cls.__init__ = _patched_decoder_init
    try:
        yield
    finally:
        position_encoding_cls.__init__ = original_position_init
        transformer_decoder_cls.__init__ = original_decoder_init


class Sam3Model(BaseGroundModel[Sam3Options, Sam3Runtime]):
    description = "SAM3 wrapper: prompt-conditioned boxes/masks over a VisionDataset JSON."
    options_cls = Sam3Options
    supports_managed_auto_device = True

    def _build_sam3_model(self, *, device_arg: str) -> Any:
        try:
            if device_arg == "cpu":
                # Upstream SAM3 eagerly precomputes a few helper caches on CUDA during
                # model construction, even for CPU builds. Those caches are later reused
                # verbatim and can trigger mixed-device inference errors, so disable the
                # eager precompute step and let the model warm the caches lazily on the
                # actual runtime device instead.
                with _patch_sam3_cpu_build_precompute():
                    return build_sam3_image_model(device=device_arg)
            return build_sam3_image_model(device=device_arg)
        except Exception as exc:
            access_error = utils.describe_hf_access_error(exc, "facebook/sam3")
            if access_error is not None:
                raise RuntimeError(
                    f"{access_error} cvsuite will also read those credentials from a nearby .env file."
                ) from exc
            raise

    def _load_auto_dispatched_model(
        self,
        dataset: VisionDataset,
        ctx: RuntimeContext[Sam3Options],
        *,
        device: torch.device,
        precision: utils.ResolvedPrecision,
    ) -> tuple[Any, torch.device, utils.ResolvedPrecision, str | None]:
        self.ensure_optional_dependency(
            package_name="accelerate>=1.6,<2",
            module_name="accelerate",
            ctx=ctx,
            reason="managed auto device placement for sam3",
        )
        _load_kwargs, placement = self.build_hf_load_kwargs(
            ctx,
            device,
            precision,
            include_default_device_map=False,
        )
        print("[sam3] building model on CPU for managed auto placement")
        model = self._build_sam3_model(device_arg="cpu")
        print("[sam3] inferring managed auto device map")
        resolved_device_map = _accelerate_infer_auto_device_map(
            model,
            max_memory=placement.max_memory,
        )
        targets = _device_map_targets(resolved_device_map)
        if any(target in {"cpu", "disk"} for target in targets):
            message = (
                "sam3 auto placement would require mixed GPU/CPU offload, but the upstream SAM3 "
                "processor uses custom tensor containers that do not move reliably through accelerate hooks yet. "
                "Falling back to CPU execution for correctness."
            )
            self.emit_runtime_warning(dataset, f"[sam3] {message}")
            print(f"[sam3] inferred auto device map targets: {', '.join(sorted(targets))}")
            print("[sam3] falling back to CPU execution")
            self._last_hf_load_placement = None
            cpu_device = torch.device("cpu")
            cpu_precision = self.resolve_runtime_precision(dataset, ctx, cpu_device)
            return model, cpu_device, cpu_precision, "cpu"
        print("[sam3] dispatching model with CPU offload support")
        model = _accelerate_dispatch_model(
            model,
            device_map=resolved_device_map,
            offload_dir=placement.offload_folder,
        )
        self._last_hf_load_placement = placement
        self.emit_hf_auto_device_report(model, placement)
        return model, device, precision, None

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[Sam3Options]) -> Sam3Runtime:
        if build_sam3_image_model is None or Sam3Processor is None:
            raise ImportError(
                "sam3 dependencies are missing; run models/setup_venv/sam3.sh inside the fm cache."
            ) from _SAM3_IMPORT_ERROR

        stage_dir = getattr(ctx, "stage_dir", None)
        cache_root = ctx.weights_dir or stage_dir
        if cache_root:
            # Keep SAM3 downloads in the durable model weights cache rather than
            # the per-run stage directory, which is cleared after each run.
            utils.ensure_hf_caches(cache_root)
            os.environ.setdefault("SAM3_CACHE", str(cache_root))
            print(f"[sam3] cache root: {cache_root}")
            print("[sam3] loading model assets; first run may download facebook/sam3 and stay quiet for a while")

        device_pref = utils.normalize_device_pref(ctx.request.device)
        if device_pref == "cpu":
            raise RuntimeError("sam3 does not support CPU execution; use --device auto or --device cuda.")
        if torch.version.cuda is None:
            raise RuntimeError("sam3 requires a CUDA-enabled PyTorch build in its managed venv.")

        device = utils.select_device(device_pref)
        if device.type != "cuda":
            raise RuntimeError(
                f"sam3 requires CUDA, but device preference {device_pref!r} resolved to {device} in this environment."
            )
        device_index = device.index if device.index is not None else 0
        arch_list = _cuda_arch_list()
        capability = torch.cuda.get_device_capability(device_index)
        device_sm = f"sm_{capability[0]}{capability[1]}"
        if device_sm not in arch_list:
            supported_sms = ", ".join(sorted(arch_list))
            raise RuntimeError(
                "sam3 requires a PyTorch build that supports the active GPU architecture; "
                f"device reports {device_sm}, but torch {torch.__version__} supports {supported_sms}. "
                "Rebuild the managed sam3 venv with CUDA 12.8 wheels or newer."
            )
        confidence_threshold = float(ctx.options.confidence_threshold)
        _patch_sam3_fused_addmm_dtype_compat()
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        runtime_device = device
        managed_auto_fallback: str | None = None
        if device_pref == "auto":
            model, runtime_device, precision, managed_auto_fallback = self._load_auto_dispatched_model(
                dataset,
                ctx,
                device=device,
                precision=precision,
            )
        else:
            self._last_hf_load_placement = None
            model = self._build_sam3_model(device_arg=str(device))
        autocast_ctx = utils.maybe_autocast_for_module(runtime_device, precision, model)
        processor = Sam3Processor(
            model,
            device=str(runtime_device),
            confidence_threshold=confidence_threshold,
        )
        print(f"[sam3] model loaded (device={runtime_device})")
        print(f"[sam3] dataset records={len(dataset.records)} prompts={len(utils.dataset_ground_prompts(dataset))}")
        return Sam3Runtime(
            device=runtime_device,
            autocast_ctx=autocast_ctx,
            processor=processor,
            managed_auto_fallback=managed_auto_fallback,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: List[Sam3Job],
        runtime: Sam3Runtime,
        ctx: RuntimeContext[Sam3Options],
    ) -> BatchResult:
        warnings: List[str] = []
        modified: List[int] = []
        class_to_id = {name: idx for idx, name in enumerate(dataset.classes)}
        has_masks_any = bool(getattr(dataset.task, "value", dataset.task) == "seg")
        has_boxes_any = bool(getattr(dataset.task, "value", dataset.task) == "det")

        for job in batch:
            rec = dataset.records[job.record_idx]
            if not job.image_path.exists():
                warnings.append(f"[sam3] missing image src={rec.image.path} resolved={job.image_path}")
                continue
            try:
                with Image.open(job.image_path) as im:
                    image = im.convert("RGB")
            except Exception as exc:
                warnings.append(f"[sam3] failed to load image src={rec.image.path} resolved={job.image_path}: {exc}")
                continue

            try:
                next_gid = self.next_group_id(rec)
                with torch.inference_mode(), runtime.autocast_ctx:
                    state = runtime.processor.set_image(image)
                    for prompt in self.ground_prompts_for_record(rec):
                        outputs, max_score = self._run_prompt_with_threshold_diagnostics(
                            runtime.processor,
                            state=state,
                            prompt=prompt,
                        )

                        boxes_np = utils.to_numpy(outputs.get("boxes") if isinstance(outputs, dict) else None)
                        scores_np = utils.to_numpy(outputs.get("scores") if isinstance(outputs, dict) else None)
                        masks_np = utils.to_numpy(outputs.get("masks") if isinstance(outputs, dict) else None)
                        if boxes_np is None or scores_np is None or len(boxes_np) == 0:
                            if max_score is not None:
                                warnings.append(
                                    "[sam3] no detections above threshold "
                                    f"src={rec.image.path} resolved={job.image_path} prompt={prompt!r} "
                                    f"max_score={max_score:.3f} threshold={float(runtime.processor.confidence_threshold):.3f}"
                                )
                            continue

                        label_id = self.ensure_class_id(dataset, prompt, class_to_id)

                        W = float(rec.image.width) or float(image.width)
                        H = float(rec.image.height) or float(image.height)
                        for idx in range(min(len(boxes_np), len(scores_np))):
                            gid = next_gid
                            next_gid += 1
                            self.append_ground_box(
                                rec,
                                box_xyxy=boxes_np[idx],
                                width=W,
                                height=H,
                                cls_id=label_id,
                                label=prompt,
                                score=float(scores_np[idx]),
                                group_id=gid,
                                prompt=prompt,
                            )
                            has_boxes_any = True

                            if masks_np is not None and idx < len(masks_np):
                                mask = masks_np[idx]
                                mask = np.squeeze(mask) if mask.ndim > 2 else mask
                                polys = utils.mask_to_polygons(mask)
                                if self.append_ground_polygons(
                                    rec,
                                    polys=polys,
                                    width=W,
                                    height=H,
                                    cls_id=label_id,
                                    label=prompt,
                                    score=float(scores_np[idx]),
                                    group_id=gid,
                                    prompt=prompt,
                                ):
                                    has_masks_any = True

                utils.append_once(rec.attributes.setdefault("fm_tasks", []), "sam3")
                self.finalize_record_task(rec, set_det_when_boxes=True)
                modified.append(job.record_idx)
            except Exception as exc:
                warnings.append(f"[sam3] failed to run inference src={rec.image.path} resolved={job.image_path}: {exc}")
                continue

        self.finalize_dataset_task(dataset, has_boxes_any=has_boxes_any, has_masks_any=has_masks_any, set_det_when_boxes=True)
        return BatchResult(modified_record_indices=modified, warnings=warnings)

    @staticmethod
    def _run_prompt_with_threshold_diagnostics(processor: Any, *, state: dict[str, Any], prompt: str) -> tuple[Any, float | None]:
        threshold_attr = getattr(processor, "confidence_threshold", None)
        if not isinstance(threshold_attr, (int, float)):
            outputs = processor.set_text_prompt(state=state, prompt=prompt)
            scores = utils.to_numpy(outputs.get("scores") if isinstance(outputs, dict) else None)
            max_score = None if scores is None or len(scores) == 0 else float(np.max(scores))
            return outputs, max_score

        original_threshold = float(threshold_attr)
        processor.confidence_threshold = -1.0
        try:
            outputs = processor.set_text_prompt(state=state, prompt=prompt)
        finally:
            processor.confidence_threshold = original_threshold

        if not isinstance(outputs, dict):
            return outputs, None

        boxes_np = utils.to_numpy(outputs.get("boxes"))
        scores_np = utils.to_numpy(outputs.get("scores"))
        masks_np = utils.to_numpy(outputs.get("masks"))
        max_score = None if scores_np is None or len(scores_np) == 0 else float(np.max(scores_np))

        if boxes_np is None or scores_np is None:
            return outputs, max_score

        keep = scores_np > original_threshold
        filtered: dict[str, Any] = {
            **outputs,
            "boxes": boxes_np[keep],
            "scores": scores_np[keep],
        }
        if masks_np is not None:
            filtered["masks"] = masks_np[keep]
        return filtered, max_score

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: Sam3Runtime,
        ctx: RuntimeContext[Sam3Options],
    ) -> dict[str, object]:
        from cvsuite.common.fm.providers.bases.vlm import build_hf_placement_meta

        placement_meta = build_hf_placement_meta(ctx, self._last_hf_load_placement)
        return self.build_ground_fm_meta(
            dataset,
            ctx,
            task_name="sam3",
            extra={
                "device": str(runtime.device),
                "confidence_threshold": float(ctx.options.confidence_threshold),
                "weights": str(ctx.weights_dir) if ctx.weights_dir else None,
                "config": str(ctx.config_path) if ctx.config_path else None,
                "managed_auto_fallback": runtime.managed_auto_fallback,
                **placement_meta,
            },
        )


MODEL = Sam3Model()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
