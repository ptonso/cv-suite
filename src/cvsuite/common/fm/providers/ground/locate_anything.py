from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseGroundModel, RecordImageJob
from cvsuite.common.fm.providers.bases.vlm import build_hf_placement_meta, resolve_device_map

DEFAULT_MODEL_ID = "nvidia/LocateAnything-3B"
ALLOWED_MODEL_IDS = (DEFAULT_MODEL_ID,)
PARAMS = "3B"
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_REMOTE_PATCH_FILENAMES = ("image_processing_locateanything.py", "modeling_qwen2.py", "generate_utils.py")


@dataclass(frozen=True)
class LocateAnythingOptions:
    model_id: str | None = None
    max_new_tokens: int | None = None
    generation_mode: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    trust_remote_code: bool | None = None
    revision: str | None = None
    force_cpu: bool | None = None


@dataclass(frozen=True)
class LocateAnythingConfig:
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 8192
    generation_mode: str = "hybrid"
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    trust_remote_code: bool = True
    revision: str | None = None
    force_cpu: bool = False


LocateAnythingJob = RecordImageJob


@dataclass
class LocateAnythingRuntime:
    cfg: LocateAnythingConfig
    model_id: str
    device: torch.device
    model_device: torch.device
    dtype: torch.dtype
    precision: utils.ResolvedPrecision
    autocast_ctx: object
    tokenizer: Any
    processor: Any
    model: Any


def _load_config(path: Path | None, options: LocateAnythingOptions) -> LocateAnythingConfig:
    cfg = LocateAnythingConfig()
    values = {field.name: getattr(cfg, field.name) for field in fields(LocateAnythingConfig)}
    if path and path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError("LocateAnything config YAML must contain a mapping/object at the top level.")
        for field in fields(LocateAnythingConfig):
            if field.name in data:
                values[field.name] = data[field.name]
    for field in fields(LocateAnythingOptions):
        value = getattr(options, field.name)
        if value is not None:
            values[field.name] = value
    return LocateAnythingConfig(**values)


def _resolve_model_id(cfg: LocateAnythingConfig) -> str:
    model_id = str(cfg.model_id or DEFAULT_MODEL_ID).strip()
    if model_id not in ALLOWED_MODEL_IDS:
        supported = ", ".join(ALLOWED_MODEL_IDS)
        raise ValueError(f"Unsupported LocateAnything model_id {model_id!r}. Supported model ids: {supported}.")
    return model_id


def _prompt_for_grounding(prompt: str) -> str:
    return f"Locate all the instances that match the following description: {prompt}."


def _coerce_answer(response: object) -> str:
    if isinstance(response, tuple):
        return _coerce_answer(response[0] if response else "")
    if isinstance(response, list):
        return _coerce_answer(response[0] if response else "")
    return str(response or "")


def _parse_boxes(answer: str, width: float, height: float) -> tuple[list[np.ndarray], list[str]]:
    """Parse normalized xyxy box tokens, dropping malformed boxes from the stochastic decoder.

    LocateAnything samples coordinates (do_sample=True), so it occasionally emits out-of-range
    or inverted boxes. These are expected noise, not invalid program state: skip them and report
    via warnings rather than aborting the whole run.
    """
    boxes: list[np.ndarray] = []
    warnings: list[str] = []
    for match in _BOX_RE.finditer(answer):
        x0, y0, x1, y1 = [int(value) for value in match.groups()]
        if min(x0, y0, x1, y1) < 0 or max(x0, y0, x1, y1) > 1000:
            warnings.append(f"dropped box outside normalized 0-1000 range: {(x0, y0, x1, y1)}")
            continue
        if x1 < x0 or y1 < y0:
            warnings.append(f"dropped inverted box: {(x0, y0, x1, y1)}")
            continue
        boxes.append(
            np.asarray(
                [
                    x0 / 1000.0 * width,
                    y0 / 1000.0 * height,
                    x1 / 1000.0 * width,
                    y1 / 1000.0 * height,
                ],
                dtype=np.float64,
            )
        )
    return boxes, warnings


def _first_module_device(module: Any, fallback: torch.device) -> torch.device:
    for getter_name in ("parameters", "buffers"):
        getter = getattr(module, getter_name, None)
        if not callable(getter):
            continue
        try:
            values = getter()
        except TypeError:
            values = getter(recurse=True)
        for value in values:
            device = getattr(value, "device", None)
            if device is not None and device.type != "meta":
                return device
    return fallback


def _move_inputs(inputs: Any, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif key == "pixel_values":
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _write_if_changed(path: Path, text: str) -> None:
    original = path.read_text(encoding="utf-8")
    if text != original:
        path.write_text(text, encoding="utf-8")


def _patch_locate_anything_remote_file(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if path.name == "image_processing_locateanything.py":
        text = text.replace(
            'AutoImageProcessor.register("LocateAnythingImageProcessor", LocateAnythingImageProcessor)',
            'AutoImageProcessor.register("LocateAnythingImageProcessor", slow_image_processor_class=LocateAnythingImageProcessor)',
        )
    elif path.name == "modeling_qwen2.py":
        if "from transformers.generation import GenerationMixin" not in text:
            text = text.replace(
                "from transformers.modeling_utils import PreTrainedModel\n",
                "from transformers.modeling_utils import PreTrainedModel\nfrom transformers.generation import GenerationMixin\n",
            )
        text = text.replace(
            "class Qwen2ForCausalLM(Qwen2PreTrainedModel):",
            "class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):",
        )
    elif path.name == "generate_utils.py":
        text = text.replace(
            "torch.tensor(out_ref, dtype=x0.dtype, device=x0.device)",
            "torch.as_tensor(out_ref, dtype=x0.dtype, device=x0.device)",
        )
    _write_if_changed(path, text)


def _patch_locate_anything_remote_code(source: utils.HFModelSource, stage_dir: Path | None) -> None:
    roots: list[Path] = []
    if source.snapshot_path is not None:
        roots.append(source.snapshot_path)
    if stage_dir is not None:
        modules_dir = stage_dir / "hf_modules"
        if modules_dir.exists():
            roots.append(modules_dir)

    for root in roots:
        for filename in _REMOTE_PATCH_FILENAMES:
            direct = root / filename
            if direct.exists():
                _patch_locate_anything_remote_file(direct)
            for path in root.rglob(filename):
                _patch_locate_anything_remote_file(path)


def _force_sdpa_attention(config: Any) -> Any:
    for target in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
        if target is not None:
            setattr(target, "_attn_implementation", "sdpa")
    return config


class LocateAnythingModel(BaseGroundModel[LocateAnythingOptions, LocateAnythingRuntime]):
    description = "NVIDIA LocateAnything-3B VLM grounding over VisionDataset JSON."
    options_cls = LocateAnythingOptions
    supports_nf4_precision = True
    supports_managed_auto_device = True

    def load_runtime(
        self,
        dataset: VisionDataset,
        ctx: RuntimeContext[LocateAnythingOptions],
    ) -> LocateAnythingRuntime:
        cfg = _load_config(ctx.config_path, ctx.options)
        model_id = _resolve_model_id(cfg)
        source = utils.materialize_hf_model_source(
            model_id,
            hub_dir=ctx.hub_dir,
            stage_dir=getattr(ctx, "stage_dir", None),
            caller_cwd=getattr(ctx, "caller_cwd", Path.cwd()),
            revision=cfg.revision,
        )
        _patch_locate_anything_remote_code(source, getattr(ctx, "stage_dir", None))
        device = utils.select_device(ctx.request.device, cfg.force_cpu)
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        default_device_map = {"": "cpu"} if cfg.force_cpu else resolve_device_map(ctx.request.device)
        load_kwargs, _placement = self.build_hf_load_kwargs(
            ctx,
            device,
            precision,
            dtype_key="dtype",
            default_device_map=default_device_map,
            default_offload_folder=True,
        )
        source_kwargs = utils.build_hf_source_kwargs(
            source,
            trust_remote_code=bool(cfg.trust_remote_code),
        )
        config = _force_sdpa_attention(AutoConfig.from_pretrained(source.load_arg, **source_kwargs))
        tokenizer = AutoTokenizer.from_pretrained(source.load_arg, **source_kwargs)
        processor = AutoProcessor.from_pretrained(source.load_arg, use_fast=False, **source_kwargs)
        model = self.load_hf_pretrained_model(
            AutoModel.from_pretrained,
            precision,
            source.load_arg,
            **utils.build_hf_source_kwargs(
                source,
                config=config,
                trust_remote_code=bool(cfg.trust_remote_code),
                **load_kwargs,
            ),
        )
        model.eval()
        model_device = _first_module_device(model, device)
        return LocateAnythingRuntime(
            cfg=cfg,
            model_id=model_id,
            device=device,
            model_device=model_device,
            dtype=precision.compute_dtype,
            precision=precision,
            autocast_ctx=utils.maybe_autocast(model_device, precision),
            tokenizer=tokenizer,
            processor=processor,
            model=model,
        )

    def _generate_answer(
        self,
        image: Image.Image,
        prompt: str,
        runtime: LocateAnythingRuntime,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _prompt_for_grounding(prompt)},
                ],
            }
        ]
        text = runtime.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = runtime.processor.process_vision_info(messages)
        inputs = runtime.processor(text=[text], images=images, videos=videos, return_tensors="pt")
        inputs = _move_inputs(inputs, runtime.model_device, runtime.dtype)
        with torch.inference_mode(), runtime.autocast_ctx:
            response = runtime.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=runtime.tokenizer,
                max_new_tokens=int(runtime.cfg.max_new_tokens),
                use_cache=True,
                generation_mode=runtime.cfg.generation_mode,
                temperature=float(runtime.cfg.temperature),
                do_sample=True,
                top_p=float(runtime.cfg.top_p),
                repetition_penalty=float(runtime.cfg.repetition_penalty),
                verbose=False,
            )
        return _coerce_answer(response)

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: list[LocateAnythingJob],
        runtime: LocateAnythingRuntime,
        ctx: RuntimeContext[LocateAnythingOptions],
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
            width = float(rec.image.width) or float(image.width)
            height = float(rec.image.height) or float(image.height)
            for prompt in self.ground_prompts_for_record(rec):
                cls_id = self.ensure_class_id(dataset, prompt, class_to_id)
                answer = self._generate_answer(image, prompt, runtime)
                parsed_boxes, parse_warnings = _parse_boxes(answer, width, height)
                for note in parse_warnings:
                    warnings.append(
                        f"[{self.model_name}] {note} idx={job.record_idx} prompt={prompt!r}"
                    )
                for box in parsed_boxes:
                    gid = next_gid
                    next_gid += 1
                    self.append_ground_box(
                        rec,
                        box_xyxy=box,
                        width=width,
                        height=height,
                        cls_id=cls_id,
                        label=prompt,
                        score=None,
                        group_id=gid,
                        prompt=prompt,
                    )

            tasks = rec.attributes.setdefault("fm_tasks", [])
            if "locate-anything" not in tasks:
                tasks.append("locate-anything")
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
        runtime: LocateAnythingRuntime,
        ctx: RuntimeContext[LocateAnythingOptions],
    ) -> dict[str, object]:
        return self.build_ground_fm_meta(
            dataset,
            ctx,
            task_name="locate-anything",
            extra={
                "device": str(runtime.model_device),
                "hf_model": runtime.model_id,
                "generation_mode": runtime.cfg.generation_mode,
                "max_new_tokens": runtime.cfg.max_new_tokens,
                "config": str(ctx.config_path) if ctx.config_path else None,
                **utils.build_precision_meta(runtime.precision),
                **build_hf_placement_meta(ctx, self._last_hf_load_placement),
            },
        )


MODEL = LocateAnythingModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
