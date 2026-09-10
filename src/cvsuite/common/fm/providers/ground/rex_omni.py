from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseGroundModel, RecordImageJob

DEFAULT_MODEL_ID = "IDEA-Research/Rex-Omni"
PARAMS = "3B"

_CATEGORY_RE = re.compile(
    r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>"
)
_COORD_RE = re.compile(r"<(\d+)>")


warnings.filterwarnings(
    "ignore",
    message=r"`do_sample` is set to `False`",
    category=UserWarning,
)

@dataclass(frozen=True)
class RexOmniOptions:
    pass


@dataclass
class RexOmniConfig:
    model_id: str = DEFAULT_MODEL_ID
    attn_implementation: str = "sdpa"
    max_tokens: int = 4096
    min_pixels: int = 16 * 28 * 28
    max_pixels: int = 2560 * 28 * 28
    temperature: float = 0.0
    top_p: float = 0.05
    top_k: int = 1
    repetition_penalty: float = 1.05
    force_cpu: bool = False


RexOmniJob = RecordImageJob


class _ScoreCapture:
    """Forces scored generation and keeps the last outputs; the upstream wrapper still receives plain id sequences."""

    def __init__(self, generate) -> None:
        self._generate = generate
        self.sequences = None
        self.scores = None
        self.input_len = 0

    def __call__(self, **kwargs):
        out = self._generate(**kwargs, output_scores=True, return_dict_in_generate=True)
        self.sequences = out.sequences
        self.scores = out.scores
        self.input_len = int(kwargs["input_ids"].shape[1])
        return out.sequences


@dataclass
class RexOmniRuntime:
    cfg: RexOmniConfig
    device: torch.device
    model: object
    score_capture: _ScoreCapture


def _load_config(path: Optional[Path]) -> RexOmniConfig:
    cfg = RexOmniConfig()
    if path and path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict):
            for f in fields(RexOmniConfig):
                if f.name in data:
                    setattr(cfg, f.name, data[f.name])
    return cfg


def _decode_token_logprobs(capture: _ScoreCapture, tokenizer) -> Tuple[List[str], List[float]]:
    ids = capture.sequences[0][capture.input_len :]
    pieces = [tokenizer.decode([int(tid)]) for tid in ids]
    logprobs = [
        float(torch.log_softmax(step[0].float(), dim=-1)[int(tid)])
        for step, tid in zip(capture.scores, ids)
    ]
    return pieces, logprobs


def _box_confidences(
    raw_output: str, pieces: List[str], logprobs: List[float]
) -> Optional[Dict[str, List[float]]]:
    """Per-category box confidences as exp(mean logprob) over each box's coordinate digit tokens.

    Mirrors rex_omni.parser.parse_standard_prediction (same truncation, category regex,
    comma split, 4-number = box rule) so each list zips with that category's box
    annotations in order. Returns None when token pieces cannot be aligned to the text.
    """
    text = "".join(pieces)
    if not text.startswith(raw_output) and not raw_output.startswith(text):
        return None

    offsets: List[Tuple[int, int]] = []
    pos = 0
    for piece in pieces:
        offsets.append((pos, pos + len(piece)))
        pos += len(piece)

    visible = raw_output.split("<|im_end|>")[0]
    search_text = visible if visible.endswith("<|box_end|>") else visible + "<|box_end|>"

    confidences: Dict[str, List[float]] = {}
    for match in _CATEGORY_RE.finditer(search_text):
        per_box = confidences.setdefault(match.group(1).strip(), [])
        base = match.start(2)
        segment_start = 0
        for segment in match.group(2).split(","):
            nums = list(_COORD_RE.finditer(segment))
            if len(nums) == 4:
                spans = [
                    (base + segment_start + num.start(1), base + segment_start + num.end(1))
                    for num in nums
                ]
                span_lps = [
                    lp
                    for (tok_s, tok_e), lp in zip(offsets, logprobs)
                    if any(tok_s < end and tok_e > start for start, end in spans)
                ]
                per_box.append(math.exp(sum(span_lps) / len(span_lps)) if span_lps else 1.0)
            segment_start += len(segment) + 1
    return confidences


class RexOmniModel(BaseGroundModel[RexOmniOptions, RexOmniRuntime]):
    description = "Rex-Omni 3B MLLM detector: grounded bounding boxes via next-point prediction over VisionDataset JSON."
    options_cls = RexOmniOptions

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[RexOmniOptions]) -> RexOmniRuntime:
        cfg = _load_config(ctx.config_path)
        device = utils.select_device(ctx.request.device, cfg.force_cpu)
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        source = utils.materialize_hf_model_source(
            cfg.model_id,
            hub_dir=ctx.hub_dir,
            stage_dir=getattr(ctx, "stage_dir", None),
            caller_cwd=getattr(ctx, "caller_cwd", Path.cwd()),
        )

        from rex_omni import RexOmniWrapper

        model = RexOmniWrapper(
            model_path=str(source.snapshot_path or source.load_arg),
            backend="transformers",
            min_pixels=int(cfg.min_pixels),
            max_pixels=int(cfg.max_pixels),
            max_tokens=int(cfg.max_tokens),
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
            top_k=int(cfg.top_k),
            repetition_penalty=float(cfg.repetition_penalty),
            torch_dtype=precision.compute_dtype,
            attn_implementation=cfg.attn_implementation,
            device_map=str(device),
        )
        capture = _ScoreCapture(model.model.generate)
        model.model.generate = capture
        return RexOmniRuntime(cfg=cfg, device=device, model=model, score_capture=capture)

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: List[RexOmniJob],
        runtime: RexOmniRuntime,
        ctx: RuntimeContext[RexOmniOptions],
    ) -> BatchResult:
        warnings: List[str] = []
        modified: List[int] = []
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

            prompts = self.ground_prompts_for_record(rec)
            cls_ids = {prompt: self.ensure_class_id(dataset, prompt, class_to_id) for prompt in prompts}

            if prompts:
                result = runtime.model.inference(images=image, task="detection", categories=list(prompts))[0]
                if not result.get("success", False):
                    detail = result.get("error") or result.get("raw_output")
                    warnings.append(f"[rex_omni] inference failed src={rec.image.path}: {detail!r}")
                    continue

                conf_by_category: Optional[Dict[str, List[float]]] = None
                try:
                    pieces, logprobs = _decode_token_logprobs(
                        runtime.score_capture, runtime.model.processor.tokenizer
                    )
                    conf_by_category = _box_confidences(str(result.get("raw_output") or ""), pieces, logprobs)
                except Exception as exc:
                    warnings.append(f"[rex_omni] confidence extraction failed src={rec.image.path}: {exc}")

                prompt_by_key = {prompt.strip().casefold(): prompt for prompt in prompts}
                width = float(rec.image.width) or float(image.width)
                height = float(rec.image.height) or float(image.height)
                next_gid = self.next_group_id(rec)
                for category, annotations in (result.get("extracted_predictions") or {}).items():
                    prompt = prompt_by_key.get(str(category).strip().casefold())
                    if prompt is None:
                        warnings.append(f"[rex_omni] unprompted category {category!r} src={rec.image.path}")
                        continue
                    boxes = [ann for ann in annotations if ann.get("type") == "box"]
                    box_confs = (conf_by_category or {}).get(category)
                    if box_confs is None or len(box_confs) != len(boxes):
                        if boxes and conf_by_category is not None:
                            warnings.append(
                                f"[rex_omni] confidence alignment failed for {category!r} src={rec.image.path}; using score 1.0"
                            )
                        box_confs = [1.0] * len(boxes)
                    for ann, conf in zip(boxes, box_confs):
                        gid = next_gid
                        next_gid += 1
                        self.append_ground_box(
                            rec,
                            box_xyxy=np.asarray(ann["coords"], dtype=np.float64),
                            width=width,
                            height=height,
                            cls_id=cls_ids[prompt],
                            label=prompt,
                            score=float(conf),
                            group_id=gid,
                            prompt=prompt,
                        )

            tasks = rec.attributes.setdefault("fm_tasks", [])
            if "rex-omni" not in tasks:
                tasks.append("rex-omni")
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
        runtime: RexOmniRuntime,
        ctx: RuntimeContext[RexOmniOptions],
    ) -> dict[str, object]:
        return self.build_ground_fm_meta(
            dataset,
            ctx,
            task_name="rex-omni",
            extra={
                "device": str(runtime.device),
                "hf_model": runtime.cfg.model_id,
                "config": str(ctx.config_path) if ctx.config_path else None,
            },
        )


MODEL = RexOmniModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
