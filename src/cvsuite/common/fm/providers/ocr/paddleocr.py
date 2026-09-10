from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import yaml

from cvsuite.common.core.enums import Task
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseOCRModel, RecordImageJob

DEFAULT_MODEL_ID = "paddleocr"
PARAMS = "unknown"
MODEL_ALIASES = ("paddleOCR",)
TEXT_LABEL = "text"


@dataclass(frozen=True)
class PaddleOCROptions:
    pass


PaddleOCRJob = RecordImageJob


@dataclass
class PaddleOCRRuntime:
    engine: Any
    use_gpu: bool


def _is_point_sequence(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    first = value[0]
    return isinstance(first, (list, tuple)) and len(first) >= 2


def _parse_text_payload(payload: object) -> Tuple[Optional[str], Optional[float]]:
    if isinstance(payload, (list, tuple)):
        text = str(payload[0]).strip() if len(payload) >= 1 and payload[0] is not None else ""
        score = payload[1] if len(payload) >= 2 else None
        return (text or None), _coerce_float(score)
    if isinstance(payload, dict):
        text = payload.get("text", payload.get("rec_text", payload.get("transcription", "")))
        score = payload.get("score", payload.get("rec_score", payload.get("confidence")))
        return (str(text).strip() or None), _coerce_float(score)
    if payload is None:
        return None, None
    return str(payload).strip() or None, None


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _iter_ocr_entries(payload: object) -> Iterator[Tuple[List[Tuple[float, float]], Optional[str], Optional[float]]]:
    if payload is None:
        return
    if isinstance(payload, dict):
        polys = payload.get("dt_polys") or payload.get("boxes")
        texts = payload.get("rec_texts") or payload.get("texts")
        scores = payload.get("rec_scores") or payload.get("scores")
        if isinstance(polys, Sequence) and isinstance(texts, Sequence):
            scores_seq = list(scores) if isinstance(scores, Sequence) else [None] * len(texts)
            for poly, text, score in zip(polys, texts, scores_seq):
                points = _normalize_poly(poly)
                if points:
                    text_value = str(text).strip() or None
                    yield points, text_value, _coerce_float(score)
            return
        for value in payload.values():
            yield from _iter_ocr_entries(value)
        return
    if isinstance(payload, (list, tuple)):
        if len(payload) == 2 and _is_point_sequence(payload[0]):
            points = _normalize_poly(payload[0])
            if points:
                text, score = _parse_text_payload(payload[1])
                yield points, text, score
            return
        for item in payload:
            yield from _iter_ocr_entries(item)


def _normalize_poly(poly: object) -> List[Tuple[float, float]]:
    if not isinstance(poly, (list, tuple)):
        return []
    points: List[Tuple[float, float]] = []
    for point in poly:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return []
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            return []
        points.append((x, y))
    return points


def _load_engine_kwargs(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("PaddleOCR config YAML must contain a top-level mapping of PaddleOCR kwargs.")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("PaddleOCR config keys must be non-empty strings.")
        kwargs[key] = value
    return kwargs


class PaddleOCRModel(BaseOCRModel[PaddleOCROptions, PaddleOCRRuntime]):
    description = "PaddleOCR wrapper: OCR boxes and text over a VisionDataset JSON."
    options_cls = PaddleOCROptions

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[PaddleOCROptions]) -> PaddleOCRRuntime:
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
        try:
            import paddle
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover - dependency resolution happens in model venv
            raise ImportError(
                "PaddleOCR dependencies are missing; run models/setup_venv/paddleocr.sh inside the fm cache."
            ) from exc

        compiled_with_cuda = bool(getattr(paddle.device, "is_compiled_with_cuda", lambda: False)())
        pref = str(ctx.request.device).lower()
        use_gpu = pref == "gpu" or (pref == "auto" and compiled_with_cuda)
        if use_gpu and not compiled_with_cuda:
            use_gpu = False

        engine_kwargs = _load_engine_kwargs(ctx.config_path)
        engine_kwargs.setdefault("use_angle_cls", True)
        engine_kwargs.setdefault("show_log", False)
        engine_kwargs["use_gpu"] = use_gpu

        engine = PaddleOCR(**engine_kwargs)
        return PaddleOCRRuntime(engine=engine, use_gpu=use_gpu)

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: List[PaddleOCRJob],
        runtime: PaddleOCRRuntime,
        ctx: RuntimeContext[PaddleOCROptions],
    ) -> BatchResult:
        warnings: List[str] = []
        modified: List[int] = []

        cls_id = self.ensure_text_class(dataset)

        for job in batch:
            rec = dataset.records[job.record_idx]
            if not job.image_path.exists():
                warnings.append(f"[paddleocr] missing image src={rec.image.path} resolved={job.image_path}")
                continue

            try:
                with Image.open(job.image_path) as image:
                    width = float(rec.image.width or image.width)
                    height = float(rec.image.height or image.height)
                raw_result = runtime.engine.ocr(str(job.image_path), cls=True)
            except Exception as exc:
                warnings.append(f"[paddleocr] failed to run OCR src={rec.image.path} resolved={job.image_path}: {exc}")
                continue

            next_gid = self.next_group_id(rec)
            changed = False
            for poly, text, score in _iter_ocr_entries(raw_result):
                if not poly:
                    continue
                self.append_ocr_box(
                    rec,
                    poly=poly,
                    width=width,
                    height=height,
                    cls_id=cls_id,
                    text=text,
                    score=score,
                    group_id=next_gid,
                    attributes={"polygon": [(float(x), float(y)) for x, y in poly]},
                )
                next_gid += 1
                changed = True

            if changed:
                utils.append_once(rec.attributes.setdefault("fm_tasks", []), "ocr")
                self.finalize_record_task(rec)
                modified.append(job.record_idx)

        self.finalize_dataset_task(dataset)
        return BatchResult(modified_record_indices=modified, warnings=warnings)

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: PaddleOCRRuntime,
        ctx: RuntimeContext[PaddleOCROptions],
    ) -> dict[str, object]:
        return self.build_ocr_fm_meta(
            ctx,
            extra={
                "use_gpu": runtime.use_gpu,
                "weights": str(ctx.weights_dir) if ctx.weights_dir else None,
                "config": str(ctx.config_path) if ctx.config_path else None,
            },
        )


MODEL = PaddleOCRModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
