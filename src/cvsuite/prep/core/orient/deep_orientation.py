from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cvsuite.common.core import Classification, FMRequest
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import FMRunner

MODEL_ID = "deep_orientation"
CLASS_LABELS = (
    "correct",
    "rotate_90_clockwise",
    "rotate_180",
    "rotate_90_counter_clockwise",
)
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_LABELS)}
CLASS_TO_CCW_DEGREES = {
    0: 0,
    1: 270,
    2: 180,
    3: 90,
}


@dataclass(frozen=True)
class OrientationPrediction:
    class_index: int
    ccw_degrees: int
    scores: tuple[float, ...] | None = None


def _scores_from_classification(classification: Classification | None) -> tuple[float, ...] | None:
    if classification is None or not classification.probs:
        return None
    return tuple(float(classification.probs.get(label, 0.0)) for label in CLASS_LABELS)


def _class_index_from_classification(classification: Classification | None) -> int:
    if classification is None:
        raise RuntimeError("Orientation model did not produce a classification.")

    meta = classification.meta or {}
    raw_idx = meta.get("class_index")
    if isinstance(raw_idx, int) and raw_idx in CLASS_TO_CCW_DEGREES:
        return raw_idx

    if classification.label in LABEL_TO_INDEX:
        return LABEL_TO_INDEX[classification.label]

    scores = _scores_from_classification(classification)
    if scores is not None and len(scores) == 4:
        return max(range(4), key=lambda idx: scores[idx])

    raise RuntimeError("Orientation model output is missing class_index and usable probabilities.")


def _prediction_from_classification(classification: Classification | None) -> OrientationPrediction:
    class_index = _class_index_from_classification(classification)
    meta = classification.meta if classification is not None else {}
    raw_ccw = meta.get("correction_degrees_ccw") if isinstance(meta, dict) else None
    if isinstance(raw_ccw, int) and raw_ccw in {0, 90, 180, 270}:
        ccw_degrees = raw_ccw
    else:
        ccw_degrees = CLASS_TO_CCW_DEGREES[class_index]
    return OrientationPrediction(
        class_index=class_index,
        ccw_degrees=ccw_degrees,
        scores=_scores_from_classification(classification),
    )


def predict_dataset(
    dataset: VisionDataset,
    *,
    device_hint: str,
    precision: str,
    batch_size: int,
    weights_dir: Path | None,
    no_resume: bool = False,
) -> list[OrientationPrediction]:
    if not dataset.records:
        return []

    # deep_orientation ships a CPU-only venv and does not implement managed
    # `--device auto` placement; resolve the hint to a concrete device here.
    device = device_hint.strip().lower()
    if device in {"", "auto"}:
        device = "cpu"

    dataset.fm_request = FMRequest(
        task="classify",
        provider=MODEL_ID,
        device=device,
        precision=precision,
        batch_size=max(1, batch_size),
        weights_dir=weights_dir,
        meta={"provider_family": "classify"},
    )
    scored = FMRunner.from_dataset(dataset).run(dataset, no_resume=no_resume)
    return [_prediction_from_classification(rec.classification) for rec in scored.records]
