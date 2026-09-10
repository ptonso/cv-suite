from __future__ import annotations

from typing import Generic, Mapping, Sequence, TypeVar

from cvsuite.common.core import Classification
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import RuntimeContext

from .record_image import BaseRecordImageModel, RecordImageJob

TOptions = TypeVar("TOptions")
TRuntime = TypeVar("TRuntime")


class BaseClassificationModel(BaseRecordImageModel[TOptions, TRuntime], Generic[TOptions, TRuntime]):
    task_name = "classify"

    def classification_threshold(self, ctx: RuntimeContext[TOptions]) -> float | None:
        return getattr(ctx.options, "threshold", None)

    def build_classification_record_meta(
        self,
        ctx: RuntimeContext[TOptions],
        *,
        candidate_labels: Sequence[str] | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        meta: dict[str, object] = {
            "provider": self.model_name,
            "device": str(ctx.request.device),
            "precision": ctx.request.precision,
            "batch_size": ctx.batch_size,
        }
        if candidate_labels is not None:
            meta["candidate_labels"] = list(candidate_labels)
        if extra:
            meta.update(dict(extra))
        return meta

    def store_classification_result(
        self,
        dataset: VisionDataset,
        job: RecordImageJob,
        *,
        ctx: RuntimeContext[TOptions],
        label: str | None,
        score: float | None,
        probs: Mapping[str, float],
        meta: Mapping[str, object],
        candidate_labels_default: Sequence[str] | None = None,
    ) -> int:
        rec = dataset.records[job.record_idx]
        classification = rec.classification or Classification()
        classification.score = None if score is None else float(score)
        classification.probs = {str(key): float(value) for key, value in probs.items()}
        classification.meta.update(dict(meta))
        if candidate_labels_default is not None:
            classification.meta.setdefault("candidate_labels", list(candidate_labels_default))
        threshold = self.classification_threshold(ctx)
        if threshold is not None:
            classification.meta["threshold"] = threshold
        classification.label = None
        if label is not None:
            if threshold is None or score is None or float(score) >= float(threshold):
                classification.label = label
        rec.classification = classification
        utils.append_once(rec.attributes.setdefault("fm_tasks", []), self.task_name)
        return job.record_idx

    def build_classification_fm_meta(
        self,
        ctx: RuntimeContext[TOptions],
        *,
        candidate_labels: Sequence[str] | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        meta: dict[str, object] = {
            "task": self.task_name,
            "provider": self.model_name,
            "device": ctx.request.device,
            "precision": ctx.request.precision,
            "batch_size": ctx.batch_size,
        }
        if candidate_labels is not None:
            meta["candidate_labels"] = list(candidate_labels)
        threshold = self.classification_threshold(ctx)
        if threshold is not None:
            meta["threshold"] = threshold
        if extra:
            meta.update(dict(extra))
        return meta
