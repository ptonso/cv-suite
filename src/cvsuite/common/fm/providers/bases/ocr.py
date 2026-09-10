from __future__ import annotations

from typing import Generic, Mapping, TypeVar

import numpy as np

from cvsuite.common.core.enums import Task
from cvsuite.common.core import BBox, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import RuntimeContext

from .record_image import BaseRecordImageModel

TOptions = TypeVar("TOptions")
TRuntime = TypeVar("TRuntime")


class BaseOCRModel(BaseRecordImageModel[TOptions, TRuntime], Generic[TOptions, TRuntime]):
    text_label = "text"

    def ensure_text_class(self, dataset: VisionDataset) -> int:
        if self.text_label not in dataset.classes:
            dataset.classes.append(self.text_label)
        return dataset.classes.index(self.text_label)

    @staticmethod
    def next_group_id(record: Record) -> int:
        gids = [box.group_id for box in record.boxes if box.group_id is not None]
        gids.extend(poly.group_id for poly in record.polys if poly.group_id is not None)
        gids.extend(shape.group_id for shape in record.labelme_shapes if shape.group_id is not None)
        return int(max(gids)) + 1 if gids else 1

    def append_ocr_box(
        self,
        record: Record,
        *,
        poly: list[tuple[float, float]],
        width: float,
        height: float,
        cls_id: int,
        text: str | None,
        score: float | None,
        group_id: int,
        attributes: Mapping[str, object] | None = None,
    ) -> None:
        xs = [point[0] for point in poly]
        ys = [point[1] for point in poly]
        cx, cy, bw, bh = utils.bbox_to_norm(
            np.array([min(xs), min(ys), max(xs), max(ys)], dtype=float),
            width,
            height,
        )
        record.boxes.append(
            BBox(
                cx=cx,
                cy=cy,
                w=bw,
                h=bh,
                cls=cls_id,
                label=self.text_label,
                score=score,
                group_id=group_id,
                kind="ocr",
                text=text,
                model=self.model_name,
                prompt=None,
                attributes=dict(attributes or {}),
            )
        )

    @staticmethod
    def finalize_record_task(record: Record) -> None:
        if record.polys:
            record.task = Task.seg
        elif record.task is None:
            record.task = Task.det

    @staticmethod
    def finalize_dataset_task(dataset: VisionDataset) -> None:
        if any(record.polys for record in dataset.records):
            dataset.task = Task.seg
        elif any(record.boxes for record in dataset.records):
            dataset.task = Task.det

    def build_ocr_fm_meta(
        self,
        ctx: RuntimeContext[TOptions],
        *,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        meta: dict[str, object] = {
            "task": "ocr",
            "provider": self.model_name,
            "device": ctx.request.device,
            "precision": ctx.request.precision,
            "batch_size": ctx.batch_size,
        }
        if extra:
            meta.update(dict(extra))
        return meta
