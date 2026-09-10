from __future__ import annotations

from typing import Generic, Mapping, Sequence, TypeVar

import numpy as np

from cvsuite.common.core.enums import Task
from cvsuite.common.core import BBox, Polygon, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import RuntimeContext

from .record_image import BaseRecordImageModel

TOptions = TypeVar("TOptions")
TRuntime = TypeVar("TRuntime")


class BaseGroundModel(BaseRecordImageModel[TOptions, TRuntime], Generic[TOptions, TRuntime]):
    def ground_prompts_for_record(self, record: Record) -> list[str]:
        return utils.ground_prompts_for_record(record)

    def dataset_ground_prompts(self, dataset: VisionDataset) -> list[str]:
        return utils.dataset_ground_prompts(dataset)

    @staticmethod
    def next_group_id(record: Record) -> int:
        gids = [box.group_id for box in record.boxes if box.group_id is not None]
        gids.extend(poly.group_id for poly in record.polys if poly.group_id is not None)
        gids.extend(shape.group_id for shape in record.labelme_shapes if shape.group_id is not None)
        return int(max(gids)) + 1 if gids else 1

    @staticmethod
    def ensure_class_id(
        dataset: VisionDataset,
        prompt: str,
        class_to_id: dict[str, int],
    ) -> int:
        if prompt not in class_to_id:
            class_to_id[prompt] = len(class_to_id)
            dataset.classes.append(prompt)
        return class_to_id[prompt]

    def append_ground_box(
        self,
        record: Record,
        *,
        box_xyxy: np.ndarray,
        width: float,
        height: float,
        cls_id: int,
        label: str,
        score: float | None,
        group_id: int,
        prompt: str,
    ) -> None:
        cx, cy, bw, bh = utils.bbox_to_norm(box_xyxy, width, height)
        record.boxes.append(
            BBox(
                cx=cx,
                cy=cy,
                w=bw,
                h=bh,
                cls=cls_id,
                label=label,
                score=None if score is None else float(score),
                group_id=group_id,
                model=self.model_name,
                prompt=prompt,
            )
        )

    def append_ground_polygons(
        self,
        record: Record,
        *,
        polys: Sequence[Sequence[tuple[float, float]]],
        width: float,
        height: float,
        cls_id: int,
        label: str,
        score: float | None,
        group_id: int,
        prompt: str,
    ) -> int:
        count = 0
        for poly in utils.polys_to_norm(list(polys), width, height):
            record.polys.append(
                Polygon(
                    points=poly,
                    cls=cls_id,
                    label=label,
                    score=None if score is None else float(score),
                    group_id=group_id,
                    model=self.model_name,
                    prompt=prompt,
                )
            )
            count += 1
        return count

    @staticmethod
    def finalize_record_task(record: Record, *, set_det_when_boxes: bool) -> None:
        if record.task is not None:
            return
        if record.polys:
            record.task = Task.seg
        elif set_det_when_boxes and record.boxes:
            record.task = Task.det

    @staticmethod
    def finalize_dataset_task(
        dataset: VisionDataset,
        *,
        has_boxes_any: bool,
        has_masks_any: bool,
        set_det_when_boxes: bool,
    ) -> None:
        if has_masks_any:
            dataset.task = Task.seg
        elif set_det_when_boxes and has_boxes_any and dataset.task is None:
            dataset.task = Task.det

    def build_ground_fm_meta(
        self,
        dataset: VisionDataset,
        ctx: RuntimeContext[TOptions],
        *,
        task_name: str,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        meta: dict[str, object] = {
            "task": task_name,
            "provider": self.model_name,
            "device": str(ctx.request.device),
            "precision": ctx.request.precision,
            "prompts": self.dataset_ground_prompts(dataset),
        }
        if extra:
            meta.update(dict(extra))
        return meta
