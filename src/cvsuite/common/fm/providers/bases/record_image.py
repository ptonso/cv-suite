from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Sequence, TypeVar

from PIL import Image

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.base import BaseFMModel, RuntimeContext

TOptions = TypeVar("TOptions")
TRuntime = TypeVar("TRuntime")
TLoaded = TypeVar("TLoaded")


@dataclass(frozen=True)
class RecordImageJob:
    record_idx: int
    image_path: Path


class BaseRecordImageModel(BaseFMModel[TOptions, RecordImageJob, TRuntime], Generic[TOptions, TRuntime]):
    allow_root_basename: bool = False

    @staticmethod
    def load_rgb_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def resolve_record_image_path(
        self,
        dataset: VisionDataset,
        record_idx: int,
        ctx: RuntimeContext[TOptions],
    ) -> Path:
        return utils.resolve_path(
            dataset.records[record_idx].image.path,
            dataset.root,
            ctx.caller_cwd,
            allow_root_basename=self.allow_root_basename,
        )

    def build_jobs(
        self,
        dataset: VisionDataset,
        ctx: RuntimeContext[TOptions],
    ) -> list[RecordImageJob]:
        return [
            RecordImageJob(
                record_idx=idx,
                image_path=self.resolve_record_image_path(dataset, idx, ctx),
            )
            for idx, _rec in enumerate(dataset.records)
        ]

    def job_id(
        self,
        job: RecordImageJob,
        dataset: VisionDataset,
        ctx: RuntimeContext[TOptions],
    ) -> str:
        return f"record:{job.record_idx}"

    def missing_image_warning(
        self,
        dataset: VisionDataset,
        job: RecordImageJob,
    ) -> str:
        return (
            f"[{self.model_name}] missing image idx={job.record_idx} "
            f"src={dataset.records[job.record_idx].image.path} resolved={job.image_path}"
        )

    def image_load_warning(
        self,
        dataset: VisionDataset,
        job: RecordImageJob,
        exc: Exception,
    ) -> str:
        return f"[{self.model_name}] failed to load image idx={job.record_idx} path={job.image_path}: {exc}"

    def collect_loaded_records(
        self,
        dataset: VisionDataset,
        batch: Sequence[RecordImageJob],
        loader: Callable[[Path], TLoaded],
    ) -> tuple[list[tuple[RecordImageJob, TLoaded]], list[str]]:
        loaded: list[tuple[RecordImageJob, TLoaded]] = []
        warnings: list[str] = []
        for job in batch:
            if not job.image_path.exists():
                warnings.append(self.missing_image_warning(dataset, job))
                continue
            try:
                loaded.append((job, loader(job.image_path)))
            except Exception as exc:
                warnings.append(self.image_load_warning(dataset, job, exc))
        return loaded, warnings
