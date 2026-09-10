from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol, Sequence

from PIL import Image

from cvsuite.common.core import FMRequest, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import FMRunner


class UprightScorer(Protocol):
    def score(self, images: Sequence[Image.Image]) -> list[float]:
        ...


def load_clip_upright_scorer(
    *,
    prompt: str,
    device_hint: str,
    precision: str,
    weights_dir: Path | None,
    no_resume: bool = False,
) -> UprightScorer:
    class _FMUprightScorer:
        def score(self, images: Sequence[Image.Image]) -> list[float]:
            if not images:
                return []

            with tempfile.TemporaryDirectory(prefix="prep-orient-") as tmpdir:
                tmp_root = Path(tmpdir)
                records = []
                for index, image in enumerate(images):
                    img_path = tmp_root / f"candidate_{index}.png"
                    image.save(img_path, format="PNG")
                    records.append(
                        Record(
                            image=ImageRecord(path=img_path, width=image.width, height=image.height),
                            split="train",
                        )
                    )

                dataset = VisionDataset(records=records, root=tmp_root)
                dataset.fm_request = FMRequest(
                    task="classify",
                    provider="clip",
                    prompt=prompt,
                    device=device_hint,
                    precision=precision,
                    batch_size=max(1, len(images)),
                    weights_dir=weights_dir,
                    meta={"provider_family": "classify"},
                )
                scored = FMRunner.from_dataset(dataset).run(dataset, no_resume=no_resume)
                return [
                    float(rec.classification.score) if rec.classification and rec.classification.score is not None else 0.0
                    for rec in scored.records
                ]

    return _FMUprightScorer()
