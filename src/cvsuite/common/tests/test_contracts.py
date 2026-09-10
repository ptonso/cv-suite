from __future__ import annotations

from pathlib import Path

from cvsuite.common.core import Embedding, FMRequest, ImageRecord, Record
from cvsuite.common.core import VisionDataset


def test_records_dataset_round_trip_preserves_fm_request_and_embeddings(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=tmp_path / "img.jpg", width=10, height=20),
                embeddings=[Embedding(vector=[0.1, 0.2], model="clip", task="classify", meta={"kind": "global"})],
            )
        ],
        fm_request=FMRequest(
            task="classify",
            provider="clip",
            prompt="cat,dog",
            batch_size=4,
            model_args={"threshold": 0.6},
        ),
    )

    payload = tmp_path / "dataset.json"
    dataset.to_json(payload)
    restored = VisionDataset.from_json(payload)

    assert restored.fm_request is not None
    assert restored.fm_request.provider == "clip"
    assert restored.fm_request.model_args["threshold"] == 0.6
    assert restored.records[0].embeddings[0].vector == [0.1, 0.2]

    image = restored.records[0].image
    assert image.path == tmp_path / "img.jpg"
    assert (image.width, image.height) == (10, 20)
