from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from cvsuite.common.core.enums import Task
from cvsuite.common.core import FMRequest, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.providers.ocr.paddleocr import (
    PaddleOCRJob,
    PaddleOCRModel,
    PaddleOCRRuntime,
    _load_engine_kwargs,
)


def _write_image(path: Path) -> None:
    Image.new("RGB", (20, 10), color=(255, 255, 255)).save(path)


class _FakeEngine:
    def ocr(self, image_path: str, cls: bool = True):
        assert cls is True
        return [
            [
                [
                    [[1.0, 1.0], [9.0, 1.0], [9.0, 5.0], [1.0, 5.0]],
                    ("HELLO", 0.91),
                ]
            ]
        ]


def test_paddleocr_process_batch_writes_ocr_boxes(tmp_path: Path) -> None:
    image_path = tmp_path / "ocr.jpg"
    _write_image(image_path)

    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=image_path, width=20, height=10))],
        fm_request=FMRequest(provider="paddleocr", task="ocr"),
        root=tmp_path,
    )

    runtime = PaddleOCRRuntime(engine=_FakeEngine(), use_gpu=False)
    result = PaddleOCRModel().process_batch(
        dataset,
        [PaddleOCRJob(record_idx=0, image_path=image_path)],
        runtime,
        None,
    )

    rec = dataset.records[0]
    assert result.modified_record_indices == [0]
    assert dataset.task == Task.det
    assert dataset.classes == ["text"]
    assert rec.task == Task.det
    assert rec.boxes[0].kind == "ocr"
    assert rec.boxes[0].label == "text"
    assert rec.boxes[0].text == "HELLO"
    assert rec.boxes[0].score == 0.91
    assert rec.boxes[0].model == "paddleocr"


def test_load_engine_kwargs_reads_yaml_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "ocr.yaml"
    config_path.write_text("lang: en\nocr_version: PP-OCRv5\n", encoding="utf-8")

    assert _load_engine_kwargs(config_path) == {
        "lang": "en",
        "ocr_version": "PP-OCRv5",
    }


def test_load_engine_kwargs_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "ocr.yaml"
    config_path.write_text("- en\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level mapping"):
        _load_engine_kwargs(config_path)
