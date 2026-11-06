from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from PIL import Image
import yaml

from cvsuite.classify.core import io as classify_io
from cvsuite.classify.output.commands import to_class_dir
from cvsuite.common.core import Classification, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.io.common import resolve_record_image_path


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=(255, 255, 255)).save(path)
    return path


def test_to_class_dir_writes_stats_yaml(tmp_path: Path) -> None:
    src = tmp_path / "src"
    cat = _write_image(src / "cat.jpg")
    dog = _write_image(src / "dog.jpg")
    dataset = VisionDataset(
        records=[
            Record(image=ImageRecord(path=cat, width=20, height=20), classification=Classification(label="cat", score=0.9)),
            Record(image=ImageRecord(path=dog, width=20, height=20), classification=Classification(label="dog", score=0.2)),
        ],
        root=src,
    )

    dst = tmp_path / "out"
    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=0.5,
            class_thresholds=None,
            class_threshold=[],
            multi_class=False,
            unlabeled_name="unlabeled",
            randomized=False,
            with_stats=True,
        ),
    )

    stats = yaml.safe_load((dst / "stats.yaml").read_text(encoding="utf-8"))
    assert stats["command"] == "to-class-dir"
    assert stats["summary"]["assignments_by_output_label"] == {"cat": 1, "unlabeled": 1}
    assert stats["summary"]["unlabeled_records"] == 1


def test_to_class_dir_stats_auto_enable_multi_class_for_multi_class_ingest(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_image(src / "cat" / "shared.jpg")
    _write_image(src / "dog" / "shared.jpg")
    dataset = classify_io.ingest(src, from_hint="multi-class")

    dst = tmp_path / "out"
    to_class_dir.run(
        dataset,
        Namespace(
            dst=dst,
            threshold=None,
            class_thresholds=None,
            class_threshold=[],
            multi_class=False,
            unlabeled_name="unlabeled",
            randomized=False,
            with_stats=True,
        ),
    )

    stats = yaml.safe_load((dst / "stats.yaml").read_text(encoding="utf-8"))
    assert stats["summary"]["multi_class_enabled"] is True
    assert stats["summary"]["multi_class_records"] == 1
    assert stats["summary"]["assignments_by_output_label"] == {"cat": 1, "dog": 1}


def test_multi_class_ingest_of_a_relative_source_resolves_its_images(tmp_path: Path, monkeypatch) -> None:
    _write_image(tmp_path / "src" / "cat" / "shared.jpg")
    _write_image(tmp_path / "src" / "dog" / "shared.jpg")
    monkeypatch.chdir(tmp_path)
    dataset = classify_io.ingest(Path("src"), from_hint="multi-class")
    assert all(resolve_record_image_path(dataset, r).is_file() for r in dataset.records)
