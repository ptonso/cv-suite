"""The migration comparison tool: `scripts/compare_datasets.py`."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from cvsuite.common.core import BBox, ImageInfo, VisionDataset, VisionRecord
from cvsuite.common.io import read_dataset, write_dataset

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "compare_datasets.py"
spec = importlib.util.spec_from_file_location("compare_datasets", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def build(tmp_path: Path, name: str, mutate=None) -> Path:
    src = tmp_path / f"src_{name}"
    src.mkdir()
    records = []
    for n in range(3):
        path = src / f"im{n}.png"
        Image.fromarray(np.full((20, 20, 3), 40 * n, np.uint8)).save(path)
        records.append(VisionRecord(image=ImageInfo(path, 20, 20), rel_image_path=Path(f"im{n}.png"), split="train", boxes=[BBox(0.5, 0.5, 0.2, 0.2, n % 2)]))
    dataset = VisionDataset(records=records, classes=["a", "b"], root=src)
    if mutate:
        mutate(dataset)
    out = tmp_path / name
    write_dataset(dataset, out, format="yolo", task="det")
    (out / "data.yaml").write_text("path: .\ntrain: train/images\n" + ("val: val/images\n" if (out / "val").exists() else "") + "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(dataset.classes)), encoding="utf-8")
    return out


def run(tmp_path: Path, mutate=None, **kw) -> list[str]:
    tmp_path = Path(tempfile.mkdtemp(dir=tmp_path))
    a, b = build(tmp_path, "a"), build(tmp_path, "b", mutate)
    return tool.compare(read_dataset(a, format="yolo"), read_dataset(b, format="yolo"), **kw)


def test_identical_datasets_match(tmp_path: Path) -> None:
    assert run(tmp_path) == []


def test_a_moved_box_is_reported(tmp_path: Path) -> None:
    def move(ds):
        ds.records[1].boxes[0].cx = 0.7

    assert any("box" in d for d in run(tmp_path, move))


def test_a_missing_record_is_reported(tmp_path: Path) -> None:
    def drop(ds):
        ds.records.pop()

    assert any(d.startswith("only in A") for d in run(tmp_path, drop))


def test_a_changed_class_is_reported(tmp_path: Path) -> None:
    def rename(ds):
        ds.classes = ["a", "c"]

    assert any(d.startswith("classes") for d in run(tmp_path, rename))


def test_splits_are_ignored_on_request(tmp_path: Path) -> None:
    def resplit(ds):
        ds.records[0].split = "val"

    assert run(tmp_path, resplit) != []
    assert run(tmp_path, resplit, ignore_splits=True) == []


def test_command_line_exit_status(tmp_path: Path) -> None:
    a, b = build(tmp_path, "a"), build(tmp_path, "b")
    assert tool.main([str(a), str(b), "--from", "yolo"]) == 0


def test_records_sharing_a_layout_path_are_matched_by_their_annotations(tmp_path: Path) -> None:
    def twin(ds):
        ds.records.append(VisionRecord(image=ds.records[0].image, rel_image_path=Path("im0.png"), split="val", boxes=[BBox(0.3, 0.3, 0.1, 0.1, 1)]))

    a, b = build(tmp_path, "a", twin), build(tmp_path, "b", twin)
    assert tool.compare(read_dataset(a, format="yolo"), read_dataset(b, format="yolo"), ignore_splits=True) == []


def test_content_key_matches_renamed_files(tmp_path: Path) -> None:
    def rename(ds):
        ds.records[0].rel_image_path = Path("renamed.png")
        ds.records[0].image = ImageInfo(ds.records[0].image.path.rename(ds.records[0].image.path.with_name("renamed.png")), 20, 20)

    assert run(tmp_path, rename, key="content") == []
