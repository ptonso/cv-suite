from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

from PIL import Image
import yaml

from cvsuite.common.core import BBox, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.label import cli as label_cli
from cvsuite.label.transform.sample.commands.run import run as run_sample


def _write_image(path: Path, *, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color=color).save(path)
    return path


def _write_yolo_root(root: Path, *, split: str, stem: str) -> Path:
    image_path = root / split / "images" / f"{stem}.jpg"
    label_path = root / split / "labels" / f"{stem}.txt"
    _write_image(image_path, color=(255, 255, 255))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")
    data = {
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "names": ["light"],
        "task": "det",
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


def test_label_sample_preserves_annotations_and_split() -> None:
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=Path("/tmp/a.jpg"), width=24, height=24),
                split="val",
                boxes=[BBox(cx=0.5, cy=0.5, w=0.4, h=0.3, cls=0, label="light")],
            ),
            Record(
                image=ImageRecord(path=Path("/tmp/b.jpg"), width=24, height=24),
                split="train",
                boxes=[BBox(cx=0.6, cy=0.4, w=0.2, h=0.2, cls=0, label="light")],
            ),
        ],
        classes=["light"],
    )

    sampled = run_sample(dataset, Namespace(count=1, frac=None, seed=0))

    assert len(sampled.records) == 1
    assert sampled.records[0].boxes
    assert sampled.records[0].split in {"train", "val"}
    assert sampled.meta["sample"]["sample_size"] == 1


def test_label_cli_sample_merges_multiple_sources_and_preserves_splits_on_output(tmp_path: Path) -> None:
    train_src = _write_yolo_root(tmp_path / "train_src", split="train", stem="light_train")
    val_src = _write_yolo_root(tmp_path / "val_src", split="val", stem="light_val")
    dst = tmp_path / "out"

    exit_code = label_cli.main(
        [
            str(train_src),
            str(val_src),
            "sample",
            "--count",
            "2",
            "--hardlink",
            "to-yolo",
            str(dst),
            "--preserve-splits",
        ]
    )

    assert exit_code == 0
    assert (dst / "train" / "labels" / "light_train.txt").exists()
    assert (dst / "val" / "labels" / "light_val.txt").exists()
    assert os.stat(dst / "train" / "images" / "light_train.jpg").st_ino == os.stat(train_src / "train" / "images" / "light_train.jpg").st_ino
    assert os.stat(dst / "val" / "images" / "light_val.jpg").st_ino == os.stat(val_src / "val" / "images" / "light_val.jpg").st_ino
