from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from cvsuite.label.core import router


def _ultra_root(root: Path, *, with_labels: bool, task_dir: str = "labels") -> Path:
    (root / "images" / "train").mkdir(parents=True)
    if with_labels:
        (root / task_dir / "train").mkdir(parents=True)
    for stem in ("a", "b", "c"):
        Image.new("RGB", (20, 16)).save(root / "images" / "train" / f"{stem}.jpg")
        if with_labels:
            (root / task_dir / "train" / f"{stem}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (root / "data.yaml").write_text(
        json.dumps({"path": ".", "train": "images/train", "val": "images/train", "nc": 1, "names": ["obj"]})
    )
    return root


def test_yolo_reads_ultralytics_images_split_layout(tmp_path: Path) -> None:
    ds = router.ingest(_ultra_root(tmp_path / "ds", with_labels=True))
    assert len(ds.records) == 6  # train + val both point at images/train
    assert sum(len(r.boxes) for r in ds.records) == 6


def test_yolo_fails_loud_when_images_have_no_labels(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="0 label files"):
        router.ingest(_ultra_root(tmp_path / "ds", with_labels=False))
