from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from cvsuite.common.core import BBox, ImageRecord, Record, VisionDataset
from cvsuite.label.core import router
from cvsuite.label.output.commands import to_coco


def _dataset(tmp_path: Path, n: int = 3) -> VisionDataset:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        p = src / f"img{i}.jpg"
        Image.new("RGB", (20, 16), color=(10 * i, 0, 0)).save(p)
        records.append(
            Record(
                image=ImageRecord(path=p, width=20, height=16),
                boxes=[BBox(cx=0.5, cy=0.5, w=0.4, h=0.4, cls=0, label="thing")],
            )
        )
    return VisionDataset(records=records, classes=["thing"], root=src)


def _ns(dst: Path, **over) -> Namespace:
    base = dict(dst=dst, val_frac=0.0, test_frac=0.0, seed=0, preserve_splits=False,
               coco_style=False, no_images=False, hardlink=False, with_stats=False)
    base.update(over)
    return Namespace(**base)


def test_to_coco_writes_a_directory_bundle(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    dst = tmp_path / "out" / "coco"
    to_coco.run(ds, _ns(dst))

    assert (dst / "train" / "_annotations.coco.json").is_file()
    assert (dst / "data.yaml").is_file()
    assert sorted(p.name for p in (dst / "train").glob("*.jpg")) == ["img0.jpg", "img1.jpg", "img2.jpg"]


def test_to_coco_rejects_a_file_target(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="directory, not a file"):
        to_coco.run(_dataset(tmp_path), _ns(tmp_path / "out.json"))


def test_to_coco_no_images_writes_only_annotations_and_round_trips(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    dst = tmp_path / "out"
    to_coco.run(ds, _ns(dst, no_images=True))

    assert (dst / "train" / "_annotations.coco.json").is_file()
    assert not list((dst / "train").glob("*.jpg"))

    back = router.ingest(dst)
    assert len(back.records) == 3
    assert sum(len(r.boxes) for r in back.records) == 3


def test_to_coco_roundtrips_with_images(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    dst = tmp_path / "out"
    to_coco.run(ds, _ns(dst))
    back = router.ingest(dst)
    assert len(back.records) == 3
