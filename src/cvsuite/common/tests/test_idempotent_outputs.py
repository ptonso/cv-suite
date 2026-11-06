"""Output writers claim what they write: a dirty rerun raises, `--overwrite` replaces, same-run collisions get a suffix."""

import argparse
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

import cvsuite.label.output.commands.to_coco as cmd_to_coco
import cvsuite.label.output.commands.to_yolo as cmd_to_yolo
from cvsuite.common.core import BBox, Classification, ImageInfo, Polygon, VisionDataset, VisionRecord
from cvsuite.common.io import write_dataset

POLY = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)]


def image(path: Path, seed: int = 0) -> ImageInfo:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.default_rng(seed).random((8, 8, 3)) * 255).astype(np.uint8)).save(path)
    return ImageInfo(path=path, width=8, height=8)


def record(path: Path, split: str = "train", polys: bool = False, seed: int = 0) -> VisionRecord:
    return VisionRecord(
        image=image(path, seed),
        split=split,
        boxes=[BBox(0.5, 0.5, 0.2, 0.2, 0, "a")],
        polys=[Polygon(POLY, 0, "a")] if polys else [],
        classification=Classification(label="cat", score=1.0),
    )


def dataset(root: Path, names: list[str], split: str = "train", polys: bool = False) -> VisionDataset:
    return VisionDataset(records=[record(root / "src" / n, split, polys, i) for i, n in enumerate(names)], classes=["a"], root=root)


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


@pytest.mark.parametrize("fmt", ["yolo", "labelme", "class-dir", "coco"])
def test_dirty_rerun_raises_and_overwrite_replaces(tmp_path: Path, fmt: str) -> None:
    ds, dst = dataset(tmp_path, ["a.jpg", "b.jpg"]), tmp_path / "out"
    write_dataset(ds, dst, format=fmt)
    first = tree(dst)

    with pytest.raises(FileExistsError, match="--overwrite"):
        write_dataset(ds, dst, format=fmt)
    assert tree(dst) == first

    write_dataset(ds, dst, format=fmt, overwrite=True)
    assert tree(dst) == first
    assert not [p for p in first if "_1" in p]


def test_same_run_collisions_get_a_suffix(tmp_path: Path) -> None:
    ds = dataset(tmp_path, ["a.jpg"])
    ds.records.append(record(tmp_path / "other" / "a.jpg", seed=5))
    write_dataset(ds, tmp_path / "out", format="yolo")
    assert tree(tmp_path / "out" / "train" / "images") == ["a.jpg", "a_1.jpg"]


def test_overwrite_refuses_the_folder_of_the_source_images(tmp_path: Path) -> None:
    ds = dataset(tmp_path, ["a.jpg"])
    src_dir = tmp_path / "src"
    write_dataset(ds, tmp_path / "out", format="labelme")
    with pytest.raises(ValueError, match="source images"):
        write_dataset(ds, src_dir, format="labelme", overwrite=True)
    assert (src_dir / "a.jpg").exists()


def test_train_then_val_into_one_root_keeps_both_in_data_yaml(tmp_path: Path) -> None:
    dst = tmp_path / "out"
    for split, name in (("train", "a.jpg"), ("val", "b.jpg")):
        cmd_to_yolo.run(dataset(tmp_path, [name], split), argparse.Namespace(dst=dst, task="det"))
    data = yaml.safe_load((dst / "data.yaml").read_text())
    assert data["train"] == "train/images"
    assert data["val"] == "val/images"
    assert tree(dst / "val" / "images") == ["b.jpg"]


def test_data_yaml_names_only_splits_on_disk(tmp_path: Path) -> None:
    dst = tmp_path / "out"
    args = argparse.Namespace(dst=dst, task="det", overwrite=True)
    cmd_to_yolo.run(dataset(tmp_path, ["a.jpg"], "val"), args)
    cmd_to_yolo.run(dataset(tmp_path, ["a.jpg"], "train"), args)
    cmd_to_yolo.run(dataset(tmp_path, ["a.jpg"], "train"), args)
    data = yaml.safe_load((dst / "data.yaml").read_text())
    assert (data["train"], data["val"]) == ("train/images", "val/images")


def test_det_and_seg_labels_pair_by_stem_for_records_of_one_image(tmp_path: Path) -> None:
    ds = dataset(tmp_path, ["a.jpg"], polys=True)
    ds.records.append(VisionRecord(image=ds.records[0].image, split="train", boxes=ds.records[0].boxes, polys=ds.records[0].polys))
    write_dataset(ds, tmp_path / "out", format="yolo")
    train = tmp_path / "out" / "train"
    assert tree(train / "images") == ["a.jpg", "a_1.jpg"]
    assert tree(train / "labels") == ["a.txt", "a_1.txt"]
    assert tree(train / "labels_seg") == ["a.txt", "a_1.txt"]


def test_coco_style_overwrite_keeps_the_images_of_every_split(tmp_path: Path) -> None:
    ds = dataset(tmp_path, ["a.jpg"])
    ds.records.append(record(tmp_path / "src" / "b.jpg", "val", seed=1))
    dst = tmp_path / "out"
    args = argparse.Namespace(dst=dst, coco_style=True, no_images=False, overwrite=True)
    cmd_to_coco.run(ds, args)
    cmd_to_coco.run(ds, args)
    assert tree(dst / "images") == ["a.jpg", "b.jpg"]
    assert tree(dst / "annotations") == ["annotations_train.json", "annotations_val.json"]
    assert set(yaml.safe_load((dst / "data.yaml").read_text())) >= {"train", "val"}


def test_class_dir_nested_class_folders_do_not_conflict(tmp_path: Path) -> None:
    """A class folder and a class folder inside it (`floors`, `floors/drain`) are both written, whatever the record order."""
    records = []
    for n, label in enumerate(["floors/drain", "floors"]):
        rec = record(tmp_path / "src" / f"{n}.jpg", seed=n)
        rec.classification = Classification(label=label, score=1.0)
        records.append(rec)
    dst = tmp_path / "out"
    write_dataset(VisionDataset(records=records, classes=["floors", "floors/drain"], root=tmp_path), dst, format="class-dir")
    assert sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*.jpg")) == ["floors/1.jpg", "floors/drain/0.jpg"]
