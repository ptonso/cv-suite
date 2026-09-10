from __future__ import annotations

from pathlib import Path

from PIL import Image
import yaml

from cvsuite.label import cli as label_cli
from cvsuite.label.core import router
from cvsuite.common.core.enums import Task
from cvsuite.common.core import BBox, ImageRecord, Polygon, Record, VQA
from cvsuite.common.core import VisionDataset
import cvsuite.label.output.commands.to_labelme as cmd_to_labelme
import cvsuite.label.output.commands.to_yolo as cmd_to_yolo


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color=color).save(path)


def _write_yolo_dataset(root: Path, *, cls_name: str, stem: str) -> Path:
    image_path = root / "train" / "images" / f"{stem}.jpg"
    label_path = root / "train" / "labels" / f"{stem}.txt"
    _write_image(image_path, color=(255, 255, 255))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")
    data_yaml = {
        "path": str(root),
        "train": "train/images",
        "names": [cls_name],
        "task": "det",
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return root


def _make_multitask_dataset(root: Path) -> VisionDataset:
    image_path = root / "multi.jpg"
    _write_image(image_path, color=(64, 128, 255))
    return VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=24, height=24),
                split="train",
                task=Task.seg,
                boxes=[BBox(cx=0.5, cy=0.5, w=0.4, h=0.3, cls=0, kind="det")],
                polys=[Polygon(points=[(0.15, 0.2), (0.8, 0.2), (0.75, 0.85), (0.2, 0.8)], cls=0)],
            )
        ],
        classes=["light"],
        task=Task.seg,
        root=root,
    )


def test_ingest_many_merges_sources_and_remaps_class_ids(tmp_path: Path) -> None:
    lights = _write_yolo_dataset(tmp_path / "lights", cls_name="light", stem="light_a")
    valves = _write_yolo_dataset(tmp_path / "valves", cls_name="valve", stem="valve_b")

    dataset = router.ingest_many([lights, valves], from_hint="yolo")

    assert dataset.classes == ["light", "valve"]
    assert dataset.root == tmp_path
    assert dataset.meta["multisrc"] is True
    assert len(dataset.records) == 2
    assert all(rec.image.path.is_absolute() for rec in dataset.records)

    class_by_stem = {rec.image.path.stem: rec.boxes[0].cls for rec in dataset.records}
    label_by_stem = {rec.image.path.stem: rec.boxes[0].label for rec in dataset.records}
    assert class_by_stem == {"light_a": 0, "valve_b": 1}
    assert label_by_stem == {"light_a": "light", "valve_b": "valve"}


def test_label_cli_accepts_multiple_sources_before_transform(tmp_path: Path) -> None:
    lights = tmp_path / "lights"
    valves = tmp_path / "valves"
    _write_image(lights / "light.jpg", color=(255, 0, 0))
    _write_image(valves / "valve.jpg", color=(0, 255, 0))
    dst = tmp_path / "labelme_out"

    exit_code = label_cli.main(
        [
            str(lights),
            str(valves),
            "to-labelme",
            "--val-frac",
            "1.0",
            str(dst),
        ]
    )

    assert exit_code == 0
    dataset = router.ingest_many([lights, valves])
    assert dataset.meta["sources"] == [str(lights), str(valves)]
    assert len(list((dst / "val").glob("*.json"))) == 2


def test_to_yolo_assigns_output_splits(tmp_path: Path) -> None:
    src = _write_yolo_dataset(tmp_path / "source", cls_name="light", stem="light_a")
    ds = router.ingest(src, from_hint="yolo")
    dst = tmp_path / "out"

    args = type(
        "Args",
        (),
        dict(
            dst=dst,
            task="det",
            val_frac=1.0,
            test_frac=0.0,
            seed=0,
            hardlink=False,
        ),
    )

    cmd_to_yolo.run(ds, args)

    assert list((dst / "val" / "labels").glob("*.txt"))
    assert not list((dst / "train" / "labels").glob("*.txt"))
    data = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))
    assert data["train"] == "val/images"
    assert data["val"] == "val/images"
    assert "test" not in data


def test_to_yolo_defaults_to_train_only_output(tmp_path: Path) -> None:
    src = _write_yolo_dataset(tmp_path / "source_default", cls_name="light", stem="light_b")
    ds = router.ingest(src, from_hint="yolo")
    ds.records[0].split = "val"
    dst = tmp_path / "out_default"

    args = type(
        "Args",
        (),
        dict(
            dst=dst,
            task="det",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            hardlink=False,
        ),
    )

    cmd_to_yolo.run(ds, args)

    assert list((dst / "train" / "labels").glob("*.txt"))
    assert not (dst / "val").exists()
    assert not (dst / "test").exists()
    data = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))
    assert data["train"] == "train/images"
    assert data["val"] == "train/images"
    assert "test" not in data


def test_to_labelme_defaults_to_flat_train_only_output(tmp_path: Path) -> None:
    src = _write_yolo_dataset(tmp_path / "source_labelme_default", cls_name="light", stem="light_flat")
    ds = router.ingest(src, from_hint="yolo")
    ds.records[0].split = "val"
    dst = tmp_path / "labelme_flat"

    cmd_to_labelme.run(
        ds,
        type(
            "Args",
            (),
            dict(dst=dst, val_frac=0.0, test_frac=0.0, seed=0, embed_image=False, hardlink=False),
        ),
    )

    assert (dst / "light_flat.json").exists()
    assert (dst / "light_flat.jpg").exists()
    assert not (dst / "train").exists()
    assert router.ingest(dst, from_hint="labelme").records[0].split == "train"


def test_to_labelme_preserve_splits_uses_split_dirs(tmp_path: Path) -> None:
    train_img = tmp_path / "train_img.jpg"
    val_img = tmp_path / "val_img.jpg"
    _write_image(train_img, color=(255, 255, 255))
    _write_image(val_img, color=(0, 255, 0))
    ds = VisionDataset(
        records=[
            Record(image=ImageRecord(path=train_img, width=24, height=24), split="train", task=Task.det),
            Record(image=ImageRecord(path=val_img, width=24, height=24), split="val", task=Task.det),
        ],
        root=tmp_path,
    )
    dst = tmp_path / "labelme_preserve"

    cmd_to_labelme.run(
        ds,
        type(
            "Args",
            (),
            dict(
                dst=dst,
                val_frac=0.0,
                test_frac=0.0,
                seed=0,
                preserve_splits=True,
                embed_image=False,
                hardlink=False,
            ),
        ),
    )

    assert (dst / "train" / "train_img.json").exists()
    assert (dst / "val" / "val_img.json").exists()
    assert not list(dst.glob("*.json"))


def test_to_yolo_writes_data_yaml_for_image_only_dataset(tmp_path: Path) -> None:
    src = tmp_path / "arranged"
    _write_image(src / "sample.jpg", color=(12, 34, 56))
    ds = router.ingest(src, from_hint="images")
    dst = tmp_path / "out_images_only"

    args = type(
        "Args",
        (),
        dict(
            dst=dst,
            task="auto",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            hardlink=False,
        ),
    )

    cmd_to_yolo.run(ds, args)

    data = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))
    assert data["task"] == "det"
    assert data["names"] == []
    assert data["train"] == "train/images"
    assert data["val"] == "train/images"


def test_to_yolo_skips_labels_vqa_when_dataset_has_no_vqa(tmp_path: Path) -> None:
    src = _write_yolo_dataset(tmp_path / "source_no_vqa", cls_name="light", stem="light_c")
    ds = router.ingest(src, from_hint="yolo")
    dst = tmp_path / "out_no_vqa"

    args = type(
        "Args",
        (),
        dict(
            dst=dst,
            task="det",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            hardlink=False,
        ),
    )

    cmd_to_yolo.run(ds, args)

    assert not list(dst.rglob("labels_vqa")), "labels_vqa should not be created for datasets without VQA entries"


def test_to_yolo_creates_labels_vqa_when_any_output_split_has_vqa(tmp_path: Path) -> None:
    src = _write_yolo_dataset(tmp_path / "source_with_vqa", cls_name="light", stem="light_d")
    ds = router.ingest(src, from_hint="yolo")
    ds.records[0].vqas = [VQA(question="What is shown?", answer="A light")]
    dst = tmp_path / "out_with_vqa"

    args = type(
        "Args",
        (),
        dict(
            dst=dst,
            task="det",
            val_frac=1.0,
            test_frac=0.0,
            seed=0,
            hardlink=False,
        ),
    )

    cmd_to_yolo.run(ds, args)

    vqa_file = dst / "val" / "labels_vqa" / "light_d.txt"
    assert vqa_file.exists(), "labels_vqa should be created when at least one exported record has VQA entries"
    assert "What is shown?" in vqa_file.read_text(encoding="utf-8")


def test_to_yolo_auto_uses_det_as_plain_labels_when_multiple_tasks_exist(tmp_path: Path) -> None:
    ds = _make_multitask_dataset(tmp_path / "multi_auto")
    dst = tmp_path / "multi_auto_out"

    args = type(
        "Args",
        (),
        dict(
            dst=dst,
            task="auto",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            hardlink=False,
        ),
    )

    cmd_to_yolo.run(ds, args)

    det_label = dst / "train" / "labels" / "multi.txt"
    seg_label = dst / "train" / "labels_seg" / "multi.txt"
    assert det_label.exists()
    assert seg_label.exists()
    assert len(det_label.read_text(encoding="utf-8").strip().split()) == 5
    assert len(seg_label.read_text(encoding="utf-8").strip().split()) > 5

    data = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))
    assert data["task"] == "det"


def test_to_yolo_explicit_task_uses_plain_labels_and_preserves_other_tasks(tmp_path: Path) -> None:
    ds = _make_multitask_dataset(tmp_path / "multi_seg")
    dst = tmp_path / "multi_seg_out"

    args = type(
        "Args",
        (),
        dict(
            dst=dst,
            task="seg",
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            hardlink=False,
        ),
    )

    cmd_to_yolo.run(ds, args)

    seg_label = dst / "train" / "labels" / "multi.txt"
    det_label = dst / "train" / "labels_det" / "multi.txt"
    assert seg_label.exists()
    assert det_label.exists()
    assert len(seg_label.read_text(encoding="utf-8").strip().split()) > 5
    assert len(det_label.read_text(encoding="utf-8").strip().split()) == 5

    data = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))
    assert data["task"] == "seg"

    ds_auto = router.ingest(dst / "data.yaml", from_hint="yolo", task="auto")
    assert ds_auto.task == Task.seg
    assert ds_auto.records and ds_auto.records[0].polys

    ds_det = router.ingest(dst / "data.yaml", from_hint="yolo", task="det")
    assert ds_det.task == Task.det
    assert ds_det.records and ds_det.records[0].boxes
