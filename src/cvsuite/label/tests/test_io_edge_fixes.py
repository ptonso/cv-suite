"""COCO empty images, multi-source merging, YOLO names, polygon boxes, negatives and context crops."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

import cvsuite.label.output.commands.to_coco as cmd_to_coco
import cvsuite.label.output.commands.to_yolo as cmd_to_yolo
import cvsuite.label.output.commands.to_labelme as cmd_to_labelme
import cvsuite.label.transform.ops.commands.crop_dets as cmd_crop_dets
import cvsuite.label.transform.ops.commands.inverse_crop_dets as cmd_inverse
import cvsuite.label.transform.ops.commands.make_negatives as cmd_negatives
from cvsuite.branch_cli import build_parser
from cvsuite.common.core import BBox, ImageInfo, Keypoints, Polygon, VisionDataset, VisionRecord
from cvsuite.common.io import read_dataset, write_dataset
from cvsuite.label import cli as label_cli
from cvsuite.label.core import router
from cvsuite.label.transform.ops.core.negatives import annotation_rects
from cvsuite.label.transform.sample.commands.run import run as run_sample

EXAMPLES = Path(__file__).resolve().parents[4] / "docs" / "example_data"


def image(path: Path, w: int = 100, h: int = 80, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)).save(path)
    return path


def coco_file(root: Path, images: list[dict], annotations: list[dict], categories: list[dict], name: str = "coco.json") -> Path:
    for item in images:
        image(root / item["file_name"], item["width"], item["height"])
    path = root / name
    path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return path


IMAGES = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 80}, {"id": 2, "file_name": "b.jpg", "width": 100, "height": 80}]
ANN = [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "area": 400}]
CATS = [{"id": 3, "name": "cat"}]


def test_coco_reader_keeps_images_without_annotations(tmp_path: Path) -> None:
    ds = read_dataset(coco_file(tmp_path, IMAGES, ANN, CATS), format="coco")
    assert sorted(r.image.path.name for r in ds.records) == ["a.jpg", "b.jpg"]
    assert sorted(len(r.boxes) for r in ds.records) == [0, 1]
    assert ds.classes == ["cat"]


def test_example_coco_dataset_reads_as_one_record() -> None:
    ds = read_dataset(EXAMPLES / "dataset-coco")
    assert len(ds.records) == 1 and ds.records[0].boxes == []


def test_empty_images_survive_to_coco_split_and_sample(tmp_path: Path) -> None:
    src = coco_file(tmp_path / "src", IMAGES, ANN, CATS)
    out = tmp_path / "out"
    assert label_cli.main([str(src), "--from", "coco", "to-coco", str(out)]) == 0
    payload = json.loads((out / "train" / "_annotations.coco.json").read_text())
    assert len(payload["images"]) == 2 and len(payload["annotations"]) == 1
    ds = router.ingest(src, from_hint="coco")
    assert len(run_sample(ds, Namespace(count=2, frac=None, seed=0)).records) == 2


@pytest.mark.parametrize(
    "annotation, message",
    [
        ({"id": 9, "image_id": 1, "category_id": 99, "bbox": [0, 0, 5, 5]}, "category id 99"),
        ({"id": 9, "image_id": 7, "category_id": 3, "bbox": [0, 0, 5, 5]}, "image id 7"),
        ({"id": 9, "image_id": 1, "bbox": [0, 0, 5, 5]}, "category id None"),
    ],
)
def test_coco_reader_rejects_unknown_ids(tmp_path: Path, annotation: dict, message: str) -> None:
    path = coco_file(tmp_path, IMAGES, [annotation], CATS)
    with pytest.raises(ValueError, match=message):
        read_dataset(path, format="coco")


def test_multi_source_coco_merges_by_class_name(tmp_path: Path) -> None:
    one = coco_file(tmp_path / "one", IMAGES, ANN, CATS)
    two = coco_file(tmp_path / "two", IMAGES, [{"id": 5, "image_id": 1, "category_id": 11, "bbox": [5, 5, 10, 10]}], [{"id": 10, "name": "dog"}, {"id": 11, "name": "cat"}])
    ds = router.ingest_many([one, two], from_hint="coco")
    assert ds.classes == ["cat", "dog"]
    assert len(ds.records) == 4
    assert sorted(b.cls for r in ds.records for b in r.boxes) == [0, 0]


def test_ingest_many_with_a_source_without_root(tmp_path: Path, monkeypatch) -> None:
    sources = []
    for name in ("x", "y"):
        path = image(tmp_path / name / "a.jpg")
        sources.append(path.parent)

    real = router.ingest

    def rootless(src, **kw):
        ds = real(src, **kw)
        ds.root = None
        return ds

    monkeypatch.setattr(router, "ingest", rootless)
    ds = router.ingest_many(sources, from_hint="images")
    assert len(ds.records) == 2


def test_yolo_names_as_list_or_mapping(tmp_path: Path) -> None:
    for form in ("names: [dog]\n", "names:\n  0: dog\n"):
        root = tmp_path / str(abs(hash(form)))
        image(root / "train" / "images" / "a.jpg")
        (root / "train" / "labels").mkdir(parents=True)
        (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (root / "data.yaml").write_text(form)
        ds = read_dataset(root)
        assert ds.classes == ["dog"] and ds.records[0].boxes[0].label == "dog"


def test_yolo_names_mapping_must_be_contiguous(tmp_path: Path) -> None:
    image(tmp_path / "train" / "images" / "a.jpg")
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "data.yaml").write_text("names:\n  0: a\n  2: b\n")
    with pytest.raises(ValueError, match="indices"):
        read_dataset(tmp_path)


def test_example_yolo_dataset_reads_names_and_boxes_through_both_entry_points() -> None:
    assert read_dataset(EXAMPLES / "dataset-yolo").classes == ["dog"]
    ds = router.ingest(EXAMPLES / "dataset-yolo")
    assert ds.classes == ["dog"] and ds.records[0].boxes


def test_names_survive_a_write_and_read(tmp_path: Path) -> None:
    ds = read_dataset(EXAMPLES / "dataset-yolo")
    cmd_to_yolo.run(ds, Namespace(dst=tmp_path / "out", task="det"))
    assert read_dataset(tmp_path / "out").classes == ["dog"]


def poly_dataset(tmp_path: Path) -> VisionDataset:
    path = image(tmp_path / "p.jpg")
    pts = [(0.2, 0.25), (0.6, 0.25), (0.6, 0.75), (0.2, 0.75)]
    record = VisionRecord(image=ImageInfo(path, 100, 80), polys=[Polygon(pts, 0, "a")], boxes=[BBox(0.4, 0.5, 0.4, 0.5, 0, "a", kind="seg")])
    return VisionDataset(records=[record], classes=["a"], root=tmp_path)


def test_polygon_only_det_needs_the_flag(tmp_path: Path) -> None:
    ds = poly_dataset(tmp_path)
    with pytest.raises(ValueError, match="not present"):
        cmd_to_yolo.run(ds, Namespace(dst=tmp_path / "o1", task="det"))
    cmd_to_yolo.run(ds, Namespace(dst=tmp_path / "o2", task="det", boxes_from_polygons=True))
    lines = (tmp_path / "o2" / "train" / "labels" / "p.txt").read_text().strip().splitlines()
    assert len(lines) == 1
    cls, cx, cy, w, h = lines[0].split()
    assert (cls, float(cx), float(cy), float(w), float(h)) == ("0", pytest.approx(0.4), pytest.approx(0.5), pytest.approx(0.4), pytest.approx(0.5))
    assert yaml.safe_load((tmp_path / "o2" / "data.yaml").read_text())["task"] == "det"
    assert (tmp_path / "o2" / "train" / "labels_seg" / "p.txt").exists()


def test_polygon_boxes_are_clipped_to_the_image(tmp_path: Path) -> None:
    path = image(tmp_path / "p.jpg")
    pts = [(0.2, -0.1), (0.6, 0.25), (1.2, 0.75), (0.2, 0.75)]
    record = VisionRecord(image=ImageInfo(path, 100, 80), polys=[Polygon(pts, 0, "a")])
    cmd_to_yolo.run(VisionDataset(records=[record], classes=["a"], root=tmp_path), Namespace(dst=tmp_path / "o", task="det", boxes_from_polygons=True))
    cx, cy, w, h = (float(v) for v in (tmp_path / "o" / "train" / "labels" / "p.txt").read_text().split()[1:])
    assert (cx, cy, w, h) == (pytest.approx(0.6), pytest.approx(0.375), pytest.approx(0.8), pytest.approx(0.75))


def dense_record(path: Path, w: int = 200, h: int = 160) -> VisionRecord:
    image(path, w, h)
    boxes = [BBox(0.25 + 0.5 * (i % 2), 0.25 + 0.5 * (i // 2), 0.2, 0.2, 0, "a") for i in range(4)]
    return VisionRecord(image=ImageInfo(path, w, h), boxes=boxes, rel_image_path=Path(path.name))


def neg_args(**kw) -> Namespace:
    base = dict(splits="", imgsz=30, per_image=4, ratio=None, size_from="fixed", on_shortfall="error", seed=1, preview=None)
    return Namespace(**{**base, **kw})


def test_negatives_never_touch_an_annotation(tmp_path: Path) -> None:
    ds = VisionDataset(records=[dense_record(tmp_path / f"d{i}.jpg") for i in range(5)], classes=["a"], root=tmp_path)
    out = cmd_negatives.run(ds, neg_args())
    negatives = [r for r in out.records if r.attributes.get("negative")]
    assert len(negatives) == 20
    source_by_path = {str(tmp_path / f"d{i}.jpg"): ds.records[i] for i in range(5)}
    for neg in negatives:
        x1, y1, x2, y2 = neg.attributes["negative_window_px"]
        assert (x2 - x1, y2 - y1) == (30, 30) and neg.image.path.exists()
        for rx1, ry1, rx2, ry2 in annotation_rects(source_by_path[neg.attributes["source_image"]]):
            assert not (x1 < rx2 and rx1 < x2 and y1 < ry2 and ry1 < y2)


def test_a_full_image_is_a_shortfall(tmp_path: Path) -> None:
    path = image(tmp_path / "full.jpg", 100, 80)
    record = VisionRecord(image=ImageInfo(path, 100, 80), boxes=[BBox(0.5, 0.5, 1.0, 1.0, 0, "a")], rel_image_path=Path("full.jpg"))
    ds = VisionDataset(records=[record], classes=["a"], root=tmp_path)
    with pytest.raises(SystemExit, match=r"full\.jpg: wanted 4, placed 0"):
        cmd_negatives.run(ds, neg_args())
    out = cmd_negatives.run(ds, neg_args(on_shortfall="warn"))
    assert len(out.records) == 1


def test_negatives_ratio_and_size_from_boxes(tmp_path: Path) -> None:
    ds = VisionDataset(records=[dense_record(tmp_path / "d.jpg")], classes=["a"], root=tmp_path)
    out = cmd_negatives.run(ds, neg_args(per_image=None, ratio=0.5))
    assert len([r for r in out.records if r.attributes.get("negative")]) == 2
    out = cmd_negatives.run(ds, neg_args(size_from="boxes", imgsz=999))
    sizes = {r.image.width for r in out.records if r.attributes.get("negative")}
    assert sizes <= {40, 32}, "windows take the larger side of a box (0.2 * 200 = 40, 0.2 * 160 = 32)"
    with pytest.raises(SystemExit):
        build_parser(cmd_negatives, "x").parse_args(["--per-image", "1", "--ratio", "1"])


def test_negative_names_are_unique_and_placement_is_deterministic(tmp_path: Path) -> None:
    records = [dense_record(tmp_path / "a" / "same.jpg"), dense_record(tmp_path / "b" / "same.jpg")]
    for r, folder in zip(records, ("a", "b")):
        r.rel_image_path = Path(folder) / "same.jpg"
    ds = VisionDataset(records=records, classes=["a"], root=tmp_path)
    first = cmd_negatives.run(ds, neg_args())
    names = [r.image.path.name for r in first.records if r.attributes.get("negative")]
    assert len(names) == len(set(names)) == 8 and any(n.startswith("a__same_") for n in names)
    again = cmd_negatives.run(ds, neg_args())
    windows = lambda d: [r.attributes["negative_window_px"] for r in d.records if r.attributes.get("negative")]  # noqa: E731
    assert windows(first) == windows(again)
    extended = VisionDataset(records=[*records, dense_record(tmp_path / "c" / "new.jpg")], classes=["a"], root=tmp_path)
    extended.records[-1].rel_image_path = Path("c/new.jpg")
    assert windows(cmd_negatives.run(extended, neg_args()))[:8] == windows(first)


def test_negatives_keep_every_record_when_splits_are_listed(tmp_path: Path) -> None:
    ds = VisionDataset(records=[dense_record(tmp_path / "t.jpg"), dense_record(tmp_path / "v.jpg")], classes=["a"], root=tmp_path)
    ds.records[1].split = "val"
    out = cmd_negatives.run(ds, neg_args(splits="val"))
    assert len([r for r in out.records if not r.attributes.get("negative")]) == 2
    assert {r.split for r in out.records if r.attributes.get("negative")} == {"val"}


def crop_args(**kw) -> Namespace:
    base = dict(imgsz=64, long_size=None, pad_type="shift", margin_frac=0.0, margin_px=0.0, manifest=None, overwrite=False)
    return Namespace(**{**base, **kw})


def corner_dataset(tmp_path: Path, w: int = 200, h: int = 120, center: tuple[float, float] = (10, 8), smooth: bool = False) -> VisionDataset:
    path = image(tmp_path / "scene.png", w, h, seed=4)
    if smooth:
        ramp = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None].repeat(h, 0).repeat(3, 2)
        Image.fromarray(ramp).save(path)
    box = BBox(cx=center[0] / w, cy=center[1] / h, w=20 / w, h=10 / h, cls=0, label="thing")
    return VisionDataset(records=[VisionRecord(image=ImageInfo(path, w, h), boxes=[box], rel_image_path=Path("scene.png"))], classes=["thing"], root=tmp_path)


def test_shift_crop_near_a_corner_stays_inside_without_padding(tmp_path: Path) -> None:
    out = cmd_crop_dets.run(corner_dataset(tmp_path), crop_args(margin_px=2))
    prov = out.records[0].attributes["crop"]
    x1, y1, x2, y2 = prov["crop_rect_px"]
    assert 0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 120
    assert abs((x2 - x1) - (y2 - y1)) <= 1 and x2 - x1 == 24
    assert (x1, y1) == (0, 0), "a box near the corner slides the square to the corner"
    with Image.open(out.records[0].image.path) as crop:
        assert crop.size == (64, 64)
    assert prov["source_box_px"] == [0.0, 3.0, 20.0, 10.0] and prov["pad_type"] == "shift" and prov["margin_px"] == 2


def test_margin_px_grows_the_window_for_every_pad_type(tmp_path: Path) -> None:
    ds = corner_dataset(tmp_path, center=(60, 50))
    for pad in ("black", "background", "shift"):
        plain = cmd_crop_dets.run(ds, crop_args(pad_type=pad))
        grown = cmd_crop_dets.run(ds, crop_args(pad_type=pad, margin_px=5))
        w0 = plain.records[0].attributes["crop"]["crop_rect_px"]
        w1 = grown.records[0].attributes["crop"]["crop_rect_px"]
        assert (w1[2] - w1[0]) - (w0[2] - w0[0]) == 10, f"{pad}: 5 pixels on each side"


def test_provenance_reproduces_the_crop(tmp_path: Path) -> None:
    ds = corner_dataset(tmp_path, center=(60, 50), smooth=True)
    out = cmd_crop_dets.run(ds, crop_args(margin_px=3))
    prov = out.records[0].attributes["crop"]
    x1, y1, x2, y2 = prov["crop_rect_px"]
    import cv2

    source = cv2.imread(str(ds.root / prov["source_image"]))
    again = cv2.resize(source[y1:y2, x1:x2], (64, 64), interpolation=cv2.INTER_AREA if 64 < x2 - x1 else cv2.INTER_LANCZOS4)
    assert np.abs(again.astype(int) - cv2.imread(str(out.records[0].image.path)).astype(int)).max() <= 6


def test_shift_requires_imgsz(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --imgsz"):
        cmd_crop_dets.run(corner_dataset(tmp_path), crop_args(imgsz=None))


def test_manifest_lists_the_crops_and_refuses_an_existing_file(tmp_path: Path) -> None:
    manifest = tmp_path / "m" / "crops.json"
    out = cmd_crop_dets.run(corner_dataset(tmp_path), crop_args(manifest=manifest))
    entries = json.loads(manifest.read_text())
    assert len(entries) == 1
    assert entries[0]["file"] == Path(out.records[0].image.path).name and entries[0]["split"] == "train" and entries[0]["label"] == "thing"
    assert {"crop_id", "source_image", "source_box_px", "crop_rect_px", "margin_frac", "margin_px", "pad_type"} <= set(entries[0])
    with pytest.raises(FileExistsError):
        cmd_crop_dets.run(corner_dataset(tmp_path), crop_args(manifest=manifest))
    cmd_crop_dets.run(corner_dataset(tmp_path), crop_args(manifest=manifest, overwrite=True))


@pytest.mark.parametrize("extra", [dict(), dict(margin_px=4.0, margin_frac=0.1)])
def test_inverse_crop_dets_pastes_shift_crops_back(tmp_path: Path, extra: dict) -> None:
    w, h = 120, 90
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:] = (12, 24, 36)
    path = tmp_path / "scene.png"
    Image.fromarray(base).save(path)
    box = BBox(cx=95 / w, cy=70 / h, w=20 / w, h=16 / h, cls=0, label="thing")
    ds = VisionDataset(records=[VisionRecord(image=ImageInfo(path, w, h), boxes=[box], rel_image_path=Path("scene.png"))], classes=["thing"], root=tmp_path)
    cropped = cmd_crop_dets.run(ds, crop_args(imgsz=48, **extra))
    meta = cropped.records[0].boxes[0].attributes["cvsuite"]["crop_dets"]
    assert meta["pad_type"] == "shift"
    crop_root = tmp_path / "modified"
    cmd_to_labelme.run(cropped, Namespace(dst=crop_root, embed_image=False))
    jpath = next(crop_root.rglob("*.json"))
    data = json.loads(jpath.read_text())
    Image.new("RGB", (48, 48), (200, 10, 10)).save(jpath.parent / data["imagePath"])
    inverted = cmd_inverse.run(ds, Namespace(modified_crops=crop_root))
    arr = np.asarray(Image.open(inverted.records[0].image.path).convert("RGB"))
    x1, y1, x2, y2 = cropped.records[0].attributes["crop"]["crop_rect_px"]
    assert np.allclose(arr[y1:y2, x1:x2].reshape(-1, 3), (200, 10, 10), atol=3)
    assert np.allclose(arr[0, 0], (12, 24, 36), atol=1)


def test_yolo_split_folders_that_are_symlinks_keep_their_place_under_the_root(tmp_path) -> None:
    from PIL import Image

    from cvsuite.common.io import read_dataset
    from cvsuite.common.io.common import resolve_record_image_path

    store = tmp_path / "store"
    (store / "images").mkdir(parents=True)
    (store / "labels").mkdir()
    Image.new("RGB", (8, 8)).save(store / "images" / "a.jpg")
    (store / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    root = tmp_path / "ds"
    (root / "train").mkdir(parents=True)
    (root / "train" / "images").symlink_to(store / "images")
    (root / "train" / "labels").symlink_to(store / "labels")
    (root / "data.yaml").write_text("path: .\ntrain: train/images\nnames: [x]\n")

    dataset = read_dataset(root, format="yolo")
    assert len(dataset.records) == 1
    assert resolve_record_image_path(dataset, dataset.records[0]).exists()
    assert dataset.records[0].rel_image_path == Path("train/images/a.jpg")


def test_labelme_finds_the_image_beside_the_json_when_image_path_is_stale(tmp_path) -> None:
    from PIL import Image

    from cvsuite.common.io import read_dataset
    from cvsuite.common.io.common import resolve_record_image_path

    folder = tmp_path / "ds" / "point1"
    folder.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(folder / "frame-001.jpg")
    (folder / "frame-001.json").write_text(json.dumps({"imagePath": "other_name.jpg", "imageHeight": 8, "imageWidth": 8, "shapes": [{"label": "a", "points": [[1, 1], [5, 1], [5, 5]], "shape_type": "polygon"}]}))
    dataset = read_dataset(tmp_path / "ds", format="labelme")
    assert len(dataset.records) == 1
    assert resolve_record_image_path(dataset, dataset.records[0]).name == "frame-001.jpg"


def coco_elsewhere(tmp_path: Path, names=("a.jpg", "b.jpg")) -> tuple[Path, Path]:
    from PIL import Image

    ann, imgs = tmp_path / "ann", tmp_path / "pics" / "sub"
    ann.mkdir()
    imgs.mkdir(parents=True)
    for n in names:
        Image.new("RGB", (20, 10)).save(imgs / n)
    coco = {
        "images": [{"id": i, "file_name": n, "width": 20, "height": 10} for i, n in enumerate(names)],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [2, 2, 8, 4]}],
        "categories": [{"id": 0, "name": "x"}],
    }
    (ann / "_annotations.coco.json").write_text(json.dumps(coco))
    return ann, tmp_path / "pics"


def test_coco_images_dir_finds_images_outside_the_folder(tmp_path: Path) -> None:
    from cvsuite.label.core import router

    ann, pics = coco_elsewhere(tmp_path)
    with pytest.raises(FileNotFoundError, match="--images-dir"):
        router.ingest_many([ann], from_hint="coco")
    dataset = router.ingest_many([ann], from_hint="coco", images_dir=[pics])
    assert len(dataset.records) == 2 and all(r.image.path.exists() for r in dataset.records)
    assert {r.rel_image_path.as_posix() for r in dataset.records} == {"sub/a.jpg", "sub/b.jpg"}


def test_coco_images_dir_duplicate_names_and_missing_images_raise(tmp_path: Path) -> None:
    from PIL import Image

    from cvsuite.label.core import router

    ann, pics = coco_elsewhere(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    Image.new("RGB", (20, 10)).save(other / "a.jpg")
    with pytest.raises(ValueError, match="both --images-dir"):
        router.ingest_many([ann], from_hint="coco", images_dir=[pics, other])
    (pics / "sub" / "b.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="b.jpg"):
        router.ingest_many([ann], from_hint="coco", images_dir=[pics])


def test_images_dir_with_a_source_that_is_not_coco_raises() -> None:
    from cvsuite.label.core import router

    with pytest.raises(SystemExit, match="no source of this run is COCO"):
        router.ingest_many([EXAMPLES / "dataset-yolo"], from_hint="yolo", images_dir=[Path(".")])


def test_label_and_eda_clis_accept_images_dir(tmp_path: Path) -> None:
    from cvsuite.label import cli as label_cli

    ann, pics = coco_elsewhere(tmp_path)
    out = tmp_path / "out"
    assert label_cli.main([str(ann), "--from", "coco", "--images-dir", str(pics), "to-yolo", str(out)]) == 0
    assert len(list(out.rglob("*.jpg"))) == 2


def test_emit_box_adds_labeled_boxes_that_avoid_annotations_and_each_other(tmp_path: Path) -> None:
    ds = VisionDataset(records=[dense_record(tmp_path / f"d{i}.jpg") for i in range(3)], classes=["a"], root=tmp_path)
    out = cmd_negatives.run(ds, neg_args(emit="box", label="background", per_image=3, size_from="boxes"))
    assert out.classes == ["a", "background"] and len(out.records) == 3
    assert not hasattr(out, "negatives_root") and "negatives_root" not in out.meta
    for before, after in zip(ds.records, out.records):
        new = [b for b in after.boxes if b.attributes.get("negative")]
        assert len(new) == 3 and all(b.label == "background" and b.cls == 1 for b in new)
        assert len(before.boxes) == 4 and len(after.boxes) == 7
        w, h = after.image.width, after.image.height
        rects = [((b.cx - b.w / 2) * w, (b.cy - b.h / 2) * h, (b.cx + b.w / 2) * w, (b.cy + b.h / 2) * h) for b in after.boxes]
        for i, a in enumerate(rects):
            for b in rects[i + 1:]:
                assert not (a[0] < b[2] - 1e-6 and b[0] < a[2] - 1e-6 and a[1] < b[3] - 1e-6 and b[1] < a[3] - 1e-6)


def test_emit_box_within_annotations_stays_inside_their_extent(tmp_path: Path) -> None:
    path = image(tmp_path / "w.jpg", 200, 160)
    boxes = [BBox(0.25, 0.25, 0.1, 0.1, 0, "a"), BBox(0.75, 0.75, 0.1, 0.1, 0, "a")]
    record = VisionRecord(image=ImageInfo(path, 200, 160), boxes=boxes, rel_image_path=Path("w.jpg"))
    ds = VisionDataset(records=[record], classes=["a"], root=tmp_path)
    out = cmd_negatives.run(ds, neg_args(emit="box", label="bg", per_image=2, imgsz=20, within="annotations"))
    for b in (b for b in out.records[0].boxes if b.attributes.get("negative")):
        assert 0.2 - 0.05 - 1e-6 <= b.cx - b.w / 2 and b.cx + b.w / 2 <= 0.8 + 0.05 + 1e-6
    bare = VisionRecord(image=ImageInfo(path, 200, 160), rel_image_path=Path("w.jpg"))
    with pytest.raises(SystemExit, match="placed 0"):
        cmd_negatives.run(VisionDataset(records=[bare], classes=[], root=tmp_path), neg_args(emit="box", label="bg", within="annotations"))


def test_emit_option_rules(tmp_path: Path) -> None:
    ds = VisionDataset(records=[dense_record(tmp_path / "d.jpg")], classes=["a"], root=tmp_path)
    for kw, message in (({"emit": "box"}, "needs --label"), ({"label": "x"}, "--emit box"), ({"emit": "box", "label": "x", "preview": tmp_path / "p.jpg"}, "--preview")):
        with pytest.raises(SystemExit, match=message):
            cmd_negatives.run(ds, neg_args(**kw))


def test_boxes_from_emit_box_are_cut_by_crop_dets(tmp_path: Path) -> None:
    from cvsuite.label.transform.ops.commands import crop_dets as cmd_crop

    ds = VisionDataset(records=[dense_record(tmp_path / "d.jpg")], classes=["a"], root=tmp_path)
    boxed = cmd_negatives.run(ds, neg_args(emit="box", label="background", per_image=2, imgsz=30))
    cropped = cmd_crop.run(boxed, build_parser(cmd_crop, "x").parse_args(["--imgsz", "32"]))
    labels = sorted((r.classification.label if r.classification else None) or (r.boxes[0].label if r.boxes else "") for r in cropped.records)
    assert labels.count("background") == 2 and labels.count("a") == 4
