import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
import pytest
import yaml

from cvsuite.label.core import router
from cvsuite.common.core.enums import Task
from cvsuite.common.core import ImageRecord, LabelMeShape, Polygon, Record, BBox
from cvsuite.common.core import VisionDataset
import cvsuite.label.output.commands.to_yolo as cmd_to_yolo
import cvsuite.label.output.commands.to_labelme as cmd_to_labelme
import cvsuite.label.output.commands.to_coco as cmd_to_coco
import cvsuite.label.output.commands.inspect as cmd_inspect
import cvsuite.label.output.commands.audit as cmd_to_audit
import cvsuite.label.transform.ops.commands.rebox as cmd_rebox
import cvsuite.label.transform.ops.commands.make_negatives as cmd_make_negatives
import cvsuite.label.transform.ops.commands.crop_dets as cmd_crop_dets
import cvsuite.label.transform.ops.commands.inverse_crop_dets as cmd_inverse_crop_dets
from cvsuite.label.transform.ops.core.inverse_crop_dets import inverse_crop_dets_dataset


def imwrite_rgb(p: Path, w: int = 64, h: int = 48, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).convert("RGB").save(p)


def write_solid_rgb(p: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8), "RGB").save(p)


def add_crop_flags_and_fill(crop_root: Path, flags: dict[str, bool], color: tuple[int, int, int]) -> None:
    jpath = next(crop_root.rglob("*.json"))
    data = json.loads(jpath.read_text(encoding="utf-8"))
    anchors = [
        shape for shape in data["shapes"]
        if shape.get("attributes", {}).get("cvsuite", {}).get("crop_dets")
    ]
    assert len(anchors) == 1
    anchors[0]["flags"] = flags
    image_path = jpath.parent / data["imagePath"]
    with Image.open(image_path) as image:
        Image.new("RGB", image.size, color=color).save(image_path)
    jpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fill_crop_by_id(crop_root: Path, crop_id: str, color: tuple[int, int, int]) -> None:
    hits = []
    for jpath in crop_root.rglob("*.json"):
        data = json.loads(jpath.read_text(encoding="utf-8"))
        anchors = [
            shape for shape in data["shapes"]
            if (
                shape.get("attributes", {})
                .get("cvsuite", {})
                .get("crop_dets", {})
                .get("crop_id")
                == crop_id
            )
        ]
        if anchors:
            hits.append((jpath, data))
    assert len(hits) == 1
    jpath, data = hits[0]
    image_path = jpath.parent / data["imagePath"]
    with Image.open(image_path) as image:
        Image.new("RGB", image.size, color=color).save(image_path)


def make_labelme_project(root: Path) -> Dict[str, Path]:
    images = root / "images"
    lm = root / "labelme"
    images.mkdir(parents=True, exist_ok=True)
    lm.mkdir(parents=True, exist_ok=True)

    # one image for detect (rectangle), one for segment (polygon), one for pose (points)
    img_det = images / "det.jpg"
    imwrite_rgb(img_det, seed=1)
    j_det = lm / "det.json"

    img_seg = images / "seg.jpg"
    imwrite_rgb(img_seg, seed=2)
    j_seg = lm / "seg.json"

    img_pose = images / "pose.jpg"
    imwrite_rgb(img_pose, seed=3)
    j_pose = lm / "pose.json"

    # LabelMe rectangle (stored as 2 pts) for detection
    j_det.write_text(json.dumps({
        "imagePath": img_det.name,
        "imageHeight": 48, "imageWidth": 64,
        "shapes": [{
            "label": "robot",
            "shape_type": "rectangle",
            "points": [[10, 12], [30, 28]],
            "flags": {}
        }]
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # LabelMe polygon for segmentation
    j_seg.write_text(json.dumps({
        "imagePath": img_seg.name,
        "imageHeight": 48, "imageWidth": 64,
        "shapes": [{
            "label": "rust",
            "shape_type": "polygon",
            "points": [[5, 5], [25, 6], [22, 20], [6, 18]],
            "flags": {}
        }]
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # LabelMe points for pose (4 keypoints of a corner-like thing)
    j_pose.write_text(json.dumps({
        "imagePath": img_pose.name,
        "imageHeight": 48, "imageWidth": 64,
        "shapes": [{
            "label": "marker",
            "shape_type": "points",
            "points": [[12, 12], [20, 12], [20, 20], [12, 20]],
            "flags": {}
        }]
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "images": images,
        "labelme": lm,
        "det_img": img_det,
        "seg_img": img_seg,
        "pose_img": img_pose,
    }


def make_yolo_root(root: Path, task: str, split: str = "train") -> Path:
    """Split-first layout: <split>/images and <split>/labels[_task]."""
    images = root / split / "images"
    labels = root / split / ("labels" if task == "det" else f"labels_{task}")
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    return root


def write_yolo_detect_pair(root: Path, split: str, stem: str, box_xywh_norm, cls: int = 0, seed: int = 10):
    img = root / split / "images" / f"{stem}.jpg"
    imwrite_rgb(img, seed=seed)
    lbl = root / split / "labels" / f"{stem}.txt"
    cx, cy, w, h = box_xywh_norm
    lbl.write_text(f"{cls} {cx} {cy} {w} {h}\n", encoding="utf-8")


def write_yolo_segment_pair(root: Path, split: str, stem: str, poly_norm: List[float], cls: int = 0, seed: int = 11):
    img = root / split / "images" / f"{stem}.jpg"
    imwrite_rgb(img, seed=seed)
    lbl = root / split / "labels_segment" / f"{stem}.txt"
    nums = " ".join(map(str, poly_norm))
    lbl.write_text(f"{cls} {nums}\n", encoding="utf-8")


def write_yolo_pose_pair(root: Path, split: str, stem: str, box_xywh_norm, kpts_triplets_norm: List[float], cls: int = 0, seed: int = 12):
    img = root / split / "images" / f"{stem}.jpg"
    imwrite_rgb(img, seed=seed)
    lbl = root / split / "labels_pose" / f"{stem}.txt"
    cx, cy, w, h = box_xywh_norm
    kpts = " ".join(map(lambda x: f"{x}", kpts_triplets_norm))
    lbl.write_text(f"{cls} {cx} {cy} {w} {h} {kpts}\n", encoding="utf-8")


def write_data_yaml(root: Path, task: str, classes: List[str]) -> Path:
    data = {
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": classes,
        "task": task,
    }
    p = root / "data.yaml"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def work(tmp_path: Path) -> Dict[str, Path]:
    return {
        "root": tmp_path,
        "labelme_src": tmp_path / "labelme_src",
        "yolo_from_lm": tmp_path / "yolo_from_lm",
        "yolo_det": tmp_path / "yolo_det",
        "yolo_seg": tmp_path / "yolo_seg",
        "yolo_pose": tmp_path / "yolo_pose",
        "labelme_out": tmp_path / "labelme_out",
        "coco_out": tmp_path / "coco_out",
        "coco_src": tmp_path / "coco_src",
        "rebox_out": tmp_path / "rebox_out",
        "neg_out": tmp_path / "neg_out",
        "inspect_snaps": tmp_path / "inspect_snaps",
        "audit_out": tmp_path / "audit_report.yaml",
    }


def test_label_pipeline_commands_end_to_end(work: Dict[str, Path]):
    # ---------- Build LabelMe project ----------
    lm_paths = make_labelme_project(work["labelme_src"])

    # ---------- labelme2yolo: detect ----------
    ds_lm = router.ingest(
        src=work["labelme_src"] / "labelme",
        from_hint="labelme",
    )
    ns_to = type("A", (), dict(
        dst=work["yolo_from_lm"] / "det",
        task="det",
        splits="",
        hardlink=False,
    ))
    cmd_to_yolo.run(ds_lm, ns_to)
    assert (work["yolo_from_lm"] / "det" / "data.yaml").exists()
    # expect labels under split with .txt
    det_lbls = list((work["yolo_from_lm"] / "det" / "train" / "labels").glob("*.txt")) + \
               list((work["yolo_from_lm"] / "det" / "val" / "labels").glob("*.txt"))
    assert det_lbls, "No detection labels written from LabelMe→YOLO"

    # ---------- labelme2yolo: segment ----------
    ns_to = type("A", (), dict(
        dst=work["yolo_from_lm"] / "seg",
        task="seg",
        splits="",
        hardlink=False,
    ))
    cmd_to_yolo.run(ds_lm, ns_to)
    assert (work["yolo_from_lm"] / "seg" / "data.yaml").exists()
    seg_lbls = list((work["yolo_from_lm"] / "seg" / "train" / "labels").glob("*.txt")) + \
               list((work["yolo_from_lm"] / "seg" / "val" / "labels").glob("*.txt"))
    assert seg_lbls, "No segmentation labels written from LabelMe→YOLO"

    # ---------- labelme2yolo: pose ----------
    ns_to = type("A", (), dict(
        dst=work["yolo_from_lm"] / "pose",
        task="pose",
        splits="",
        hardlink=False,
    ))
    cmd_to_yolo.run(ds_lm, ns_to)
    assert (work["yolo_from_lm"] / "pose" / "data.yaml").exists()
    pose_lbls = list((work["yolo_from_lm"] / "pose" / "train" / "labels").glob("*.txt")) + \
               list((work["yolo_from_lm"] / "pose" / "val" / "labels").glob("*.txt"))
    assert pose_lbls, "No pose labels written from LabelMe→YOLO"

    # ---------- yolo2labelme ----------
    # Build a small YOLO detect dataset in split-first layout and convert to LabelMe
    yolo_det = make_yolo_root(work["yolo_det"], task="det")
    write_yolo_detect_pair(yolo_det, "train", "a", (0.5, 0.5, 0.4, 0.3), 0, 7)
    write_data_yaml(yolo_det, "det", ["robot"])
    ds_yolo = router.ingest(src=yolo_det, from_hint="yolo", task="auto")
    ns = type("A", (), dict(
        dst=work["labelme_out"],
        split="train",
        embed_image=False,
        hardlink=False
    ))
    cmd_to_labelme.run(ds_yolo, ns)
    assert list(work["labelme_out"].rglob("*.json")), "yolo2labelme wrote no JSONs"

    # ---------- yolo2coco ----------
    ns = type("A", (), dict(
        dst=work["coco_out"],
        split="train",
    ))
    cmd_to_coco.run(ds_yolo, ns)
    coco_json = work["coco_out"] / "train" / "_annotations.coco.json"
    assert coco_json.exists(), "yolo2coco did not emit COCO json"
    assert (work["coco_out"] / "data.yaml").exists()

    # ---------- coco2yolo ----------
    coco_src = work["coco_src"]
    coco_src.mkdir(parents=True, exist_ok=True)
    # Minimal COCO detect file consistent with yolo2coco output shape
    coco_min = {
        "images": [{"id": 1, "file_name": "cocoimg.jpg", "width": 64, "height": 48}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 12, 20, 16]}],
        "categories": [{"id": 0, "name": "thing"}],
    }
    (coco_src / "coco.json").write_text(json.dumps(coco_min, indent=2), encoding="utf-8")
    imwrite_rgb(coco_src / "cocoimg.jpg", seed=9)
    yolo_from_coco = work["root"] / "yolo_from_coco"
    ds_coco = router.ingest(src=coco_src / "coco.json", from_hint="coco", task="det")
    cmd_to_yolo.run(ds_coco, type("A", (), dict(dst=yolo_from_coco, task="det", splits="train", hardlink=False)))
    assert list((yolo_from_coco / "train" / "labels").glob("*.txt")), "coco2yolo wrote no labels"

    # ---------- inspect (skip GUI in tests) ----------
    # Not invoked here to avoid GUI requirements.

    # ---------- audit ----------
    ds_det = router.ingest(src=(work["yolo_from_lm"] / "det" / "data.yaml"), from_hint="yolo", task="auto")
    ns = type("A", (), dict(
        dst=work["audit_out"],
        split="train,val",
        thresholds=""
    ))
    cmd_to_audit.run(ds_det, ns)
    assert work["audit_out"].exists(), "audit did not emit report"
    report = yaml.safe_load(work["audit_out"].read_text(encoding="utf-8")) or {}
    assert "overall" in report

    # ---------- rebox (pose) ----------
    # Use pose dataset produced earlier and recompute boxes in memory.
    ds_pose = router.ingest(src=(work["yolo_from_lm"] / "pose" / "data.yaml"), from_hint="yolo", task="auto")
    ns = type("A", (), dict(
        mode="adaptive",
        margin=0.01,
        edge_tol=0.004,
        splits="train,val",
    ))
    ds_reboxed = cmd_rebox.run(ds_pose, ns)
    assert any(rec.boxes for rec in ds_reboxed.records), "rebox did not produce pose boxes"

    # ---------- make_negatives ----------
    ds_neg_src = router.ingest(src=(work["yolo_from_lm"] / "det" / "data.yaml"), from_hint="yolo", task="auto")
    ns = type("A", (), dict(
        splits="train",
        imgsz=32,
        per_image=2,
        min_gap=0.02,
        iou_thresh=0.0,
        seed=123,
        preview=work["neg_out"] / "preview.jpg"
    ))
    ds_neg = cmd_make_negatives.run(ds_neg_src, ns)
    negs = [rec.image.path for rec in ds_neg.records if rec.attributes.get("negative")]
    assert negs, "make_negatives wrote no crops"
    assert all(path.exists() for path in negs), "make_negatives returned missing crop paths"
    assert (work["neg_out"] / "preview.jpg").exists(), "make_negatives preview sheet missing"


def test_yolo_segmentation_in_plain_labels_dir(tmp_path: Path):
    root = make_yolo_root(tmp_path / "yolo_seg_plain", task="det")
    img = root / "train" / "images" / "plain.jpg"
    imwrite_rgb(img, seed=42)
    lbl = root / "train" / "labels" / "plain.txt"
    poly = [0.1, 0.1, 0.3, 0.1, 0.3, 0.25, 0.12, 0.28]
    lbl.write_text(f"0 {' '.join(map(str, poly))}\n", encoding="utf-8")
    data = {
        "path": str(root),
        "train": "train/images",
        "names": ["fire"],
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    ds = router.ingest(src=root / "data.yaml", from_hint="yolo", task="auto")
    assert ds.task == Task.seg
    assert ds.records and ds.records[0].polys, "segmentation polygons were not parsed from labels/"


def test_coco_data_yaml_ingest(tmp_path: Path):
    root = tmp_path / "coco_from_yaml"
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    img = images / "img.jpg"
    imwrite_rgb(img, seed=5)
    ann_path = root / "annotations_train.json"
    coco = {
        "images": [{"id": 1, "file_name": img.name, "width": 64, "height": 48}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
            "bbox": [0, 0, 10, 10],
            "iscrowd": 0,
        }],
        "categories": [{"id": 0, "name": "fire"}],
    }
    ann_path.write_text(json.dumps(coco), encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": root.name,
        "format": "coco",
        "task": "seg",
        "names": ["fire"],
        "train": ann_path.name,
    }, sort_keys=False), encoding="utf-8")

    ds = router.ingest(src=data_yaml, from_hint="auto", task="auto")
    assert ds.task == Task.seg
    assert ds.records and ds.records[0].polys, "COCO polygons were not ingested from data.yaml"
    assert ds.root == root


def test_labelme_write_preserves_existing_shapes_and_adds_new_polys(tmp_path: Path):
    lm_paths = make_labelme_project(tmp_path / "lm_src")
    ds = router.ingest(src=lm_paths["labelme"], from_hint="labelme")
    poly = Polygon(points=[(0.1, 0.1), (0.3, 0.1), (0.3, 0.3)], cls=0, label="robot")
    ds.records[0].polys.append(poly)
    target_stem = ds.records[0].image.path.stem

    out = tmp_path / "lm_out"
    ns = type("A", (), dict(
        dst=out,
        split="",
        embed_image=False,
        hardlink=False,
    ))
    cmd_to_labelme.run(ds, ns)

    jpath = next(p for p in out.rglob("*.json") if p.stem == target_stem)
    data = json.loads(jpath.read_text(encoding="utf-8"))
    shape_types = [s["shape_type"] for s in data.get("shapes", [])]
    assert "rectangle" in shape_types, "Original shapes missing after write"
    assert "polygon" in shape_types, "New polygons were not written alongside existing shapes"


def test_crop_dets_rewrites_boxes_and_clips_polygons(tmp_path: Path):
    img = tmp_path / "scene.jpg"
    imwrite_rgb(img, w=100, h=80, seed=101)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=100, height=80),
                split="val",
                task=Task.seg,
                boxes=[BBox(cx=0.5, cy=0.5, w=0.4, h=0.5, cls=0, label="thing", score=0.93)],
                polys=[
                    Polygon(
                        points=[(0.2, 0.25), (0.75, 0.25), (0.75, 0.75), (0.2, 0.75)],
                        cls=0,
                        label="thing",
                    )
                ],
            )
        ],
        classes=["thing"],
        task=Task.seg,
        root=tmp_path,
    )

    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=None, long_size=None, pad_type="black", margin_frac=0.0)),
    )

    assert cropped.meta["transform"] == "crop-dets"
    assert cropped.meta["crop_count"] == 1
    assert len(cropped.records) == 1

    rec = cropped.records[0]
    assert rec.split == "val"
    assert rec.image.path.exists()
    assert rec.image.width == 40
    assert rec.image.height == 40
    assert rec.classification is not None
    assert rec.classification.label == "thing"
    assert rec.classification.score == pytest.approx(0.93)
    assert "crop_source" not in rec.attributes
    assert rec.boxes and rec.boxes[0].cx == pytest.approx(0.5)
    assert rec.boxes[0].cy == pytest.approx(0.5)
    assert rec.boxes[0].w == pytest.approx(1.0)
    assert rec.boxes[0].h == pytest.approx(1.0)
    assert rec.polys, "Expected polygon to be preserved on the crop"
    xs = [pt[0] for pt in rec.polys[0].points]
    ys = [pt[1] for pt in rec.polys[0].points]
    assert min(xs) == pytest.approx(0.0)
    assert max(xs) == pytest.approx(1.0)
    assert min(ys) == pytest.approx(0.0)
    assert max(ys) == pytest.approx(1.0)
    assert rec.labelme_shapes == []
    assert rec.vqas == []
    assert rec.embeddings == []


def test_crop_dets_labelme_roundtrip_preserves_reversible_metadata(tmp_path: Path):
    img = tmp_path / "scene.jpg"
    imwrite_rgb(img, w=100, h=80, seed=111)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=100, height=80),
                split="train",
                task=Task.det,
                boxes=[BBox(cx=0.5, cy=0.5, w=0.4, h=0.5, cls=0, label="thing")],
            )
        ],
        classes=["thing"],
        task=Task.det,
        root=tmp_path,
    )

    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=64, long_size=None, pad_type="black", margin_frac=0.1)),
    )

    meta = cropped.records[0].boxes[0].attributes["cvsuite"]["crop_dets"]
    assert meta == {"crop_id": "train:scene.jpg:box:0", "margin_frac": 0.1}

    out = tmp_path / "crop_labelme"
    cmd_to_labelme.run(
        cropped,
        type("Args", (), dict(dst=out, val_frac=0.0, test_frac=0.0, seed=0, embed_image=False, hardlink=False)),
    )
    assert not (out / "train").exists()
    payload = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    assert "crop_source" not in payload.get("attributes", {})
    reloaded = router.ingest(out, from_hint="labelme")
    restored = reloaded.records[0].boxes[0].attributes["cvsuite"]["crop_dets"]
    assert restored == meta


def test_inverse_crop_dets_pastes_crops_flags_and_preserves_unchanged_records(tmp_path: Path):
    base = np.zeros((30, 30, 3), dtype=np.uint8)
    base[:] = (10, 20, 30)
    base[10:20, 10:20] = (80, 90, 100)
    img = tmp_path / "scene.png"
    write_solid_rgb(img, base)
    other = tmp_path / "other.png"
    write_solid_rgb(other, np.full((12, 12, 3), 150, dtype=np.uint8))
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=30, height=30),
                split="train",
                task=Task.seg,
                boxes=[
                    BBox(
                        cx=0.5,
                        cy=0.5,
                        w=10 / 30,
                        h=10 / 30,
                        cls=0,
                        label="thing",
                        group_id=7,
                        attributes={"flags": {"with-corrosion": False}},
                    )
                ],
                polys=[
                    Polygon(
                        points=[(10 / 30, 10 / 30), (20 / 30, 10 / 30), (20 / 30, 20 / 30), (10 / 30, 20 / 30)],
                        cls=0,
                        label="thing",
                        group_id=7,
                    )
                ],
            ),
            Record(image=ImageRecord(path=other, width=12, height=12), split="train", task=Task.det),
        ],
        classes=["thing"],
        task=Task.seg,
        root=tmp_path,
    )
    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=None, long_size=None, pad_type="black", margin_frac=0.0)),
    )
    crop_root = tmp_path / "modified_crops"
    cmd_to_labelme.run(
        cropped,
        type("Args", (), dict(dst=crop_root, val_frac=0.0, test_frac=0.0, seed=0, embed_image=False, hardlink=False)),
    )
    assert not (crop_root / "train").exists()
    add_crop_flags_and_fill(crop_root, {"with-corrosion": True, "object-blue": False}, (1, 200, 3))

    inverted = cmd_inverse_crop_dets.run(ds, type("Args", (), dict(modified_crops=crop_root)))

    changed = inverted.records[0]
    with Image.open(changed.image.path) as image:
        arr = np.asarray(image.convert("RGB"))
    assert np.allclose(arr[15, 15], np.array([1, 200, 3]), atol=2)
    assert tuple(arr[5, 5]) == (10, 20, 30)
    assert changed.boxes[0].attributes["flags"] == {"with-corrosion": True, "object-blue": False}
    assert changed.polys[0].attributes["flags"] == {"with-corrosion": True, "object-blue": False}
    assert changed.attributes["cvsuite"]["inverse_crop_dets"]["applied"][0]["crop_id"] == "train:scene.png:box:0"
    assert inverted.records[1].image.path == other.resolve()


def test_inverse_crop_dets_pastes_smaller_overlapping_crops_last(tmp_path: Path):
    img = tmp_path / "scene.png"
    write_solid_rgb(img, np.full((30, 30, 3), 10, dtype=np.uint8))
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=30, height=30),
                split="train",
                task=Task.det,
                boxes=[
                    BBox(cx=15 / 30, cy=15 / 30, w=10 / 30, h=10 / 30, cls=0, label="thing"),
                    BBox(cx=15 / 30, cy=15 / 30, w=20 / 30, h=20 / 30, cls=0, label="thing"),
                ],
            )
        ],
        classes=["thing"],
        task=Task.det,
        root=tmp_path,
    )
    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=None, long_size=None, pad_type="black", margin_frac=0.0)),
    )
    crop_root = tmp_path / "modified_overlap_crops"
    cmd_to_labelme.run(
        cropped,
        type("Args", (), dict(dst=crop_root, val_frac=0.0, test_frac=0.0, seed=0, embed_image=False, hardlink=False)),
    )
    fill_crop_by_id(crop_root, "train:scene.png:box:0", (200, 1, 3))
    fill_crop_by_id(crop_root, "train:scene.png:box:1", (2, 180, 4))

    inverted = cmd_inverse_crop_dets.run(ds, type("Args", (), dict(modified_crops=crop_root)))

    with Image.open(inverted.records[0].image.path) as image:
        arr = np.asarray(image.convert("RGB"))
    assert np.allclose(arr[7, 7], np.array([2, 180, 4]), atol=2)
    assert np.allclose(arr[15, 15], np.array([200, 1, 3]), atol=2)
    assert [
        item["crop_id"]
        for item in inverted.records[0].attributes["cvsuite"]["inverse_crop_dets"]["applied"]
    ] == ["train:scene.png:box:1", "train:scene.png:box:0"]


@pytest.mark.parametrize(
    "args",
    [
        dict(imgsz=None, long_size=18, pad_type="black", margin_frac=0.0),
        dict(imgsz=24, long_size=None, pad_type="gray", margin_frac=0.0),
        dict(imgsz=24, long_size=None, pad_type="background", margin_frac=0.0),
    ],
)
def test_inverse_crop_dets_pastes_resized_crop_modes(tmp_path: Path, args: dict[str, object]):
    base = np.zeros((30, 40, 3), dtype=np.uint8)
    base[:] = (12, 24, 36)
    base[12:18, 15:25] = (70, 80, 90)
    img = tmp_path / "scene.png"
    write_solid_rgb(img, base)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=40, height=30),
                split="train",
                task=Task.det,
                boxes=[BBox(cx=20 / 40, cy=15 / 30, w=10 / 40, h=6 / 30, cls=0, label="thing")],
            )
        ],
        classes=["thing"],
        task=Task.det,
        root=tmp_path,
    )
    cropped = cmd_crop_dets.run(ds, type("Args", (), args))
    meta = cropped.records[0].boxes[0].attributes["cvsuite"]["crop_dets"]
    expected_meta = {"crop_id": "train:scene.png:box:0", "margin_frac": 0.0}
    if args["pad_type"] == "background":
        expected_meta["pad_type"] = "background"
    assert meta == expected_meta
    crop_root = tmp_path / f"modified_{args['pad_type']}_{args['long_size'] or args['imgsz']}"
    cmd_to_labelme.run(
        cropped,
        type("Args", (), dict(dst=crop_root, val_frac=0.0, test_frac=0.0, seed=0, embed_image=False, hardlink=False)),
    )
    assert not (crop_root / "train").exists()
    add_crop_flags_and_fill(crop_root, {}, (2, 180, 4))

    inverted = cmd_inverse_crop_dets.run(ds, type("Args", (), dict(modified_crops=crop_root)))

    with Image.open(inverted.records[0].image.path) as image:
        arr = np.asarray(image.convert("RGB"))
    assert np.allclose(arr[15, 20], np.array([2, 180, 4]), atol=2)


def test_inverse_crop_dets_rejects_duplicate_crop_ids(tmp_path: Path):
    img = tmp_path / "scene.png"
    write_solid_rgb(img, np.zeros((20, 20, 3), dtype=np.uint8))
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=20, height=20),
                split="train",
                task=Task.det,
                boxes=[BBox(cx=0.5, cy=0.5, w=0.5, h=0.5, cls=0, label="thing")],
            )
        ],
        classes=["thing"],
        task=Task.det,
        root=tmp_path,
    )
    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=None, long_size=None, pad_type="black", margin_frac=0.0)),
    )
    duplicated = VisionDataset(
        records=[cropped.records[0], deepcopy(cropped.records[0])],
        classes=cropped.classes,
        task=cropped.task,
        root=cropped.root,
    )

    with pytest.raises(ValueError, match="Duplicate crop_id"):
        inverse_crop_dets_dataset(ds, duplicated)


def test_crop_dets_synthesizes_anchor_from_polygon_only(tmp_path: Path):
    img = tmp_path / "poly_only.jpg"
    imwrite_rgb(img, w=120, h=60, seed=202)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=120, height=60),
                split="train",
                task=Task.seg,
                polys=[
                    Polygon(
                        points=[(0.2, 0.2), (0.6, 0.2), (0.6, 0.8), (0.2, 0.8)],
                        cls=0,
                        label="rust",
                        score=0.71,
                    )
                ],
            )
        ],
        classes=["rust"],
        task=Task.seg,
        root=tmp_path,
    )

    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=None, long_size=64, pad_type="black", margin_frac=0.0)),
    )

    assert len(cropped.records) == 1
    rec = cropped.records[0]
    assert rec.image.width == 64
    assert rec.image.height == 48
    assert "crop_source" not in rec.attributes
    assert rec.boxes[0].attributes["cvsuite"]["crop_dets"] == {
        "crop_id": "train:poly_only.jpg:polygon:0",
        "margin_frac": 0.0,
    }
    assert rec.boxes and rec.boxes[0].w == pytest.approx(1.0)
    assert rec.polys, "Polygon-only inputs should still keep segmentation on the crop"
    assert rec.classification is not None
    assert rec.classification.label == "rust"
    assert rec.classification.score == pytest.approx(0.71)
    assert cropped.meta["crop_long_size"] == 64


def test_crop_dets_imgsz_black_padding_rewrites_box_to_square_canvas(tmp_path: Path):
    img = tmp_path / "padded.jpg"
    imwrite_rgb(img, w=100, h=100, seed=212)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=100, height=100),
                split="train",
                task=Task.det,
                boxes=[BBox(cx=0.5, cy=0.5, w=0.4, h=0.2, cls=0, label="thing")],
            )
        ],
        classes=["thing"],
        task=Task.det,
        root=tmp_path,
    )

    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=64, long_size=None, pad_type="black", margin_frac=0.0)),
    )

    rec = cropped.records[0]
    assert rec.image.width == 64
    assert rec.image.height == 64
    assert rec.boxes and rec.boxes[0].w == pytest.approx(1.0)
    assert rec.boxes[0].h == pytest.approx(0.5)
    assert cropped.meta["crop_pad_type"] == "black"


def test_crop_dets_gray_padding_fills_canvas(tmp_path: Path):
    img = tmp_path / "gray_pad.jpg"
    imwrite_rgb(img, w=80, h=80, seed=218)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=80, height=80),
                split="train",
                task=Task.det,
                boxes=[BBox(cx=0.5, cy=0.5, w=0.5, h=0.2, cls=0, label="thing")],
            )
        ],
        classes=["thing"],
        task=Task.det,
        root=tmp_path,
    )

    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=64, long_size=None, pad_type="gray", margin_frac=0.0)),
    )

    with Image.open(cropped.records[0].image.path) as crop_img:
        pixel = crop_img.convert("RGB").getpixel((0, 0))
    assert pixel == (127, 127, 127)


def test_crop_dets_background_padding_uses_surrounding_image_context(tmp_path: Path):
    img = tmp_path / "background_pad.png"
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :40] = (255, 0, 0)
    arr[:, 40:60] = (0, 255, 0)
    arr[:, 60:] = (0, 0, 255)
    Image.fromarray(arr, "RGB").save(img)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=100, height=100),
                split="train",
                task=Task.det,
                boxes=[BBox(cx=0.5, cy=0.5, w=0.2, h=0.6, cls=0, label="thing")],
            )
        ],
        classes=["thing"],
        task=Task.det,
        root=tmp_path,
    )

    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=60, long_size=None, pad_type="background", margin_frac=0.0)),
    )

    rec = cropped.records[0]
    assert rec.boxes and rec.boxes[0].w == pytest.approx(20.0 / 60.0)
    assert rec.boxes[0].h == pytest.approx(1.0)
    assert rec.boxes[0].attributes["cvsuite"]["crop_dets"] == {
        "crop_id": "train:background_pad.png:box:0",
        "margin_frac": 0.0,
        "pad_type": "background",
    }
    with Image.open(rec.image.path) as crop_img:
        rgb = np.asarray(crop_img.convert("RGB"))

    left = rgb[30, 5]
    center = rgb[30, 30]
    right = rgb[30, 55]
    assert left[0] > 200 and left[1] < 60 and left[2] < 60
    assert center[1] > 200 and center[0] < 60 and center[2] < 60
    assert right[2] > 200 and right[0] < 60 and right[1] < 60


def test_crop_dets_background_padding_uses_black_when_square_overflows_image(tmp_path: Path):
    img = tmp_path / "background_edge.png"
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:] = (220, 40, 40)
    Image.fromarray(arr, "RGB").save(img)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=100, height=100),
                split="train",
                task=Task.det,
                boxes=[BBox(cx=0.9, cy=0.5, w=0.2, h=0.6, cls=0, label="thing")],
            )
        ],
        classes=["thing"],
        task=Task.det,
        root=tmp_path,
    )

    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=60, long_size=None, pad_type="background", margin_frac=0.0)),
    )

    rec = cropped.records[0]
    assert rec.boxes and rec.boxes[0].w == pytest.approx(20.0 / 60.0)
    assert rec.boxes[0].h == pytest.approx(1.0)
    with Image.open(rec.image.path) as crop_img:
        rgb = np.asarray(crop_img.convert("RGB"))

    left = rgb[30, 5]
    right = rgb[30, 55]
    assert left[0] > 180 and left[1] < 80 and left[2] < 80
    assert right.max() < 25


def test_crop_dets_margin_frac_expands_crop_and_preserves_relative_box(tmp_path: Path):
    img = tmp_path / "margin_scene.jpg"
    imwrite_rgb(img, w=100, h=100, seed=717)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=100, height=100),
                split="train",
                task=Task.det,
                boxes=[BBox(cx=0.5, cy=0.5, w=0.2, h=0.2, cls=0, label="thing")],
            )
        ],
        classes=["thing"],
        task=Task.det,
        root=tmp_path,
    )

    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=None, long_size=None, pad_type="black", margin_frac=0.25)),
    )

    rec = cropped.records[0]
    assert rec.image.width == 30
    assert rec.image.height == 30
    assert rec.boxes and rec.boxes[0].w == pytest.approx(20.0 / 30.0)
    assert rec.boxes[0].h == pytest.approx(20.0 / 30.0)
    assert cropped.meta["crop_margin_frac"] == pytest.approx(0.25)


def test_crop_dets_to_yolo_det_splits_one_source_into_many_records(tmp_path: Path):
    img = tmp_path / "multi.jpg"
    imwrite_rgb(img, w=100, h=100, seed=303)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=100, height=100),
                split="train",
                task=Task.det,
                boxes=[
                    BBox(cx=0.25, cy=0.35, w=0.2, h=0.2, cls=0, label="robot"),
                    BBox(cx=0.7, cy=0.65, w=0.2, h=0.2, cls=1, label="valve"),
                ],
            )
        ],
        classes=["robot", "valve"],
        task=Task.det,
        root=tmp_path,
    )

    dst = tmp_path / "yolo_det"
    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=None, long_size=None, pad_type="black", margin_frac=0.0)),
    )
    assert len(cropped.records) == 2
    assert len({rec.image.path.name for rec in cropped.records}) == 2

    args = type(
        "Args",
        (),
        dict(dst=dst, task="det", val_frac=0.0, test_frac=0.0, seed=0, hardlink=False),
    )
    cmd_to_yolo.run(cropped, args)

    labels = sorted((dst / "train" / "labels").glob("*.txt"))
    images = sorted((dst / "train" / "images").glob("*.jpg"))
    assert len(labels) == 2
    assert len(images) == 2
    for path in labels:
        parts = path.read_text(encoding="utf-8").strip().split()
        assert len(parts) == 5
        assert parts[1:] == ["0.500000", "0.500000", "1.000000", "1.000000"]


def test_crop_dets_to_yolo_seg_preserves_polygons(tmp_path: Path):
    img = tmp_path / "seg.jpg"
    imwrite_rgb(img, w=100, h=80, seed=404)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=100, height=80),
                split="train",
                task=Task.seg,
                boxes=[BBox(cx=0.5, cy=0.5, w=0.4, h=0.5, cls=0, label="rust")],
                polys=[Polygon(points=[(0.2, 0.25), (0.75, 0.25), (0.75, 0.75), (0.2, 0.75)], cls=0, label="rust")],
            )
        ],
        classes=["rust"],
        task=Task.seg,
        root=tmp_path,
    )

    dst = tmp_path / "yolo_seg"
    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=None, long_size=None, pad_type="black", margin_frac=0.0)),
    )
    args = type(
        "Args",
        (),
        dict(dst=dst, task="seg", val_frac=0.0, test_frac=0.0, seed=0, hardlink=False),
    )
    cmd_to_yolo.run(cropped, args)

    label_path = next((dst / "train" / "labels").glob("*.txt"))
    parts = label_path.read_text(encoding="utf-8").strip().split()
    coords = [float(value) for value in parts[1:]]
    assert len(coords) >= 6
    assert all(0.0 <= value <= 1.0 for value in coords)
    assert any(value == pytest.approx(0.0) for value in coords)
    assert any(value == pytest.approx(1.0) for value in coords)


def test_crop_dets_to_labelme_clears_stale_source_shapes(tmp_path: Path):
    img = tmp_path / "labelme_source.jpg"
    imwrite_rgb(img, w=100, h=80, seed=505)
    ds = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=img, width=100, height=80),
                split="train",
                task=Task.seg,
                boxes=[BBox(cx=0.5, cy=0.5, w=0.4, h=0.5, cls=0, label="thing")],
                polys=[Polygon(points=[(0.25, 0.25), (0.7, 0.25), (0.7, 0.75), (0.25, 0.75)], cls=0, label="thing")],
                labelme_shapes=[
                    LabelMeShape(label="thing", shape_type="rectangle", points=[(10, 10), (40, 30)]),
                    LabelMeShape(label="other", shape_type="rectangle", points=[(70, 10), (90, 20)]),
                ],
            )
        ],
        classes=["thing"],
        task=Task.seg,
        root=tmp_path,
    )

    out = tmp_path / "labelme_out"
    cropped = cmd_crop_dets.run(
        ds,
        type("Args", (), dict(imgsz=None, long_size=None, pad_type="black", margin_frac=0.0)),
    )
    cmd_to_labelme.run(
        cropped,
        type("Args", (), dict(dst=out, val_frac=0.0, test_frac=0.0, seed=0, embed_image=False, hardlink=False)),
    )

    assert not (out / "train").exists()
    data = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    labels = [shape["label"] for shape in data["shapes"]]
    shape_types = [shape["shape_type"] for shape in data["shapes"]]
    assert labels == ["thing", "thing"]
    assert shape_types == ["rectangle", "polygon"]
