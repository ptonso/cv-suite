"""`select` and `remap-classes` (`specs/common/select_remap.md`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from cvsuite.branch_cli import build_parser
from cvsuite.common.core import BBox, Classification, ImageInfo, Keypoints, LabelMeShape, Polygon, SemanticMaskRef, VisionDataset, VisionRecord
from cvsuite.common.io import read_dataset, write_dataset
from cvsuite.common.transforms import remap as remap_mod
from cvsuite.common.transforms import select as select_mod
from cvsuite.label import cli as label_cli

SQUARE = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
TINY = [(0.5, 0.5), (0.52, 0.5), (0.52, 0.52)]


def run_select(ds: VisionDataset, *argv: str) -> VisionDataset:
    return select_mod.run(ds, build_parser(select_mod, "select").parse_args(list(argv)))


def run_remap(ds: VisionDataset, mapping: dict, *argv: str, tmp: Path) -> VisionDataset:
    path = tmp / "map.yaml"
    path.write_text(yaml.safe_dump(mapping))
    return remap_mod.run(ds, build_parser(remap_mod, "remap").parse_args(["--map", str(path), *argv]))


def det_record(name: str, boxes: list[tuple[float, int]], size: int = 100) -> VisionRecord:
    return VisionRecord(
        image=ImageInfo(Path(name), size, size),
        rel_image_path=Path(name),
        boxes=[BBox(0.5, 0.5, side, side, cls, ["a", "b", "c"][cls]) for side, cls in boxes],
    )


def det_dataset() -> VisionDataset:
    return VisionDataset(
        records=[
            det_record("big.jpg", [(0.5, 0), (0.5, 1)]),
            det_record("mixed.jpg", [(0.5, 2), (0.05, 0)]),
            det_record("small.jpg", [(0.05, 1)]),
            det_record("empty.jpg", []),
        ],
        classes=["a", "b", "c"],
        root=Path("/x"),
    )


def names(ds: VisionDataset) -> list[str]:
    return [r.rel_image_path.name for r in ds.records]


def test_min_box_px_removes_small_annotations_and_requires_empty() -> None:
    with pytest.raises(SystemExit, match="--empty"):
        run_select(det_dataset(), "--min-box-px", "10")
    kept = run_select(det_dataset(), "--min-box-px", "10", "--empty", "keep")
    assert names(kept) == ["big.jpg", "mixed.jpg", "small.jpg", "empty.jpg"]
    assert [len(r.boxes) for r in kept.records] == [2, 1, 0, 0]
    dropped = run_select(det_dataset(), "--min-box-px", "10", "--empty", "drop")
    assert names(dropped) == ["big.jpg", "mixed.jpg", "empty.jpg"]
    info = dropped.meta["step_info"]
    assert info["annotations_removed"] == {"min_box_px": 2} and info["records_dropped_by_size"] == 1


def test_side_and_area_flags() -> None:
    ds = run_select(det_dataset(), "--min-side-px", "10", "--empty", "keep")
    assert [len(r.boxes) for r in ds.records] == [2, 1, 0, 0]
    ds = run_select(det_dataset(), "--min-area-frac", "0.01", "--empty", "keep")
    assert [len(r.boxes) for r in ds.records] == [2, 1, 0, 0]


def test_polygons_and_their_boxes_leave_together() -> None:
    def seg(points, cls=0) -> tuple[Polygon, BBox]:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        return Polygon(points, cls, "a"), BBox(min(xs) + w / 2, min(ys) + h / 2, w, h, cls, "a", kind="seg")

    (p1, b1), (p2, b2) = seg(SQUARE), seg(TINY)
    record = VisionRecord(image=ImageInfo(Path("a.jpg"), 100, 100), rel_image_path=Path("a.jpg"), polys=[p1, p2], boxes=[b1, b2])
    out = run_select(VisionDataset(records=[record], classes=["a"]), "--min-box-px", "10", "--empty", "drop")
    assert len(out.records[0].polys) == 1 and len(out.records[0].boxes) == 1
    assert out.records[0].polys[0].points == SQUARE


def test_pose_box_removal_removes_its_keypoints() -> None:
    def pose(side, x):
        return BBox(x, 0.5, side, side, 0, "a", kind="pose"), Keypoints([(x, 0.5, 2.0)], 0, "a")

    (b1, k1), (b2, k2) = pose(0.5, 0.3), pose(0.02, 0.8)
    record = VisionRecord(image=ImageInfo(Path("a.jpg"), 100, 100), rel_image_path=Path("a.jpg"), boxes=[b1, b2], kpts=[k1, k2])
    out = run_select(VisionDataset(records=[record], classes=["a"]), "--min-box-px", "10", "--empty", "keep")
    assert len(out.records[0].boxes) == 1 and out.records[0].kpts == [k1]


def test_keep_and_drop_class_renumber_in_original_order() -> None:
    out = run_select(det_dataset(), "--drop-class", "a", "--empty", "keep")
    assert out.classes == ["b", "c"]
    assert [[(b.cls, b.label) for b in r.boxes] for r in out.records] == [[(0, "b")], [(1, "c")], [(0, "b")], []]
    kept = run_select(det_dataset(), "--keep-class", "c", "--empty", "drop")
    assert kept.classes == ["c"] and names(kept) == ["mixed.jpg", "empty.jpg"]
    assert [b.cls for b in kept.records[0].boxes] == [0]


def test_exclude_name_matches_the_layout_path() -> None:
    out = run_select(det_dataset(), "--exclude-name", "^(big|small)", "--exclude-name", "empty")
    assert names(out) == ["mixed.jpg"]
    assert out.meta["step_info"]["records_excluded_by_name"] == 3


def test_select_error_cases() -> None:
    with pytest.raises(ValueError, match="does not have: zzz"):
        run_select(det_dataset(), "--drop-class", "zzz", "--empty", "keep")
    with pytest.raises(SystemExit):
        build_parser(select_mod, "s").parse_args(["--keep-class", "a", "--drop-class", "b"])
    with pytest.raises(SystemExit, match="no records"):
        run_select(det_dataset(), "--exclude-name", ".")
    with pytest.raises(SystemExit, match="at least one"):
        run_select(det_dataset())
    with pytest.raises(ValueError, match="boxes or polygons"):
        run_select(VisionDataset(records=[VisionRecord(image=ImageInfo(Path("a.jpg"), 8, 8), rel_image_path=Path("a.jpg"))], classes=[]), "--min-box-px", "3")
    masked = det_dataset()
    masked.records[0].semantic_mask = SemanticMaskRef(Path("m.png"))
    with pytest.raises(ValueError, match="semantic masks"):
        run_select(masked, "--drop-class", "a", "--empty", "keep")


def test_class_branch_select_has_no_pixel_flags() -> None:
    from cvsuite.classify.transform.commands import select as class_select

    parser = build_parser(class_select, "select")
    with pytest.raises(SystemExit):
        parser.parse_args(["--min-box-px", "3"])
    assert parser.parse_args(["--drop-class", "a"]).drop_class == "a"


def cls_dataset() -> VisionDataset:
    def rec(name, label, probs):
        return VisionRecord(image=ImageInfo(Path(name), 8, 8), rel_image_path=Path(name), classification=Classification(label=label, score=probs[label], probs=probs))

    return VisionDataset(
        records=[rec("1.jpg", "cat", {"cat": 0.6, "lynx": 0.3, "dog": 0.1}), rec("2.jpg", "dog", {"cat": 0.1, "lynx": 0.1, "dog": 0.8}), rec("3.jpg", "lynx", {"cat": 0.2, "lynx": 0.7, "dog": 0.1})],
        classes=["cat", "lynx", "dog"],
    )


def test_class_merge_sums_probs_and_follows_labels(tmp_path: Path) -> None:
    out = run_remap(cls_dataset(), {"merge": {"feline": ["cat", "lynx"]}}, tmp=tmp_path)
    assert out.classes == ["feline", "dog"]
    first = out.records[0].classification
    assert first.label == "feline" and first.probs == pytest.approx({"feline": 0.9, "dog": 0.1}) and first.score == pytest.approx(0.9)
    assert out.records[1].classification.label == "dog"


def test_class_select_drops_labels_and_records() -> None:
    out = run_select(cls_dataset(), "--drop-class", "dog")
    assert out.classes == ["cat", "lynx"] and names(out) == ["1.jpg", "2.jpg", "3.jpg"]
    assert all("dog" not in r.classification.probs for r in out.records)
    assert out.records[1].classification.label in {"cat", "lynx"}, "a dropped winner falls to the best remaining class"

    plain = VisionDataset(
        records=[VisionRecord(image=ImageInfo(Path(f"{n}.jpg"), 8, 8), rel_image_path=Path(f"{n}.jpg"), classification=Classification(label=n)) for n in ("cat", "dog")],
        classes=["cat", "dog"],
    )
    assert names(run_select(plain, "--drop-class", "dog")) == ["cat.jpg"]


def test_multi_label_attributes_are_mapped_and_deduplicated(tmp_path: Path) -> None:
    record = VisionRecord(
        image=ImageInfo(Path("a.jpg"), 8, 8), rel_image_path=Path("a.jpg"),
        classification=Classification(label="cat", score=1.0, probs={"cat": 1.0, "lynx": 1.0}),
        attributes={"class_dir_multi_labels": ["cat", "lynx", "dog"], "classify_multi_class_labels": ["cat", "lynx", "dog"]},
    )
    out = run_remap(VisionDataset(records=[record], classes=["cat", "lynx", "dog"]), {"merge": {"feline": ["cat", "lynx"]}, "drop": ["dog"]}, tmp=tmp_path)
    assert out.records[0].attributes["class_dir_multi_labels"] == ["feline"]
    assert out.records[0].attributes["classify_multi_class_labels"] == ["feline"]


def test_remap_merge_rename_drop_others_order(tmp_path: Path) -> None:
    ds = VisionDataset(
        records=[det_record("a.jpg", [(0.5, 0), (0.5, 1), (0.5, 2)])],
        classes=["a", "b", "c"],
    )
    out = run_remap(ds, {"merge": {"ab": ["a", "b"]}, "rename": {"c": "see"}}, tmp=tmp_path)
    assert out.classes == ["ab", "see"]
    assert [(b.cls, b.label) for b in out.records[0].boxes] == [(0, "ab"), (0, "ab"), (1, "see")]

    out = run_remap(ds, {"rename": {"c": "see"}, "drop": ["a"]}, "--empty", "keep", tmp=tmp_path)
    assert out.classes == ["see", "b"]

    out = run_remap(ds, {"merge": {"ab": ["a", "b"]}}, "--others", "other", tmp=tmp_path)
    assert out.classes == ["ab", "other"] and out.records[0].boxes[2].label == "other"

    out = run_remap(ds, {"merge": {"ab": ["a", "b", "c"]}}, "--others", "other", tmp=tmp_path)
    assert out.classes == ["ab"]


@pytest.mark.parametrize(
    "mapping, argv, message",
    [
        ({"merge": {"x": ["a"]}, "rename": {"a": "y"}}, [], "more than once"),
        ({"merge": {"x": ["a", "a"]}}, [], "more than once"),
        ({"rename": {"zzz": "y"}}, [], "does not have: zzz"),
        ({"rename": {"a": "b"}}, [], "also an existing class"),
        ({"merge": {"b": ["a", "c"]}}, [], "also an existing class"),
        ({"merge": {"x": ["a"]}, "rename": {"b": "x"}}, [], "twice"),
        ({"merge": {"x": ["a"]}}, ["--others", "x"], "also created"),
        ({"bogus": {}}, [], "only these sections"),
        ({"drop": ["a"]}, [], "--empty"),
        ({"merge": {"x": "a"}}, [], "must list"),
    ],
)
def test_remap_error_cases(tmp_path: Path, mapping: dict, argv: list[str], message: str) -> None:
    ds = VisionDataset(records=[det_record("a.jpg", [(0.5, 0), (0.5, 1), (0.5, 2)])], classes=["a", "b", "c"])
    with pytest.raises((ValueError, SystemExit), match=message):
        run_remap(ds, mapping, *argv, tmp=tmp_path)


def test_a_merge_may_reuse_its_own_source_name(tmp_path: Path) -> None:
    ds = VisionDataset(records=[det_record("a.jpg", [(0.5, 0), (0.5, 1)])], classes=["a", "b", "c"])
    out = run_remap(ds, {"merge": {"a": ["a", "b"]}}, tmp=tmp_path)
    assert out.classes == ["a", "c"]


def test_remap_rejects_semantic_masks(tmp_path: Path) -> None:
    ds = det_dataset()
    ds.records[0].semantic_mask = SemanticMaskRef(Path("m.png"))
    with pytest.raises(ValueError, match="semantic masks"):
        run_remap(ds, {"rename": {"a": "z"}}, tmp=tmp_path)


def test_ids_index_classes_and_names_are_unique_after_both(tmp_path: Path) -> None:
    ds = det_dataset()
    ds.records[0].kpts = [Keypoints([(0.5, 0.5, 2.0)], 2, "c")]
    ds.records[0].polys = [Polygon(SQUARE, 1, "b")]
    out = run_remap(run_select(ds, "--drop-class", "a", "--empty", "keep"), {"merge": {"bc": ["b", "c"]}}, tmp=tmp_path)
    assert out.classes == ["bc"]
    for r in out.records:
        for item in [*r.boxes, *r.polys, *r.kpts]:
            assert item.cls == 0 and item.label == "bc"


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100)).save(path)


def test_coco_round_trip_after_a_merge_has_fresh_categories(tmp_path: Path) -> None:
    src = tmp_path / "src"
    records = []
    for i in range(2):
        write_image(src / f"i{i}.jpg")
        records.append(VisionRecord(image=ImageInfo(src / f"i{i}.jpg", 100, 100), boxes=[BBox(0.5, 0.5, 0.3, 0.3, i, "ab"[i])]))
    ds = VisionDataset(records=records, classes=["a", "b"], root=src, round_trip={"coco": {"categories": [{"id": 5, "name": "a"}, {"id": 9, "name": "b"}]}})
    out, _ = remap_mod.remap(ds, {"merge": {"ab": ["a", "b"]}, "rename": {}, "drop": []})
    write_dataset(out, tmp_path / "coco", format="coco")
    payload = json.loads((tmp_path / "coco" / "train" / "_annotations.coco.json").read_text())
    assert payload["categories"] == [{"id": 0, "name": "ab"}]
    assert {a["category_id"] for a in payload["annotations"]} == {0}


def test_labelme_round_trip_after_a_drop_and_a_size_filter(tmp_path: Path) -> None:
    src = tmp_path / "src"
    write_image(src / "a.jpg")
    shapes = [
        {"label": "a", "points": [[10, 10], [60, 60]], "group_id": None, "shape_type": "rectangle", "flags": {}},
        {"label": "b", "points": [[20, 20], [60, 60]], "group_id": None, "shape_type": "rectangle", "flags": {}},
        {"label": "b", "points": [[70, 70], [72, 72]], "group_id": None, "shape_type": "rectangle", "flags": {}},
    ]
    (src / "a.json").write_text(json.dumps({"version": "5", "flags": {}, "shapes": shapes, "imagePath": "a.jpg", "imageData": None, "imageHeight": 100, "imageWidth": 100}))
    ds = read_dataset(src, format="labelme")
    assert len(ds.records[0].labelme_shapes) == 3
    out = run_select(ds, "--min-box-px", "10", "--drop-class", "a", "--empty", "keep")
    assert [s.label for s in out.records[0].labelme_shapes] == ["b"]
    write_dataset(out, tmp_path / "out", format="labelme")
    written = json.loads(next((tmp_path / "out").rglob("*.json")).read_text())
    assert [(s["label"], s["points"][0]) for s in written["shapes"]] == [("b", [20.0, 20.0])]


def test_label_cli_chain_writes_the_narrowed_vocabulary(tmp_path: Path) -> None:
    src = tmp_path / "ds"
    (src / "train" / "images").mkdir(parents=True)
    (src / "train" / "labels").mkdir(parents=True)
    for i in range(10):
        write_image(src / "train" / "images" / f"s{i}.jpg")
        (src / "train" / "labels" / f"s{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.5 0.5\n0 0.2 0.2 0.05 0.05\n")
    (src / "data.yaml").write_text("names: [dog, cat]\n")
    (tmp_path / "classes.yaml").write_text("rename: {dog: puppy}\n")
    out = tmp_path / "out"
    code = label_cli.main([str(src), "select", "--min-box-px", "20", "--empty", "drop", "remap-classes", "--map", str(tmp_path / "classes.yaml"), "--others", "other", "to-yolo", str(out)])
    assert code == 0
    assert yaml.safe_load((out / "data.yaml").read_text())["names"] == ["puppy", "other"]
    labels = sorted(p.read_text().strip() for p in (out / "train" / "labels").glob("*.txt"))
    assert len(labels) == 10 and all(len(line.splitlines()) == 1 for line in labels)
