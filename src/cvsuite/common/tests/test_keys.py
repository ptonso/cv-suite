"""Key specs and grouping: `common/keys.py`."""

import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cvsuite.common.core import ImageInfo, VisionDataset, VisionRecord
from cvsuite.common.io import read_dataset
from cvsuite.common.keys import group_records, layout_path, parse_key


def rec(rel: str, **attributes) -> VisionRecord:
    return VisionRecord(image=ImageInfo(Path(rel), 8, 8), rel_image_path=Path(rel), attributes=attributes)


def ds(*rels: str, **kw) -> VisionDataset:
    return VisionDataset(records=[rec(r, **kw) for r in rels], root=Path("/nowhere"))


@pytest.mark.parametrize(
    "rel",
    ["train/images/pcb1/a.jpg", "images/train/pcb1/a.jpg", "pcb1/a.jpg", "valid/pcb1/a.jpg"],
)
def test_layout_path_drops_split_and_images_folders(rel: str) -> None:
    dataset = ds(rel)
    assert layout_path(dataset.records[0], dataset) == "pcb1/a.jpg"


def test_layout_path_keeps_the_file_name_and_inner_folders() -> None:
    dataset = ds("train/train/a.jpg", "images/images", "pcb/train/a.jpg")
    assert layout_path(dataset.records[0], dataset) == "a.jpg"
    assert layout_path(dataset.records[1], dataset) == "images"
    assert layout_path(dataset.records[2], dataset) == "pcb/train/a.jpg"


def test_layout_path_falls_back_to_the_image_path() -> None:
    record = VisionRecord(image=ImageInfo(Path("/data/set/val/x/a.jpg"), 8, 8))
    dataset = VisionDataset(records=[record], root=Path("/data/set"))
    assert layout_path(record, dataset) == "x/a.jpg"
    assert layout_path(record, VisionDataset(records=[record])) == "a.jpg"


def test_regex_dir_attr_values() -> None:
    dataset = ds("pcb1/Data/Anomaly/000.JPG", "oil-3-cam2-0042.jpg")
    assert parse_key(r"regex:^(\w+)-").values(ds("oil-3-cam2-0042.jpg")) == ["oil"]
    assert parse_key("regex:\\d+").values(ds("a_17.jpg")) == ["17"]
    assert parse_key("dir:0").values(ds("pcb1/Data/Anomaly/000.JPG")) == ["pcb1"]
    assert parse_key("dir:-1").values(ds("pcb1/Data/Anomaly/000.JPG")) == ["Anomaly"]
    assert dataset.records and parse_key("attr:board").values(ds("a.jpg", board=7)) == ["7"]


def test_unresolved_records_raise_naming_the_spec_and_paths() -> None:
    with pytest.raises(ValueError, match=r"regex:zzz.*a\.jpg"):
        parse_key("regex:zzz").values(ds("a.jpg", "b.jpg"))
    with pytest.raises(ValueError, match="dir:2"):
        parse_key("dir:2").values(ds("x/a.jpg"))
    with pytest.raises(ValueError, match="attr:board"):
        parse_key("attr:board").values(ds("a.jpg"))


def test_unknown_kind_and_bad_arguments_raise() -> None:
    with pytest.raises(ValueError, match="Valid kinds"):
        parse_key("nope:1")
    with pytest.raises(ValueError, match="Valid kinds"):
        parse_key("dir")
    with pytest.raises(ValueError, match="integer"):
        parse_key("dir:x")
    with pytest.raises(ValueError, match="bits"):
        parse_key("similar:99")


def test_similar_is_rejected_for_per_record_values() -> None:
    with pytest.raises(ValueError, match="--group-by"):
        parse_key("similar:4").values(ds("a.jpg"))


def test_file_key_matches_layout_path_then_unique_file_name(tmp_path: Path) -> None:
    table = tmp_path / "map.csv"
    table.write_text("x/a.jpg,g1\nb.jpg,g2\nc.jpg,g3\n")
    dataset = ds("x/a.jpg", "y/b.jpg", "z/c.jpg")
    assert parse_key(f"file:{table}").values(dataset) == ["g1", "g2", "g3"]
    clash = ds("y/b.jpg", "z/b.jpg")
    with pytest.raises(ValueError, match="does not resolve"):
        parse_key(f"file:{table}").values(clash)


def test_file_key_rejects_bad_tables(tmp_path: Path) -> None:
    three = tmp_path / "three.csv"
    three.write_text("a,b,c\n")
    with pytest.raises(ValueError, match="two columns"):
        parse_key(f"file:{three}")
    twice = tmp_path / "twice.csv"
    twice.write_text("a,1\na,2\n")
    with pytest.raises(ValueError, match="repeats"):
        parse_key(f"file:{twice}")
    with pytest.raises(ValueError, match="not found"):
        parse_key(f"file:{tmp_path / 'none.csv'}")


def test_groups_are_connected_components_of_all_specs() -> None:
    dataset = VisionDataset(
        records=[rec("a1.jpg", scene="s1"), rec("a2.jpg", scene="s2"), rec("b1.jpg", scene="s2"), rec("c1.jpg", scene="s3")],
        root=Path("/nowhere"),
    )
    assert group_records(dataset, []) == ["a1.jpg", "a2.jpg", "b1.jpg", "c1.jpg"]
    by_letter = group_records(dataset, ["regex:^(\\w)"])
    assert by_letter == ["a1.jpg", "a1.jpg", "b1.jpg", "c1.jpg"]
    both = group_records(dataset, ["regex:^(\\w)", "attr:scene"])
    assert both == ["a1.jpg", "a1.jpg", "a1.jpg", "c1.jpg"]


def test_group_ids_do_not_depend_on_record_order() -> None:
    names = [f"{c}{i}.jpg" for c in "abc" for i in range(4)]
    base = ds(*names)
    expected = dict(zip(names, group_records(base, ["regex:^(\\w)"])))
    for seed in range(5):
        shuffled = list(names)
        random.Random(seed).shuffle(shuffled)
        got = dict(zip(shuffled, group_records(ds(*shuffled), ["regex:^(\\w)"])))
        assert got == expected


def test_similar_groups_join_near_duplicates(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    base = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    other = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    for name, array in [("a.png", base), ("a_copy.png", base), ("z.png", other)]:
        Image.fromarray(array).save(tmp_path / name)
    dataset = VisionDataset(records=[
        VisionRecord(image=ImageInfo(Path(n), 64, 64), rel_image_path=Path(n)) for n in ["a.png", "a_copy.png", "z.png"]
    ], root=tmp_path)
    assert group_records(dataset, ["similar:2"]) == ["a.png", "a.png", "z.png"]
    from cvsuite.common import similarity

    bits = similarity.hash_bits([tmp_path / n for n in ["a.png", "a_copy.png", "z.png"]])
    assert group_records(dataset, ["similar:2"], bits=bits) == ["a.png", "a.png", "z.png"]


def test_layout_path_after_reading_each_layout(tmp_path: Path) -> None:
    def img(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (10, 20, 30)).save(path)

    yolo = tmp_path / "yolo"
    img(yolo / "images" / "train" / "a.jpg")
    (yolo / "labels" / "train").mkdir(parents=True)
    (yolo / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (yolo / "dataset.yaml").write_text("train: images/train\nnames:\n  0: a\n")
    flat = tmp_path / "flat" / "site1"
    img(flat / "a.jpg")
    classes = tmp_path / "classes"
    img(classes / "cat" / "a.jpg")

    got_yolo = read_dataset(yolo)
    assert [layout_path(r, got_yolo) for r in got_yolo.records] == ["a.jpg"]
    got_flat = read_dataset(tmp_path / "flat")
    assert [layout_path(r, got_flat) for r in got_flat.records] == ["site1/a.jpg"]
    got_cls = read_dataset(classes)
    assert [layout_path(r, got_cls) for r in got_cls.records] == ["cat/a.jpg"]


def test_split_key_reads_the_record_split() -> None:
    dataset = ds("a.jpg", "b.jpg", "c.jpg")
    for record, split in zip(dataset.records, ["train", "val", "test"]):
        record.split = split
    assert parse_key("split").values(dataset) == ["train", "val", "test"]
    with pytest.raises(ValueError, match="split"):
        parse_key("splits")
    with pytest.raises(ValueError, match="Unknown key spec"):
        parse_key("split:x")
