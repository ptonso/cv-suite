"""The `split` transform and fold writing (`specs/common/split.md`)."""

from __future__ import annotations

import filecmp
from pathlib import Path

import pytest
from PIL import Image

from cvsuite.branch_cli import build_parser
from cvsuite.common.core import BBox, ImageInfo, Polygon, VisionDataset, VisionRecord
from cvsuite.common.io.folds import fold_targets, require_single_fold
from cvsuite.common.transforms import split as split_mod
from cvsuite.label import cli as label_cli


def run_split(dataset: VisionDataset, *argv: str) -> VisionDataset:
    return split_mod.run(dataset, build_parser(split_mod, "split").parse_args(["--tolerance", "1", *argv]))


def make(n: int = 40, groups: int = 10) -> VisionDataset:
    records = []
    for i in range(n):
        label = "a" if i % 2 == 0 else "b"
        records.append(
            VisionRecord(
                image=ImageInfo(Path(f"g{i % groups}_{i}.jpg"), 8, 8),
                rel_image_path=Path(f"g{i % groups}_{i}.jpg"),
                boxes=[BBox(0.5, 0.5, 0.2, 0.2, 0 if label == "a" else 1, label)],
            )
        )
    return VisionDataset(records=records, classes=["a", "b"], root=Path("/x"))


def splits_of(ds: VisionDataset) -> dict[str, str]:
    return {r.rel_image_path.name: r.split for r in ds.records}


def test_ratio_split_keeps_groups_together_and_is_deterministic() -> None:
    ds = make()
    out = run_split(ds, "--group-by", r"regex:^(g\d+)_", "--val-frac", "0.2", "--test-frac", "0.2", "--seed", "3")
    by_group: dict[str, set[str]] = {}
    for r in out.records:
        by_group.setdefault(r.rel_image_path.name.split("_")[0], set()).add(r.split)
    assert all(len(v) == 1 for v in by_group.values())
    assert {r.split for r in out.records} == {"train", "val", "test"}
    assert splits_of(out) == splits_of(run_split(ds, "--group-by", r"regex:^(g\d+)_", "--val-frac", "0.2", "--test-frac", "0.2", "--seed", "3"))
    assert all(r.split == "train" for r in ds.records), "the input dataset is not mutated"


def test_split_is_independent_of_record_order() -> None:
    ds = make()
    reversed_ds = VisionDataset(records=list(reversed(ds.records)), classes=ds.classes, root=ds.root)
    args = ("--group-by", r"regex:^(g\d+)_", "--val-frac", "0.3")
    assert splits_of(run_split(ds, *args)) == splits_of(run_split(reversed_ds, *args))


def test_step_info_reports_shares() -> None:
    out = run_split(make(), "--val-frac", "0.25")
    info = out.meta["step_info"]
    assert info["mode"] == "ratio" and info["groups"]["val"] > 0 and info["max_deviation"] <= 0.25


def test_folds_partition_and_only_fold_matches() -> None:
    ds = make(60, 20)
    full = run_split(ds, "--folds", "3", "--val-frac", "0.2", "--group-by", r"regex:^(g\d+)_")
    assert full.meta["folds"] == {"k": 3, "present": [0, 1, 2]}
    tests = [{r.rel_image_path.name for r in ds_i.records if r.split == "test"} for ds_i, _ in fold_targets(full, Path("x"))]
    assert sum(len(t) for t in tests) == 60 and set().union(*tests) == {r.rel_image_path.name for r in ds.records}
    only = run_split(ds, "--folds", "3", "--val-frac", "0.2", "--group-by", r"regex:^(g\d+)_", "--only-fold", "1")
    (ds_only, dst_only), = list(fold_targets(only, Path("x")))
    assert dst_only == Path("x/fold1")
    ds_full_1 = dict((dst.name, d) for d, dst in fold_targets(full, Path("x")))["fold1"]
    assert splits_of(ds_only) == splits_of(ds_full_1)


def test_fold_targets_strip_fold_state_and_review_outputs_need_one_fold() -> None:
    full = run_split(make(60, 20), "--folds", "3")
    for ds_i, _ in fold_targets(full, Path("x")):
        assert "folds" not in ds_i.meta and all("fold_splits" not in r.attributes for r in ds_i.records)
    with pytest.raises(SystemExit, match="--only-fold"):
        require_single_fold(full)
    one = require_single_fold(run_split(make(60, 20), "--folds", "3", "--only-fold", "2"))
    assert any(r.split == "test" for r in one.records)
    plain = make()
    assert require_single_fold(plain) is plain
    assert list(fold_targets(plain, Path("d"))) == [(plain, Path("d"))]


def test_keep_pins_the_listed_splits() -> None:
    ds = make(40, 40)
    for r in ds.records[:6]:
        r.split = "test"
    out = run_split(ds, "--keep", "test", "--val-frac", "0.25")
    assert [r.split for r in out.records[:6]] == ["test"] * 6
    assert sum(r.split == "test" for r in out.records) == 6
    assert any(r.split == "val" for r in out.records)


def test_keep_pins_the_whole_group() -> None:
    ds = make(40, 10)
    ds.records[0].split = "test"
    out = run_split(ds, "--keep", "test", "--val-frac", "0.25", "--group-by", r"regex:^(g\d+)_")
    assert {r.split for r in out.records if r.rel_image_path.name.startswith("g0_")} == {"test"}


def test_keep_with_conflicting_pins_in_one_group_raises() -> None:
    ds = make(40, 10)
    ds.records[0].split, ds.records[10].split = "test", "train"
    with pytest.raises(SystemExit, match="different splits"):
        run_split(ds, "--keep", "test,train", "--group-by", r"regex:^(g\d+)_", "--val-frac", "0.1")


@pytest.mark.parametrize(
    "argv, message",
    [
        ([], "positive"),
        (["--folds", "1"], "at least 2"),
        (["--folds", "3", "--test-frac", "0.1"], "--test-frac"),
        (["--folds", "3", "--keep", "test"], "--keep"),
        (["--folds", "3", "--only-fold", "3"], "indices"),
        (["--val-frac", "0.2", "--only-fold", "0"], "needs --folds"),
        (["--val-frac", "0.2", "--test-frac", "0.1", "--keep", "test"], "pins that split"),
        (["--val-frac", "0.2", "--keep", "bogus"], "--keep takes"),
        (["--val-frac", "0.7", "--test-frac", "0.5"], "below 1"),
        (["--val-frac", "0.2", "--stratify-by", "similar:4"], "--group-by"),
    ],
)
def test_invalid_options_raise(argv: list[str], message: str) -> None:
    with pytest.raises((SystemExit, ValueError), match=message):
        run_split(make(), *argv)


def test_tolerance_failure_exits_with_the_table() -> None:
    ds = make(60, 1)
    with pytest.raises(SystemExit, match="tolerance"):
        split_mod.run(ds, build_parser(split_mod, "split").parse_args(["--val-frac", "0.5", "--group-by", "regex:^(g\\d+)_", "--stratify-by", "none", "--tolerance", "0.0"]))


def test_stratify_by_class_counts_polygons_once() -> None:
    record = VisionRecord(
        image=ImageInfo(Path("a.jpg"), 8, 8),
        boxes=[BBox(0.5, 0.5, 0.2, 0.2, 0, "a", kind="seg")],
        polys=[Polygon([(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)], 0, "a")],
    )
    matrix, columns = split_mod.weights(VisionDataset(records=[record], classes=["a"]), "class")
    assert columns == ("a",) and matrix.tolist() == [[1.0]]


def test_stratify_by_class_uses_classification_labels_and_empty_column() -> None:
    from cvsuite.common.core import Classification

    records = [
        VisionRecord(image=ImageInfo(Path("a.jpg"), 8, 8), classification=Classification(label="cat")),
        VisionRecord(image=ImageInfo(Path("b.jpg"), 8, 8)),
    ]
    matrix, columns = split_mod.weights(VisionDataset(records=records, classes=["cat"]), "class")
    assert columns == ("cat", "(empty)") and matrix.tolist() == [[1, 0], [0, 1]]


def test_label_cli_split_folds_writes_each_fold_and_only_fold_matches(tmp_path: Path) -> None:
    src = tmp_path / "src"
    for i in range(12):
        (src / "train" / "images").mkdir(parents=True, exist_ok=True)
        (src / "train" / "labels").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (i * 10, 0, 0)).save(src / "train" / "images" / f"s{i}.jpg")
        (src / "train" / "labels" / f"s{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.4 0.4\n")
    (src / "data.yaml").write_text("names:\n  0: a\n  1: b\n")
    full, only = tmp_path / "full", tmp_path / "only"
    args = ["split", "--folds", "3", "--val-frac", "0.25", "--tolerance", "1"]
    assert label_cli.main([str(src), *args, "to-coco", str(full)]) == 0
    assert sorted(p.name for p in full.iterdir()) == ["fold0", "fold1", "fold2"]
    assert label_cli.main([str(src), *args, "--only-fold", "1", "to-coco", str(only)]) == 0
    assert [p.name for p in only.iterdir()] == ["fold1"]
    assert not filecmp.dircmp(full / "fold1", only / "fold1").diff_files
    assert (full / "fold1" / "data.yaml").exists()


def board_dataset() -> VisionDataset:
    records = [
        VisionRecord(image=ImageInfo(Path(f"b{n}_{i}.jpg"), 8, 8), rel_image_path=Path(f"b{n}_{i}.jpg"), boxes=[BBox(0.5, 0.5, 0.2, 0.2, 0, "a")])
        for n in range(20) for i in range(2)
    ]
    return VisionDataset(records=records, classes=["a"], root=Path("/x"))


def test_on_siblings_drop_keeps_the_pinned_split_exact() -> None:
    ds = board_dataset()
    ds.records[0].split = "test"
    ds.records[6].split = "test"
    pinned = [r.rel_image_path for r in ds.records if r.split == "test"]
    args = ("--keep", "test", "--val-frac", "0.25", "--group-by", r"regex:^(b\d+)_")
    joined = run_split(ds, *args)
    assert sum(r.split == "test" for r in joined.records) == 4
    dropped = run_split(ds, *args, "--on-siblings", "drop")
    assert [r.rel_image_path for r in dropped.records if r.split == "test"] == pinned
    assert len(dropped.records) == 40 - 2
    assert dropped.meta["step_info"]["siblings_dropped"] == 2
    assert not {r.rel_image_path.name.split("_")[0] for r in dropped.records if r.split == "test"} & {r.rel_image_path.name.split("_")[0] for r in dropped.records if r.split != "test"}


def test_on_siblings_needs_keep() -> None:
    with pytest.raises(SystemExit, match="needs --keep"):
        run_split(board_dataset(), "--val-frac", "0.2", "--on-siblings", "drop")


def test_class_size_columns_split_each_class_by_box_size() -> None:
    records = [
        VisionRecord(image=ImageInfo(Path("a.jpg"), 100, 100), boxes=[BBox(0.5, 0.5, 0.1, 0.1, 0, "a"), BBox(0.5, 0.5, 0.3, 0.3, 0, "a"), BBox(0.5, 0.5, 0.3, 0.3, 1, "b")]),
        VisionRecord(image=ImageInfo(Path("b.jpg"), 100, 100)),
    ]
    matrix, columns = split_mod.weights(VisionDataset(records=records, classes=["a", "b"]), "class-size:20")
    assert columns == ("a|<20", "a|>=20", "b|<20", "b|>=20", "(empty)")
    assert matrix.tolist() == [[1, 1, 0, 1, 0], [0, 0, 0, 0, 1]]
    _, three = split_mod.weights(VisionDataset(records=records, classes=["a", "b"]), "class-size:5,20")
    assert three[:3] == ("a|<5", "a|5-20", "a|>=20")


def test_class_size_rejects_bad_thresholds_and_classification_data() -> None:
    from cvsuite.common.core import Classification

    ds = VisionDataset(records=[VisionRecord(image=ImageInfo(Path("a.jpg"), 8, 8), classification=Classification(label="cat"))], classes=["cat"])
    with pytest.raises(SystemExit, match="boxes or polygons"):
        split_mod.weights(ds, "class-size:20")
    for bad in ("class-size:20,10", "class-size:0", "class-size:x"):
        with pytest.raises(SystemExit):
            split_mod.weights(board_dataset(), bad)


def test_split_balances_class_size_columns() -> None:
    records = []
    for i in range(80):
        side = 0.1 if i % 4 == 0 else 0.4
        records.append(VisionRecord(image=ImageInfo(Path(f"i{i}.jpg"), 100, 100), rel_image_path=Path(f"i{i}.jpg"), boxes=[BBox(0.5, 0.5, side, side, 0, "a")]))
    out = run_split(VisionDataset(records=records, classes=["a"], root=Path("/x")), "--stratify-by", "class-size:20", "--val-frac", "0.25", "--test-frac", "0.25")
    for split in ("train", "val", "test"):
        small = sum(1 for r in out.records if r.split == split and r.boxes[0].w < 0.2)
        assert small in (range(3, 11) if split == "train" else range(2, 8))
