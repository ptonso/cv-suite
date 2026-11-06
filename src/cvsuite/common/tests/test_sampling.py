"""Nested sampling: a smaller sample is a subset of a larger one (`specs/common/sampling.md`)."""

from __future__ import annotations

import random
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from cvsuite.classify.transform.core.sample import run as class_sample
from cvsuite.common.core import Classification, ImageInfo, VisionDataset, VisionRecord
from cvsuite.common.sampling import rank_order, sample_size, take
from cvsuite.label import cli as label_cli
from cvsuite.label.transform.sample.commands.run import run as label_sample
from cvsuite.prep import cli as prep_cli


@pytest.mark.parametrize("n, sizes", [(200, (20, 40, 80)), (5000, (1000, 1500, 2500)), (30, (1, 5, 30))])
def test_samples_nest_across_sizes(n: int, sizes: tuple[int, ...]) -> None:
    keys = [f"img_{i}.jpg" for i in range(n)]
    taken = [set(take(keys, 3, k)) for k in sizes]
    assert all(small <= large for small, large in zip(taken, taken[1:]))
    assert [len(t) for t in taken] == list(sizes)


def test_selection_ignores_input_order_and_new_items_keep_relative_rank() -> None:
    keys = [f"k{i}" for i in range(100)]
    chosen = {keys[i] for i in take(keys, 1, 30)}
    shuffled = list(keys)
    random.Random(0).shuffle(shuffled)
    assert {shuffled[i] for i in take(shuffled, 1, 30)} == chosen
    ranked = [keys[i] for i in rank_order(keys, 1)]
    extended = keys + ["extra"]
    assert [extended[i] for i in rank_order(extended, 1) if extended[i] != "extra"] == ranked


def test_seeds_differ_and_duplicates_raise() -> None:
    keys = [f"k{i}" for i in range(100)]
    assert take(keys, 1, 30) != take(keys, 2, 30)
    with pytest.raises(ValueError, match="duplicates"):
        rank_order(["a", "b", "a"], 0)


def test_sample_size_rules() -> None:
    assert sample_size(10, count=3, frac=None) == 3
    assert sample_size(10, count=None, frac=0.25) == 2
    assert sample_size(10, count=None, frac=0.01) == 1
    assert sample_size(10, count=None, frac=0.0) == 0
    with pytest.raises(SystemExit):
        sample_size(10, count=11, frac=None)
    with pytest.raises(SystemExit):
        sample_size(10, count=None, frac=1.5)


def label_dataset(n: int = 200) -> VisionDataset:
    return VisionDataset(
        records=[VisionRecord(image=ImageInfo(Path(f"i{i}.jpg"), 8, 8), rel_image_path=Path(f"i{i}.jpg")) for i in range(n)],
        root=Path("/x"),
    )


def test_label_sample_nests() -> None:
    ds = label_dataset()
    pick = lambda frac: {r.rel_image_path.name for r in label_sample(ds, Namespace(count=None, frac=frac, seed=4)).records}  # noqa: E731
    assert pick(0.1) <= pick(0.25) <= pick(0.5)
    assert len(pick(0.1)) == 20


def test_label_sample_survives_repeated_layout_paths_across_sources(tmp_path: Path) -> None:
    records = []
    for source in ("one", "two"):
        path = tmp_path / source / "a.jpg"
        path.parent.mkdir()
        Image.new("RGB", (8, 8)).save(path)
        records.append(VisionRecord(image=ImageInfo(path, 8, 8), rel_image_path=Path("a.jpg")))
    ds = VisionDataset(records=records, root=tmp_path)
    assert len(label_sample(ds, Namespace(count=2, frac=None, seed=0)).records) == 2


def test_class_sample_nests_per_class() -> None:
    records = [
        VisionRecord(image=ImageInfo(Path(f"{label}_{i}.jpg"), 8, 8), rel_image_path=Path(f"{label}/{label}_{i}.jpg"), classification=Classification(label=label))
        for label in ("cat", "dog") for i in range(100)
    ]
    ds = VisionDataset(records=records, classes=["cat", "dog"], root=Path("/x"))
    pick = lambda frac: {r.rel_image_path.name for r in class_sample(ds, Namespace(mode="preserve", max_n_per_class=None, max_frac_per_class=frac, with_replacement=False, seed=2)).records}  # noqa: E731
    assert pick(0.1) <= pick(0.25)
    assert len(pick(0.1)) == 20


def test_prep_sample_nests_and_reproduces_without_a_seed(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(60):
        Image.new("RGB", (8, 8), (i, 0, 0)).save(src / f"p{i}.jpg")

    def run(frac: str, *extra: str) -> set[str]:
        out = tmp_path / ("out_" + frac + "_".join(extra))
        assert prep_cli.main(["sample", str(src), str(out), "--frac", frac, *extra]) == 0
        return {p.name for p in out.iterdir()}

    assert run("0.2") == run("0.2", "--seed", "0")
    assert run("0.2", "--seed", "3") <= run("0.5", "--seed", "3")
    assert run("0.2", "--seed", "3") != run("0.2", "--seed", "4")


def test_label_cli_sample_fraction_is_a_subset(tmp_path: Path) -> None:
    src = tmp_path / "ds"
    (src / "train" / "images").mkdir(parents=True)
    (src / "train" / "labels").mkdir(parents=True)
    for i in range(40):
        Image.new("RGB", (8, 8), (i, 0, 0)).save(src / "train" / "images" / f"s{i}.jpg")
        (src / "train" / "labels" / f"s{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (src / "data.yaml").write_text("names: [a]\n")
    for frac in ("0.25", "0.5"):
        assert label_cli.main([str(src), "sample", "--frac", frac, "--seed", "5", "to-yolo", str(tmp_path / f"o{frac}")]) == 0
    small = {p.name for p in (tmp_path / "o0.25" / "train" / "images").iterdir()}
    large = {p.name for p in (tmp_path / "o0.5" / "train" / "images").iterdir()}
    assert small < large and len(small) == 10


def split_dataset(per_split: dict[str, int]) -> VisionDataset:
    records = [
        VisionRecord(image=ImageInfo(Path(f"{split}{i}.jpg"), 8, 8), rel_image_path=Path(f"{split}{i}.jpg"), split=split, classification=Classification(label="cat" if i % 2 else "dog"))
        for split, n in per_split.items() for i in range(n)
    ]
    return VisionDataset(records=records, classes=["cat", "dog"], root=Path("/x"))


def test_label_sample_by_split_keeps_the_fraction_in_every_split_and_nests() -> None:
    ds = split_dataset({"train": 100, "val": 30, "test": 20})
    run = lambda frac: label_sample(ds, Namespace(count=None, frac=frac, seed=1, stratify_by="split"))  # noqa: E731
    out = run(0.2)
    counts = {s: sum(r.split == s for r in out.records) for s in ("train", "val", "test")}
    assert counts == {"train": 20, "val": 6, "test": 4}
    assert {r.rel_image_path.name for r in run(0.1).records} <= {r.rel_image_path.name for r in out.records}
    assert out.meta["step_info"]["strata"] == 3
    assert [r.rel_image_path for r in out.records] == sorted((r.rel_image_path for r in out.records), key=[x.rel_image_path for x in ds.records].index)


def test_label_sample_count_per_stratum_and_errors() -> None:
    ds = split_dataset({"train": 10, "val": 3})
    assert len(label_sample(ds, Namespace(count=2, frac=None, seed=0, stratify_by="split")).records) == 4
    with pytest.raises(SystemExit, match="Stratum `val`"):
        label_sample(ds, Namespace(count=5, frac=None, seed=0, stratify_by="split"))
    with pytest.raises(SystemExit, match="already samples per class"):
        label_sample(ds, Namespace(count=1, frac=None, seed=0, stratify_by="class"))
    for i, record in enumerate(ds.records):
        record.attributes["group"] = "a" if i % 2 else "b"
    joint = label_sample(ds, Namespace(count=1, frac=None, seed=0, stratify_by="split,attr:group"))
    assert len(joint.records) == 4


def test_class_sample_by_split_keeps_class_proportions_per_split() -> None:
    ds = split_dataset({"train": 100, "val": 40})
    out = class_sample(ds, Namespace(mode="preserve", max_n_per_class=None, max_frac_per_class=0.25, with_replacement=False, seed=0, stratify_by="split"))
    seen = {(r.split, r.classification.label) for r in out.records}
    assert seen == {(s, c) for s in ("train", "val") for c in ("cat", "dog")}
    assert sum(r.split == "train" for r in out.records) == 24 and sum(r.split == "val" for r in out.records) == 10
    assert set(out.meta["step_info"]["targets"]) == {"train/cat", "train/dog", "val/cat", "val/dog"}
