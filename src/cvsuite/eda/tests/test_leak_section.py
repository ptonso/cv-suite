"""The eda leak section on datasets where the label is, or is not, encoded in file metadata."""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import pytest
import yaml

pytest.importorskip("matplotlib")

from cvsuite.eda import cli as eda_cli  # noqa: E402
from cvsuite.eda.tests.planted import planted  # noqa: E402

FIGURES = ["01_metadata_cues.png", "02_cue_bins.png", "03_cue_contribution.png"]
TABLES = ["guess_accuracy.csv", "cue_bins.csv", "cue_contribution.csv"]
CUES = ["resolution", "aspect", "file_bytes", "container", "brightness", "sharpness", "name_digits", "file_index", "mtime_minute", "combined"]


def report(src: Path, out: Path, *args: str) -> Path:
    assert eda_cli.main([str(src), "--from", "class-dir", "to-report", str(out), "--only", "leak", *args]) == 0
    return out


def accuracy(out: Path, group: str = "") -> dict[str, dict]:
    rows = csv.DictReader((out / "leak" / "tables" / "guess_accuracy.csv").open())
    return {r["cue"]: r for r in rows if r["group"] == group and r["target"] in ("(label)", "(pooled)")}


def leaky(root: Path, **kwargs) -> Path:
    return planted(root, size_leak=True, per_class=40, test_per_class=20, **kwargs)


def test_a_size_leak_beats_the_floor_for_file_bytes_and_not_for_brightness(tmp_path: Path) -> None:
    out = report(leaky(tmp_path / "ds"), tmp_path / "r")
    for name in FIGURES:
        assert (out / "leak" / "figures" / name).is_file(), name
    for name in TABLES:
        assert (out / "leak" / "tables" / name).is_file(), name
    rows = accuracy(out)
    assert list(rows) == CUES
    chance = float(rows["file_bytes"]["chance"])
    assert chance == pytest.approx(1 / 3, abs=1e-3)
    assert float(rows["file_bytes"]["accuracy"]) > float(rows["file_bytes"]["floor"]) + 0.25
    assert float(rows["brightness"]["accuracy"]) < chance + 0.2
    assert float(rows["container"]["accuracy"]) == pytest.approx(chance, abs=1e-3), "the JPEG settings carry no label here"
    assert float(rows["combined"]["accuracy"]) > float(rows["combined"]["floor"]) + 0.2
    assert yaml.safe_load((out / "report.yaml").read_text())["notes"]["leak"]["plan"] == "fit on train, score on test"


def test_shuffled_labels_leave_every_cue_near_chance(tmp_path: Path) -> None:
    src = leaky(tmp_path / "ds")
    files = sorted(p for p in src.rglob("*.jpg"))
    labels = [p.parent.name for p in files]
    random.Random(0).shuffle(labels)
    for path, label in zip(files, labels):
        target = path.parent.parent / label / path.name
        target.parent.mkdir(exist_ok=True)
        if target != path:
            shutil.move(path, target)
    rows = accuracy(report(src, tmp_path / "r"))
    for cue, row in rows.items():
        assert float(row["accuracy"]) < float(row["chance"]) + 0.22, cue
        assert abs(float(row["floor"]) - float(row["chance"])) < 0.08, cue


def test_file_name_digits_reach_one_when_names_run_sequentially_by_class(tmp_path: Path) -> None:
    out = report(planted(tmp_path / "ds", classes=4, per_class=12, test_per_class=4, sequential_names=True), tmp_path / "r")
    rows = accuracy(out)
    assert float(rows["name_digits"]["accuracy"]) == 1.0
    assert float(rows["name_digits"]["floor"]) < 0.6


def test_unseen_bins_predict_the_majority_class(tmp_path: Path) -> None:
    from cvsuite.eda.core import cues
    import numpy as np

    ids, labels = cues.encode(["a", "a", "b", "c"], False, np.array([True, True, True, False]))
    assert ids.tolist() == [1, 1, 2, 0] and labels == ["unseen", "a", "b"]
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    ids, labels = cues.encode(values, True, np.array([True, True, True, True, False]))
    assert ids[-1] == len(labels) - 2, "a value above every fitted edge lands in the top bin"
    from cvsuite.eda.core.leak import _balanced

    predicted_all_majority = _balanced(np.array([1, 1, 2]), np.array([0, 0, 1]), np.array([0, 7]), np.array([0, 1]), 2)
    assert predicted_all_majority == pytest.approx(0.5), "ids never seen in fitting predict class 0, the majority"


def test_same_seed_gives_the_same_tables(tmp_path: Path) -> None:
    src = leaky(tmp_path / "ds")
    a, b = report(src, tmp_path / "a", "--seed", "4"), report(src, tmp_path / "b", "--seed", "4")
    for name in TABLES:
        assert (a / "leak" / "tables" / name).read_bytes() == (b / "leak" / "tables" / name).read_bytes()


def test_a_dataset_without_a_usable_split_is_scored_by_grouped_cross_validation(tmp_path: Path) -> None:
    src = leaky(tmp_path / "ds", groups=10, splits=("train", "train"))
    out = report(src, tmp_path / "r", "--group-by", r"regex:(g\d)_")
    notes = yaml.safe_load((out / "report.yaml").read_text())["notes"]["leak"]
    assert "cross-validation" in notes["plan"]
    rows = accuracy(out)
    assert int(rows["file_bytes"]["scoring_rows"]) == 180, "every image is scored once"
    assert float(rows["file_bytes"]["accuracy"]) > float(rows["file_bytes"]["floor"]) + 0.2


def test_within_group_accuracy_is_reported_for_groups_that_hold_both_splits(tmp_path: Path) -> None:
    src = leaky(tmp_path / "ds", groups=3)
    out = report(src, tmp_path / "r", "--group-by", r"regex:(g\d)_", "--top", "2")
    rows = list(csv.DictReader((out / "leak" / "tables" / "guess_accuracy.csv").open()))
    groups = {r["group"] for r in rows if r["group"]}
    assert groups == {"g0", "g1", "other (1)"}, "the cap rule keeps --top groups plus one other entry"
    within = [r for r in rows if r["group"] == "g0" and r["cue"] == "file_bytes"]
    assert within and float(within[0]["accuracy"]) > float(within[0]["chance"])


def test_annotated_datasets_are_scored_one_versus_rest_with_a_pooled_row(tmp_path: Path) -> None:
    from cvsuite.eda.tests.test_eda import make_dataset

    src = make_dataset(tmp_path / "ds", n=60, classes=4, splits=("train", "val", "test"))
    assert eda_cli.main([str(src), "to-report", str(tmp_path / "r"), "--only", "leak", "--top", "3"]) == 0
    rows = list(csv.DictReader((tmp_path / "r" / "leak" / "tables" / "guess_accuracy.csv").open()))
    targets = {r["target"] for r in rows}
    assert "(pooled)" in targets and len(targets) == 4
    pooled = next(r for r in rows if r["cue"] == "file_bytes" and r["target"] == "(pooled)")
    assert float(pooled["chance"]) == 0.5
    assert "one-versus-rest" in yaml.safe_load((tmp_path / "r" / "report.yaml").read_text())["notes"]["leak"]["target"]


def test_one_class_datasets_skip_leak_with_a_reason(tmp_path: Path) -> None:
    src = planted(tmp_path / "ds", classes=1)
    assert eda_cli.main([str(src), "--from", "class-dir", "to-report", str(tmp_path / "r")]) == 0
    section = yaml.safe_load((tmp_path / "r" / "report.yaml").read_text())["sections"]["leak"]
    assert section["status"] == "skipped" and "two distinct labels" in section["reason"]
    assert not (tmp_path / "r" / "leak").exists()
    with pytest.raises(SystemExit, match="cannot run"):
        eda_cli.main([str(src), "--from", "class-dir", "to-report", str(tmp_path / "r2"), "--only", "leak"])


def test_every_rate_carries_its_baselines(tmp_path: Path) -> None:
    out = report(leaky(tmp_path / "ds"), tmp_path / "r")
    for row in csv.DictReader((out / "leak" / "tables" / "guess_accuracy.csv").open()):
        assert 0.0 < float(row["chance"]) <= 1.0 and 0.0 <= float(row["floor"]) <= 1.0
    steps = list(csv.DictReader((out / "leak" / "tables" / "cue_contribution.csv").open()))
    assert [int(s["step"]) for s in steps] == list(range(1, 10)) and steps[0]["cue_added"] in ("file_bytes", "name_digits", "file_index")


def test_the_rule_compares_classes_by_the_share_of_their_records_that_show_a_value() -> None:
    import numpy as np

    from cvsuite.eda.core.leak import _balanced

    ids = np.r_[np.full(60, 1), np.full(40, 2), np.full(100, 1), np.full(300, 2)]
    y = np.r_[np.zeros(100, dtype=int), np.ones(400, dtype=int)]
    assert _balanced(ids, y, ids, y, 2) == pytest.approx((0.6 + 0.75) / 2), "class A shows value 1 in 60% of its records, class B in 25%"
