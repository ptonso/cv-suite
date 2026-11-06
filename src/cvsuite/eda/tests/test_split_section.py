"""The eda split section on datasets with planted twins, units, missing classes and straddling groups."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("matplotlib")

from cvsuite.common import similarity  # noqa: E402
from cvsuite.eda import cli as eda_cli  # noqa: E402
from cvsuite.eda.tests.planted import noise, planted, save  # noqa: E402

FIGURES = ["01_twin_rank.png", "02_twin_examples.jpg", "03_class_balance.png", "04_independent_units.png", "05_missing_classes.png"]
TABLES = ["twins.csv", "class_balance.csv", "units.csv", "missing_classes.csv"]


def report(src: Path, out: Path, *args: str) -> Path:
    assert eda_cli.main([str(src), "--from", "class-dir", "to-report", str(out), "--only", "split", *args]) == 0
    return out


def table(out: Path, name: str) -> list[dict]:
    return list(csv.DictReader((out / "split" / "tables" / name).open()))


def notes(out: Path) -> dict:
    return yaml.safe_load((out / "report.yaml").read_text())["notes"]["split"]


def test_planted_twins_are_flagged_and_a_random_split_shows_the_baseline(tmp_path: Path) -> None:
    out = report(planted(tmp_path / "ds", twins=True), tmp_path / "r")
    for name in FIGURES:
        assert (out / "split" / "figures" / name).is_file(), name
    for name in TABLES:
        assert (out / "split" / "tables" / name).is_file(), name
    assert notes(out)["twin_share"]["test"] >= 0.95
    assert 0.2 < notes(out)["baseline_share"]["test"] < 0.8
    rows = table(out, "twins.csv")
    assert len(rows) == 12 and all(r["is_twin"] == "1" and int(r["score"]) >= 58 for r in rows)
    assert all(r["closest_train"].split("/")[-1].startswith("t") for r in rows)


def test_unrelated_images_have_no_twins(tmp_path: Path) -> None:
    out = report(planted(tmp_path / "ds"), tmp_path / "r")
    assert notes(out)["twin_share"]["test"] <= 0.1 and notes(out)["baseline_share"]["test"] <= 0.1


def test_hash_and_ssim_agree_on_which_rows_are_twins(tmp_path: Path) -> None:
    src = planted(tmp_path / "ds", twins=True)
    for n, path in enumerate(sorted((src / "test" / "c1").iterdir())):
        save(noise(900 + n), path, 70)
    flagged = {}
    for measure in ("hash", "ssim"):
        out = report(src, tmp_path / measure, "--measure", measure)
        flagged[measure] = {r["path"] for r in table(out, "twins.csv") if r["is_twin"] == "1"}
    assert flagged["hash"] == flagged["ssim"]
    assert len(flagged["hash"]) == 8
    assert yaml.safe_load((tmp_path / "ssim" / "report.yaml").read_text())["notes"]["split"]["measure"] == "ssim"


def test_the_threshold_is_validated_in_the_unit_of_the_measure(tmp_path: Path) -> None:
    src = planted(tmp_path / "ds")
    with pytest.raises(SystemExit, match="whole number of bits"):
        report(src, tmp_path / "a", "--twin-threshold", "7.5")
    with pytest.raises(SystemExit, match=r"\[0, 64\]"):
        report(src, tmp_path / "b", "--twin-threshold", "70")
    with pytest.raises(SystemExit, match=r"\[0, 1\]"):
        report(src, tmp_path / "c", "--measure", "ssim", "--twin-threshold", "1.5")
    src = planted(tmp_path / "ds2", twins=True)
    exact, loose = report(src, tmp_path / "d", "--twin-threshold", "0"), report(src, tmp_path / "e", "--twin-threshold", "6")
    assert notes(exact)["threshold"] == 0.0 and notes(exact)["twin_share"]["test"] < notes(loose)["twin_share"]["test"]


def test_near_identical_test_images_count_as_one_unit_with_the_matching_half_width(tmp_path: Path) -> None:
    src = tmp_path / "ds"
    for c in range(2):
        for i in range(6):
            save(noise(c * 100 + i), src / "train" / f"c{c}" / f"t{c}_{i}.jpg", 75)
    for base in range(3):
        for variant, quality in enumerate((60, 70, 80, 90)):
            save(noise(500 + base), src / "test" / "c0" / f"x{base}_{variant}.jpg", quality)
    for i in range(4):
        save(noise(700 + i), src / "test" / "c1" / f"y{i}.jpg", 75)
    out = report(src, tmp_path / "r")
    rows = {r["class"]: r for r in table(out, "units.csv")}
    assert (rows["c0"]["rows"], rows["c0"]["units"]) == ("12", "3")
    assert float(rows["c0"]["half_width_95"]) == pytest.approx(1.96 * math.sqrt(0.25 / 3), abs=1e-4)
    assert (rows["c1"]["rows"], rows["c1"]["units"]) == ("4", "4")
    assert (rows["(all)"]["rows"], rows["(all)"]["units"]) == ("16", "7")
    assert float(rows["c1"]["half_width_95"]) < float(rows["c0"]["half_width_95"])


def test_a_class_absent_from_test_is_listed(tmp_path: Path) -> None:
    out = report(planted(tmp_path / "ds", test_classes=2, test_per_class=6), tmp_path / "r")
    rows = {r["bucket"]: r for r in table(out, "missing_classes.csv")}
    assert rows["0"]["classes"] == "1" and rows["0"]["names"] == "c2"
    assert rows["<5"]["classes"] == "1" and rows["<20"]["classes"] == "3"


def test_class_balance_lists_every_class_against_the_split_shares(tmp_path: Path) -> None:
    out = report(planted(tmp_path / "ds", classes=5, per_class=8, test_per_class=2), tmp_path / "r", "--top", "2")
    rows = table(out, "class_balance.csv")
    assert len(rows) == 5 * 2, "tables keep every class while the figure is capped"
    test_row = next(r for r in rows if r["class"] == "c0" and r["split"] == "test")
    assert float(test_row["class_share"]) == pytest.approx(0.2) and float(test_row["split_share"]) == pytest.approx(0.2)


def straddle_dataset(root: Path) -> Path:
    seed = 0
    for group, splits in (("gA", ("train", "test")), ("gB", ("train",)), ("gC", ("test",)), ("gD", ("train", "test"))):
        for split in splits:
            for i in range(3):
                seed += 1
                save(noise(seed), root / split / f"c{seed % 2}" / f"{group}_{split}{i}.jpg", 75)
    return root


def test_a_group_spanning_train_and_test_is_reported_and_confined_groups_are_not(tmp_path: Path) -> None:
    out = report(straddle_dataset(tmp_path / "ds"), tmp_path / "r", "--group-by", r"regex:(g[A-D])_")
    assert (out / "split" / "figures" / "06_group_straddle.png").is_file()
    rows = table(out, "straddle.csv")
    groups = {r["name"]: r for r in rows if r["kind"] == "group"}
    assert set(groups) == {"gA", "gD"}
    assert (groups["gA"]["rows_train"], groups["gA"]["rows_test"]) == ("3", "3")
    pair = next(r for r in rows if r["kind"] == "pair")
    assert (pair["name"], pair["groups"], pair["rows"]) == ("train+test", "2", "12")
    assert (out / "split" / "by_group" / "gA" / "figures" / "01_twin_rank.png").is_file()
    assert not (out / "split" / "by_group" / "gB").exists(), "a group without eval images has no curve"
    assert any("gB" in o["reason"] for o in yaml.safe_load((out / "report.yaml").read_text())["figures_not_drawn"])


def test_without_group_by_the_straddle_figure_is_omitted_and_listed(tmp_path: Path) -> None:
    out = report(planted(tmp_path / "ds"), tmp_path / "r")
    assert not (out / "split" / "figures" / "06_group_straddle.png").exists() and not (out / "split" / "tables" / "straddle.csv").exists()
    assert any(o["file"].endswith("06_group_straddle.png") for o in yaml.safe_load((out / "report.yaml").read_text())["figures_not_drawn"])


def test_by_group_curves_respect_top(tmp_path: Path) -> None:
    out = report(planted(tmp_path / "ds", groups=6, per_class=12, test_per_class=6), tmp_path / "r", "--group-by", r"regex:(g\d)_", "--top", "2")
    assert len(list((out / "split" / "by_group").iterdir())) <= 2


def test_max_images_samples_each_large_split_and_records_it(tmp_path: Path) -> None:
    src = planted(tmp_path / "ds", twins=True)
    first, second = report(src, tmp_path / "a", "--max-images", "5"), report(src, tmp_path / "b", "--max-images", "5")
    samples = yaml.safe_load((first / "report.yaml").read_text())["samples"]
    taken = {s["split"]: s for s in samples if s.get("section") == "split"}
    assert taken["train"]["of"] == 36 and taken["train"]["taken"] == 5 and taken["test"]["of"] == 12 and taken["test"]["taken"] == 5
    assert (first / "split" / "tables" / "twins.csv").read_text() == (second / "split" / "tables" / "twins.csv").read_text()
    assert len(table(first, "twins.csv")) == 5
    full = report(src, tmp_path / "c", "--max-images", "100")
    assert not [s for s in yaml.safe_load((full / "report.yaml").read_text())["samples"] if s.get("section") == "split"]


def test_sections_reuse_the_table_hashes_and_never_hash_again(tmp_path: Path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("a section recomputed hashes")

    monkeypatch.setattr(similarity, "hash_bits", forbidden)
    out = report(planted(tmp_path / "ds", twins=True), tmp_path / "r", "--group-by", "similar:6")
    assert notes(out)["twin_share"]["test"] >= 0.95
    groups = {r["group"] for r in csv.DictReader((out / "images.csv").open())}
    assert len(groups) == 36, "each train image and its twin share a group, one group per base image"


def test_same_seed_gives_the_same_tables(tmp_path: Path) -> None:
    src = planted(tmp_path / "ds", twins=True)
    a, b = report(src, tmp_path / "a", "--seed", "3"), report(src, tmp_path / "b", "--seed", "3")
    for name in TABLES:
        assert (a / "split" / "tables" / name).read_bytes() == (b / "split" / "tables" / name).read_bytes()


def test_a_dataset_without_a_test_or_val_split_skips_split(tmp_path: Path, capsys) -> None:
    src = planted(tmp_path / "ds", splits=("train", "train"))
    assert eda_cli.main([str(src), "--from", "class-dir", "to-report", str(tmp_path / "r")]) == 0
    skipped = yaml.safe_load((tmp_path / "r" / "report.yaml").read_text())["sections"]["split"]
    assert skipped["status"] == "skipped" and "two splits" in skipped["reason"]
    with pytest.raises(SystemExit, match="cannot run"):
        eda_cli.main([str(src), "--from", "class-dir", "to-report", str(tmp_path / "r2"), "--only", "split"])


def test_nearest_candidates_do_not_change_the_ssim_ranking_on_a_small_set(tmp_path: Path) -> None:
    """The 25 hash candidates find the same best match as a scan of every train image."""
    from cvsuite.eda.core import split as split_mod
    from cvsuite.eda.core.sections import Context
    from cvsuite.eda.core.table import build_table
    from cvsuite.label.core import router

    src = planted(tmp_path / "ds", twins=True, per_class=10, test_per_class=5)
    dataset = router.ingest_many([src], from_hint="class-dir")
    table_ = build_table(dataset, [])
    ctx = Context(dataset=dataset, table=table_, measure="ssim", twin_threshold=0.6)
    train = np.array([i for i, r in enumerate(table_.rows) if r["split"] == "train"])
    test = np.array([i for i, r in enumerate(table_.rows) if r["split"] == "test"])
    found = split_mod.measure_twins(ctx, "test", test, train)
    full = [max(similarity.ssim(table_.thumbs[i], table_.thumbs[j]) for j in train) for i in test]
    assert np.allclose(found.scores, full)
