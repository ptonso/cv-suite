"""The eda branch: image table, sections, cap rule, chaining and report folder rules."""

from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

pytest.importorskip("matplotlib")

from cvsuite.eda import cli as eda_cli  # noqa: E402
from cvsuite.eda.core import style  # noqa: E402
from cvsuite.eda.core.table import COLUMNS  # noqa: E402

PROFILE = ["figures/01_contact_sheet.jpg", "figures/02_class_examples.jpg", "figures/03_extremes.jpg"]
STRUCTURE = [f"figures/{n}" for n in ("01_class_counts.png", "02_class_by_split.png", "03_long_tail.png", "04_box_size.png", "05_instances_per_image.png", "06_box_centers.png", "07_resolutions.png")] + ["tables/structure.csv", "tables/box_sizes.csv"]


def make_dataset(root: Path, *, n: int = 24, classes: int = 3, splits: tuple[str, ...] = ("train", "val"), size: int = 48) -> Path:
    names = [f"c{i}" for i in range(classes)]
    rng = np.random.default_rng(0)
    for i in range(n):
        split = splits[i % len(splits)]
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        Image.fromarray((rng.random((size, size + (i % 3) * 8, 3)) * 255).astype("uint8")).save(root / split / "images" / f"b{i % 4}_{i}.jpg")
        lines = [f"{(i + j) % classes} {0.3 + 0.1 * j:.2f} {0.4 + 0.05 * j:.2f} {0.2 + 0.02 * i / n:.3f} {0.2:.2f}" for j in range(1 + i % 3)]
        (root / split / "labels" / f"b{i % 4}_{i}.txt").write_text("\n".join(lines) + "\n")
    entries = "".join(f"{split}: {split}/images\n" for split in splits)
    (root / "data.yaml").write_text(entries + "names: [" + ", ".join(names) + "]\n")
    return root


def tree_digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha1(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def report_yaml(out: Path) -> dict:
    return yaml.safe_load((out / "report.yaml").read_text())


def test_full_report_writes_every_available_section(tmp_path: Path) -> None:
    src = make_dataset(tmp_path / "ds")
    out = tmp_path / "r"
    assert eda_cli.main([str(src), "to-report", str(out)]) == 0
    for name in [*PROFILE]:
        assert (out / "profile" / name).is_file()
    for name in STRUCTURE:
        assert (out / "structure" / name).is_file()
    assert (out / "health" / "tables" / "health.csv").is_file()
    report = report_yaml(out)
    assert report["sections"]["profile"]["status"] == "ran" and report["sections"]["health"]["status"] == "ran"
    assert report["sections"]["split"]["status"] == "ran" and report["sections"]["leak"]["status"] == "ran"
    assert (out / "split" / "figures" / "01_twin_rank.png").is_file() and (out / "leak" / "figures" / "01_metadata_cues.png").is_file()
    assert report["records"] == 24


def test_single_split_skips_split_section_with_its_reason(tmp_path: Path, capsys) -> None:
    src = make_dataset(tmp_path / "ds", splits=("train",))
    out = tmp_path / "r"
    eda_cli.main([str(src), "to-report", str(out)])
    assert "needs at least two splits" in report_yaml(out)["sections"]["split"]["reason"]
    assert "skipped split" in capsys.readouterr().out
    assert any(o["file"].endswith("02_class_by_split.png") for o in report_yaml(out)["figures_not_drawn"])
    assert not (out / "structure" / "figures" / "02_class_by_split.png").exists()


def test_images_csv_columns_and_values(tmp_path: Path) -> None:
    src = make_dataset(tmp_path / "ds", n=6)
    out = tmp_path / "r"
    eda_cli.main([str(src), "to-report", str(out), "--only", "health"])
    rows = list(csv.DictReader((out / "images.csv").open()))
    assert tuple(rows[0]) == tuple(c for c in COLUMNS if c != "group")
    assert len(rows) == 6
    row = next(r for r in rows if r["path"] == "b0_0.jpg")
    assert (row["split"], row["width"], row["height"], row["ext"]) == ("train", "48", "48", "jpg")
    assert row["class"] == "c0" and row["n_instances"] == "1"
    assert len(row["phash"]) == 16 and float(row["brightness"]) > 0 and float(row["sharpness"]) > 0 and int(row["file_bytes"]) > 0


def test_group_by_adds_a_column_and_panels_for_the_largest_groups(tmp_path: Path) -> None:
    src = make_dataset(tmp_path / "ds")
    out = tmp_path / "r"
    eda_cli.main([str(src), "to-report", str(out), "--group-by", r"regex:^(b\d)_", "--top", "2", "--only", "profile,structure"])
    rows = list(csv.DictReader((out / "images.csv").open()))
    assert {r["group"] for r in rows} == {"b0", "b1", "b2", "b3"}
    panels = sorted(p.name for p in (out / "structure" / "by_group").iterdir())
    assert len(panels) == 2
    assert (out / "structure" / "by_group" / panels[0] / "tables" / "structure.csv").is_file()
    assert (out / "profile" / "by_group" / panels[0] / "figures" / "01_contact_sheet.jpg").is_file()
    assert (out / "structure" / "figures" / "01_class_counts.png").is_file(), "pooled outputs stay"
    assert "by_group/" in " ".join(report_yaml(out)["sections"]["structure"]["files"])


def test_cap_rule_on_a_thousand_classes(tmp_path: Path) -> None:
    counts = {f"class_{i:04d}": float(1000 - i) for i in range(1000)}
    entries, k = style.top_entries(counts, 20)
    assert len(entries) == 21 and k == 980 and entries[-1][0] == "other (980)"
    assert entries[-1][1] == pytest.approx(sum(counts.values()) - sum(v for _, v in entries[:-1]))
    assert style.top_entries({"a": 1.0, "b": 2.0}, 5) == ([("b", 2.0), ("a", 1.0)], 0)


def test_structure_table_lists_every_class_while_figures_are_capped(tmp_path: Path) -> None:
    src = make_dataset(tmp_path / "ds", n=60, classes=60)
    out = tmp_path / "r"
    eda_cli.main([str(src), "to-report", str(out), "--only", "structure", "--top", "5"])
    rows = list(csv.DictReader((out / "structure" / "tables" / "structure.csv").open()))
    assert len(rows) == 60
    assert (out / "structure" / "figures" / "03_long_tail.png").is_file(), "the whole distribution has its own figure"
    assert len(list(csv.DictReader((out / "structure" / "tables" / "box_sizes.csv").open()))) == 60


def test_only_runs_named_sections_and_rejects_unknown_or_impossible_ones(tmp_path: Path) -> None:
    src = make_dataset(tmp_path / "ds", splits=("train",))
    out = tmp_path / "r"
    eda_cli.main([str(src), "to-report", str(out), "--only", "health"])
    assert (out / "health").is_dir() and not (out / "structure").exists()
    assert report_yaml(out)["sections"]["structure"]["reason"] == "not selected by --only"
    with pytest.raises(SystemExit, match="Unknown section"):
        eda_cli.main([str(src), "to-report", str(tmp_path / "r2"), "--only", "bogus"])
    with pytest.raises(SystemExit, match="cannot run"):
        eda_cli.main([str(src), "to-report", str(tmp_path / "r3"), "--only", "split"])


def test_the_source_is_unchanged_and_a_rerun_needs_overwrite(tmp_path: Path) -> None:
    src = make_dataset(tmp_path / "ds")
    before = tree_digest(src)
    out = tmp_path / "r"
    eda_cli.main([str(src), "to-report", str(out)])
    assert tree_digest(src) == before
    with pytest.raises(FileExistsError, match="--overwrite"):
        eda_cli.main([str(src), "to-report", str(out)])
    stray = out / "notes.txt"
    stray.write_text("mine")
    eda_cli.main([str(src), "to-report", str(out), "--only", "structure", "--overwrite"])
    assert stray.read_text() == "mine" and (out / "health" / "tables" / "health.csv").is_file()
    assert report_yaml(out)["sections"]["profile"]["reason"] == "not selected by --only"
    assert tree_digest(src) == before


def test_a_missing_matplotlib_raises_before_ingest(tmp_path: Path, monkeypatch) -> None:
    called = []
    monkeypatch.setattr(eda_cli.router, "ingest_many", lambda **kw: called.append(kw))
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(SystemExit, match=r'pip install "cv-suite\[eda\]"'):
        eda_cli.main([str(tmp_path), "to-report", str(tmp_path / "r")])
    assert not called


def test_help_stays_offline_without_matplotlib() -> None:
    code = "import sys; from cvsuite.eda import cli; cli.main(['--help']); cli.main(['to-report', '--help']); assert 'matplotlib' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "to-report" in result.stdout


def test_transforms_run_before_the_report(tmp_path: Path) -> None:
    src = make_dataset(tmp_path / "ds", n=24, classes=3)
    out = tmp_path / "r"
    code = eda_cli.main([
        str(src), "select", "--drop-class", "c0", "--empty", "drop", "split", "--group-by", r"regex:^(b\d)_", "--val-frac", "0.3", "--stratify-by", "none", "--tolerance", "1",
        "to-report", str(out), "--only", "health,structure",
    ])
    assert code == 0
    rows = list(csv.DictReader((out / "images.csv").open()))
    assert len(rows) < 24 and {r["split"] for r in rows} == {"train", "val"} and not any("c0" in r["class"] for r in rows)
    assert [s["name"] for s in report_yaml(out)["steps"]] == ["select", "split"]


def test_model_backed_transforms_are_not_registered(tmp_path: Path) -> None:
    assert "ground" not in eda_cli.TRANSFORM_COMMANDS and "ocr" not in eda_cli.TRANSFORM_COMMANDS
    assert {"select", "remap-classes", "split", "sample", "filter", "ops"} <= set(eda_cli.TRANSFORM_COMMANDS)
    with pytest.raises(SystemExit, match="model-backed"):
        eda_cli.main([str(tmp_path), "ground", "--provider", "sam3", "--prompt", "x", "to-report", str(tmp_path / "r")])


def test_an_unreadable_image_becomes_a_health_row_not_a_crash(tmp_path: Path) -> None:
    src = make_dataset(tmp_path / "ds", n=6)
    bad = src / "train" / "images" / "b0_0.jpg"
    bad.write_bytes(b"not an image")
    out = tmp_path / "r"
    assert eda_cli.main([str(src), "to-report", str(out), "--only", "health,structure"]) == 0
    health = {r["check"]: r for r in csv.DictReader((out / "health" / "tables" / "health.csv").open())}
    assert health["unreadable_or_zero_size_images"]["count"] == "1" and "b0_0.jpg" in health["unreadable_or_zero_size_images"]["examples"]
    row = next(r for r in csv.DictReader((out / "images.csv").open()) if r["path"] == "b0_0.jpg")
    assert row["width"] == "" and row["phash"] == ""
    assert report_yaml(out)["unreadable_images"][0]["path"] == "b0_0.jpg"


def test_health_checks_find_planted_problems(tmp_path: Path) -> None:
    src = make_dataset(tmp_path / "ds", n=8, splits=("train",))
    (src / "train" / "labels" / "b0_0.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n0 0.95 0.5 0.3 0.2\n1 0.5 0.5 0 0.2\n")
    (src / "train" / "labels" / "b1_1.txt").write_text("")
    shutil.copy(src / "train" / "images" / "b2_2.jpg", src / "train" / "images" / "copy_of_b2_2.jpg")
    out = tmp_path / "r"
    eda_cli.main([str(src), "to-report", str(out), "--only", "health"])
    health = {r["check"]: r for r in csv.DictReader((out / "health" / "tables" / "health.csv").open())}
    assert int(health["duplicate_boxes"]["count"]) == 1
    assert int(health["boxes_outside_image"]["count"]) == 1
    assert int(health["zero_area_boxes"]["count"]) == 1
    assert int(health["images_without_annotations"]["count"]) >= 2
    assert health["exact_duplicate_files"]["count"] == "2"


def test_polygon_derived_boxes_count_once(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    Image.fromarray((np.random.default_rng(1).random((40, 40, 3)) * 255).astype("uint8")).save(root / "train" / "images" / "a.jpg")
    (root / "train" / "labels" / "a.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    (root / "data.yaml").write_text("names: [a]\ntask: segment\n")
    out = tmp_path / "r"
    eda_cli.main([str(root), "to-report", str(out), "--only", "structure"])
    rows = list(csv.DictReader((out / "structure" / "tables" / "structure.csv").open()))
    assert rows[0]["instances"] == "1"
    assert next(csv.DictReader((out / "images.csv").open()))["n_instances"] == "1"


def test_classification_datasets_skip_box_figures(tmp_path: Path) -> None:
    root = tmp_path / "cls"
    for label in ("cat", "dog"):
        (root / label).mkdir(parents=True)
        for i in range(4):
            Image.fromarray((np.random.default_rng(i).random((32, 32, 3)) * 255).astype("uint8")).save(root / label / f"{label}{i}.jpg")
    out = tmp_path / "r"
    assert eda_cli.main([str(root), "--from", "class-dir", "to-report", str(out)]) == 0
    assert (out / "structure" / "figures" / "01_class_counts.png").is_file()
    assert not (out / "structure" / "figures" / "04_box_size.png").exists()
    assert (out / "profile" / "figures" / "01_contact_sheet.jpg").is_file()
    assert any("no boxes" in o["reason"] for o in report_yaml(out)["figures_not_drawn"])


def test_eda_accepts_images_dir_for_coco(tmp_path: Path) -> None:
    from cvsuite.label.tests.test_io_edge_fixes import coco_elsewhere

    ann, pics = coco_elsewhere(tmp_path)
    assert eda_cli.main([str(ann), "--from", "coco", "--images-dir", str(pics), "to-report", str(tmp_path / "rep"), "--only", "structure"]) == 0
    assert (tmp_path / "rep" / "report.yaml").is_file()
