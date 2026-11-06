"""Chained transforms: stage splitting, model-backed rule, steps and cleanup (`specs/pipeline.md#chaining`)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from cvsuite.branch_cli import split_rest
from cvsuite.classify import cli as classify_cli
from cvsuite.label import cli as label_cli
from cvsuite.vlm import cli as vlm_cli


def label_split(rest: list[str], **kw):
    transforms = {name: (lambda name=name: label_cli.transform_module(name)) for name in label_cli.TRANSFORM_COMMANDS}
    return split_rest(rest, transforms, label_cli.COMMANDS, prog="cvsuite label", **kw)


def test_two_transforms_then_output() -> None:
    steps, command, args = label_split(["sample", "--count", "3", "split", "--val-frac", "0.2", "to-yolo", "./out"])
    assert [s.name for s in steps] == ["sample", "split"]
    assert steps[0].args.count == 3 and steps[1].args.val_frac == 0.2
    assert command == "to-yolo" and args == ["./out"]


def test_transforms_may_share_a_flag_name() -> None:
    steps, _, _ = label_split(["sample", "--count", "3", "--seed", "1", "split", "--val-frac", "0.2", "--seed", "2", "to-yolo", "o"])
    assert (steps[0].args.seed, steps[1].args.seed) == (1, 2)


def test_a_command_name_consumed_as_a_value_stays_an_argument() -> None:
    steps, command, args = label_split(["ground", "--provider", "sam3", "--prompt", "sample", "to-yolo", "./out"])
    assert [s.name for s in steps] == ["ground"]
    assert steps[0].args.prompt == "sample"
    assert command == "to-yolo" and args == ["./out"]


def test_no_output_command_returns_none() -> None:
    steps, command, args = label_split(["sample", "--count", "1"])
    assert [s.name for s in steps] == ["sample"] and command is None and args == []


def test_unknown_token_is_named() -> None:
    with pytest.raises(SystemExit, match="'bogus'"):
        label_split(["bogus", "to-yolo", "o"])


def test_a_stage_no_cut_satisfies_names_where_it_started() -> None:
    with pytest.raises(SystemExit, match="'sample'.*--frac"):
        label_split(["sample", "--frac", "abc", "to-yolo", "o"])


def test_model_backed_transform_must_come_first_and_alone() -> None:
    with pytest.raises(SystemExit, match="must come first"):
        label_split(["sample", "--count", "1", "ground", "--provider", "sam3", "--prompt", "x", "to-yolo", "o"])
    with pytest.raises(SystemExit, match="at most one model-backed"):
        label_split(["ground", "--provider", "sam3", "--prompt", "x", "ocr", "--provider", "paddleocr", "to-yolo", "o"])
    steps, _, _ = label_split(["ground", "--provider", "sam3", "--prompt", "x", "split", "--val-frac", "0.2", "to-yolo", "o"])
    assert [s.name for s in steps] == ["ground", "split"]


def test_max_transforms_is_enforced() -> None:
    with pytest.raises(SystemExit, match="at most 1"):
        label_split(["sample", "--count", "1", "split", "--val-frac", "0.2", "to-yolo", "o"], max_transforms=1)


def test_vlm_rejects_two_transforms(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="at most 1"):
        vlm_cli.main([str(tmp_path), "caption", "--provider", "qwen", "caption", "--provider", "qwen", "to-json", str(tmp_path / "o")])


def write_yolo(src: Path) -> None:
    (src / "train" / "images").mkdir(parents=True)
    (src / "train" / "labels").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(src / "train" / "images" / "a.jpg")
    (src / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (src / "data.yaml").write_text("names:\n  0: obj\n")


def image_dir(root: Path, names: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        Image.new("RGB", (16, 16), (i * 20 % 255, 10, 10)).save(root / name)
    return root


def test_chain_records_steps_in_stats(tmp_path: Path) -> None:
    src = image_dir(tmp_path / "src", [f"a_{i}.jpg" for i in range(10)])
    out = tmp_path / "out"
    code = label_cli.main([
        str(src), "sample", "--frac", "1.0", "split", "--val-frac", "0.3", "--stratify-by", "none", "--tolerance", "1",
        "to-yolo", str(out), "--with-stats",
    ])
    assert code == 0
    stats = yaml.safe_load((out / "stats.yaml").read_text())
    assert [s["name"] for s in stats["steps"]] == ["sample", "split"]
    assert stats["steps"][0]["info"]["sample_size"] == 10
    assert stats["steps"][1]["info"]["mode"] == "ratio"
    assert stats["steps"][1]["args"]["val_frac"] == 0.3
    assert len(list((out / "val" / "images").glob("*"))) == 3


def test_class_chain_runs_sample_then_split(tmp_path: Path) -> None:
    src = tmp_path / "src"
    for label in ("cat", "dog"):
        image_dir(src / label, [f"{label}_{i}.jpg" for i in range(10)])
    out = tmp_path / "out"
    code = classify_cli.main([str(src), "sample", "--mode", "preserve", "--max-n-per-class", "8", "split", "--val-frac", "0.25", "to-class-dir", str(out)])
    assert code == 0
    assert len(list((out / "val" / "cat").glob("*.jpg"))) == 2
    assert len(list((out / "train" / "dog").glob("*.jpg"))) == 6


def test_temp_folders_are_removed_after_the_run(tmp_path: Path, monkeypatch) -> None:
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.setattr("tempfile.tempdir", None)
    src = tmp_path / "src"
    write_yolo(src)
    out = tmp_path / "out"
    assert label_cli.main([str(src), "ops", "crop-dets", "--imgsz", "16", "to-yolo", str(out)]) == 0
    assert list(out.rglob("*.jpg"))
    assert not [p for p in tmp.iterdir() if p.name.startswith("vislabel_")]


def test_temp_folders_are_removed_when_the_run_fails(tmp_path: Path, monkeypatch) -> None:
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.setattr("tempfile.tempdir", None)
    src = tmp_path / "src"
    write_yolo(src)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        label_cli.main([str(src), "ops", "crop-dets", "--imgsz", "16", "to-labelme", str(out)])
    assert not [p for p in tmp.iterdir() if p.name.startswith("vislabel_")]
