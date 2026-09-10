from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from cvsuite.prep import cli as prep_cli


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def test_prep_sample_pools_multiple_flat_folders_into_flat_output(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    dst = tmp_path / "out"
    _write_image(left / "same.jpg", color=(255, 0, 0))
    _write_image(right / "same.jpg", color=(0, 255, 0))
    _write_image(right / "other.jpg", color=(0, 0, 255))

    exit_code = prep_cli.main(["sample", str(left), str(right), str(dst), "--count", "3", "--seed", "0"])

    assert exit_code == 0
    assert (dst / "same.jpg").exists()
    assert (dst / "same_1.jpg").exists()
    assert (dst / "other.jpg").exists()
    assert len(list(dst.glob("*.jpg"))) == 3


def test_prep_sample_rejects_nested_directories(tmp_path: Path) -> None:
    src = tmp_path / "nested"
    src.mkdir()
    _write_image(src / "a.jpg", color=(255, 255, 255))
    _write_image(src / "child" / "b.jpg", color=(0, 0, 0))

    with pytest.raises(SystemExit, match="flat raw image folder"):
        prep_cli.main(["sample", str(src), str(tmp_path / "out"), "--count", "1"])


def test_prep_sample_rejects_annotated_sources(tmp_path: Path) -> None:
    src = tmp_path / "labelme"
    src.mkdir()
    _write_image(src / "scene.jpg", color=(255, 255, 255))
    (src / "scene.json").write_text(
        json.dumps(
            {
                "imagePath": "scene.jpg",
                "imageHeight": 16,
                "imageWidth": 16,
                "shapes": [{"label": "thing", "shape_type": "rectangle", "points": [[1, 1], [5, 5]]}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="cvsuite label sample"):
        prep_cli.main(["sample", str(src), str(tmp_path / "out"), "--count", "1"])


def test_prep_sample_dry_run_reports_planned_actions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "out"
    _write_image(src / "a.jpg", color=(255, 0, 0))
    _write_image(src / "b.jpg", color=(0, 255, 0))

    exit_code = prep_cli.main(["sample", str(src), str(dst), "--frac", "0.5", "--seed", "1", "--dry-run"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Planned actions: 1" in out
    assert not dst.exists()


def test_prep_sample_ignores_prep_artifacts(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "out"
    _write_image(src / "a.jpg", color=(255, 0, 0))
    _write_image(src / "b.jpg", color=(0, 255, 0))
    (src / "manifest.yaml").write_text("version: 1\n")
    _write_image(src / "_non_images" / "notes.txt.jpg", color=(0, 0, 255))  # any file under the artefact dir

    exit_code = prep_cli.main(["sample", str(src), str(dst), "--count", "2", "--seed", "0"])

    assert exit_code == 0
    assert {p.name for p in dst.iterdir()} == {"a.jpg", "b.jpg"}


def test_prep_sample_still_rejects_a_real_nested_image_dir(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_image(src / "a.jpg", color=(255, 0, 0))
    _write_image(src / "class_b" / "b.jpg", color=(0, 255, 0))

    with pytest.raises(SystemExit):
        prep_cli.main(["sample", str(src), str(tmp_path / "out"), "--count", "1"])


def test_prep_sample_hardlinks_when_requested(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "out"
    original = src / "a.jpg"
    _write_image(original, color=(255, 0, 0))

    exit_code = prep_cli.main(["sample", str(src), str(dst), "--count", "1", "--hardlink"])

    sampled = dst / "a.jpg"
    assert exit_code == 0
    assert sampled.exists()
    assert os.stat(sampled).st_ino == os.stat(original).st_ino
