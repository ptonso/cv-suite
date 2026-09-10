from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from cvsuite.prep.core.fs import iter_files, require_image_source


def _img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)


def test_iter_files_yields_sorted_pre_order(tmp_path: Path) -> None:
    _img(tmp_path / "c.jpg")
    _img(tmp_path / "a.jpg")
    _img(tmp_path / "b" / "z.jpg")
    _img(tmp_path / "b" / "a.jpg")
    _img(tmp_path / "a_dir" / "x.jpg")

    got = [p.relative_to(tmp_path).as_posix() for p in iter_files(tmp_path)]
    # deterministic pre-order: a directory's own files (sorted) before its
    # sub-directories (sorted), regardless of filesystem order
    assert got == ["a.jpg", "c.jpg", "a_dir/x.jpg", "b/a.jpg", "b/z.jpg"]
    assert list(iter_files(tmp_path)) == list(iter_files(tmp_path))


def test_require_image_source_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not found"):
        require_image_source(tmp_path / "nope")


def test_require_image_source_not_a_dir(tmp_path: Path) -> None:
    f = tmp_path / "f.jpg"
    _img(f)
    with pytest.raises(SystemExit, match="must be a directory"):
        require_image_source(f)


def test_require_image_source_no_images(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(SystemExit, match="No images found"):
        require_image_source(tmp_path)


def test_require_image_source_accepts_a_dir_with_images(tmp_path: Path) -> None:
    _img(tmp_path / "a.jpg")
    require_image_source(tmp_path)  # no raise
