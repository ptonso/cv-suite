"""`--link-mode copy|hard|sym`: one `copy_or_link`, no fallback (`specs/pipeline.md#link-mode`)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest
from PIL import Image

import cvsuite.label.output.commands.to_coco as cmd_to_coco
import cvsuite.label.output.commands.to_yolo as cmd_to_yolo
from cvsuite.common.core import BBox, Classification, ImageInfo, VisionDataset, VisionRecord
from cvsuite.common.io import write_dataset
from cvsuite.common.io import common


def source(tmp_path: Path) -> Path:
    path = tmp_path / "src" / "a.jpg"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path)
    return path


def dataset(path: Path, **meta) -> VisionDataset:
    record = VisionRecord(image=ImageInfo(path, 8, 8), boxes=[BBox(0.5, 0.5, 0.2, 0.2, 0, "a")])
    return VisionDataset(records=[record], classes=["a"], root=path.parent, meta=meta)


def test_each_mode_places_the_file(tmp_path: Path) -> None:
    src = source(tmp_path)
    common.copy_or_link(src, tmp_path / "c.jpg", link_mode="copy")
    common.copy_or_link(src, tmp_path / "h.jpg", link_mode="hard")
    common.copy_or_link(src, tmp_path / "s.jpg", link_mode="sym")
    assert not os.path.samefile(src, tmp_path / "c.jpg") and not (tmp_path / "c.jpg").is_symlink()
    assert os.path.samefile(src, tmp_path / "h.jpg") and not (tmp_path / "h.jpg").is_symlink()
    assert (tmp_path / "s.jpg").is_symlink() and (tmp_path / "s.jpg").resolve() == src.resolve()
    assert os.readlink(tmp_path / "s.jpg") == str(src.resolve())


def test_a_failed_hard_link_raises_instead_of_copying(tmp_path: Path, monkeypatch) -> None:
    src = source(tmp_path)

    def fail(*_args):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(OSError, match="hard .*a.jpg"):
        common.copy_or_link(src, tmp_path / "h.jpg", link_mode="hard")
    assert not (tmp_path / "h.jpg").exists()


def test_existing_destination_and_unknown_mode_raise(tmp_path: Path) -> None:
    src = source(tmp_path)
    common.copy_or_link(src, tmp_path / "x.jpg")
    with pytest.raises(FileExistsError):
        common.copy_or_link(src, tmp_path / "x.jpg")
    with pytest.raises(ValueError, match="link mode"):
        common.copy_or_link(src, tmp_path / "y.jpg", link_mode="move")


@pytest.mark.parametrize("fmt", ["yolo", "coco", "labelme", "class-dir"])
@pytest.mark.parametrize("mode", ["hard", "sym"])
def test_every_writer_honours_the_link_mode(tmp_path: Path, fmt: str, mode: str) -> None:
    src = source(tmp_path)
    ds = dataset(src)
    ds.records[0].classification = Classification(label="a")
    dst = tmp_path / "out"
    write_dataset(ds, dst, format=fmt, link_mode=mode)
    images = [p for p in dst.rglob("a*.jpg")]
    assert images
    assert all(p.is_symlink() == (mode == "sym") for p in images)
    assert all(os.path.samefile(p, src) for p in images)


def test_sym_of_a_cleanup_folder_image_raises(tmp_path: Path) -> None:
    src = source(tmp_path)
    ds = dataset(src, cleanup_dirs=[str(src.parent)])
    with pytest.raises(ValueError, match="temporary folder"):
        write_dataset(ds, tmp_path / "out", format="yolo", link_mode="sym")
    write_dataset(ds, tmp_path / "out2", format="yolo", link_mode="hard")


def test_output_commands_accept_the_flag_and_reject_hardlink() -> None:
    for module in (cmd_to_yolo, cmd_to_coco):
        parser = argparse.ArgumentParser()
        module.attach(parser)
        assert parser.parse_args(["out", "--link-mode", "sym"]).link_mode == "sym"
        assert parser.parse_args(["out"]).link_mode == "copy"
        with pytest.raises(SystemExit):
            parser.parse_args(["out", "--hardlink"])
        with pytest.raises(SystemExit):
            parser.parse_args(["out", "--val-frac", "0.2"])
