from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from cvsuite.prep.cli import main


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _blocks(size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(7)
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    step = size // 4
    for i in range(4):
        for j in range(4):
            arr[i * step:(i + 1) * step, j * step:(j + 1) * step] = rng.integers(0, 256, 3)
    return arr


def _checkerboard(size: int = 128, tile: int = 16) -> np.ndarray:
    yy, xx = np.indices((size, size))
    mask = ((yy // tile) + (xx // tile)) % 2
    arr = np.where(mask[..., None] == 0, 0, 255).astype(np.uint8)
    return np.repeat(arr, 3, axis=-1)


def _save(path: Path, img: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def test_arrange_dedups_against_existing_dst_and_writes_manifest(tmp_path: Path) -> None:
    src = tmp_path / "raw"
    dst = tmp_path / "dst"
    _write_image(dst / "000001.jpg", (255, 0, 0))
    _write_image(src / "duplicate.jpg", (255, 0, 0))
    _write_image(src / "new.jpg", (0, 255, 0))

    exit_code = main(["arrange", str(src), str(dst), "--exact-dedup", "--rename-seq"])

    assert exit_code == 0
    assert (dst / "000001.jpg").exists()
    assert (dst / "000002.jpg").exists()
    assert not (dst / "000003.jpg").exists()
    manifest = yaml.safe_load((dst / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["command"] == "arrange"
    assert manifest["src"] == str(src)
    assert manifest["dst"] == str(dst)
    assert manifest["options"]["dedup_mode"] == "exact"
    assert manifest["options"]["rename_seq"] is True
    assert manifest["stats"]["skipped_duplicates"] == 1


def test_arrange_rename_seq_continues_after_existing_dst_sequence(tmp_path: Path) -> None:
    src = tmp_path / "incoming"
    dst = tmp_path / "dst"
    _write_image(dst / "000007.jpg", (255, 0, 0))
    _write_image(src / "new.jpg", (0, 255, 0))

    exit_code = main(["arrange", str(src), str(dst), "--rename-seq", "--no-manifest"])

    assert exit_code == 0
    assert (dst / "000008.jpg").exists()
    assert not (dst / "000001.jpg").exists()
    assert not (dst / "manifest.yaml").exists()


def test_arrange_dst_subdir_uses_requested_name_after_sequence(tmp_path: Path) -> None:
    src = tmp_path / "batch_a"
    dst = tmp_path / "dst"
    _write_image(dst / "000007.jpg", (255, 0, 0))
    _write_image(src / "new.jpg", (0, 255, 0))

    exit_code = main([
        "arrange",
        str(src),
        str(dst),
        "--rename-seq",
        "--dst-subdir",
        "review",
        "--no-manifest",
    ])

    assert exit_code == 0
    assert (dst / "review" / "000008.jpg").exists()
    assert not (dst / "batch_a" / "000008.jpg").exists()
    assert not (dst / "000008.jpg").exists()


def test_arrange_rename_seq_preserves_dirs_without_flatten(tmp_path: Path) -> None:
    src = tmp_path / "incoming"
    dst = tmp_path / "dst"
    _write_image(src / "class_a" / "first.jpg", (255, 0, 0))
    _write_image(src / "class_b" / "second.jpg", (0, 255, 0))

    exit_code = main([
        "arrange",
        str(src),
        str(dst),
        "--rename-seq",
        "--dst-subdir",
        "new",
        "--no-manifest",
    ])

    assert exit_code == 0
    assert (dst / "new" / "class_a" / "000001.jpg").exists()
    assert (dst / "new" / "class_b" / "000002.jpg").exists()
    assert not (dst / "new" / "000001.jpg").exists()


def test_arrange_dst_subdir_dedups_against_dst_root(tmp_path: Path) -> None:
    src = tmp_path / "batch_b"
    dst = tmp_path / "dst"
    _write_image(dst / "000001.jpg", (255, 0, 0))
    _write_image(src / "duplicate.jpg", (255, 0, 0))
    _write_image(src / "unique.jpg", (0, 0, 255))

    exit_code = main([
        "arrange",
        str(src),
        str(dst),
        "--exact-dedup",
        "--rename-seq",
        "--dst-subdir",
        "new-data",
    ])

    assert exit_code == 0
    assert (dst / "new-data" / "000002.jpg").exists()
    assert len(list((dst / "new-data").glob("*.jpg"))) == 1
    assert (dst / "new-data" / "manifest.yaml").exists()
    assert not (dst / "manifest.yaml").exists()


def test_arrange_perceptual_dedup_marks_near_duplicates(tmp_path: Path) -> None:
    src = tmp_path / "raw"
    dst = tmp_path / "dst"
    base = Image.fromarray(_blocks())
    _save(dst / "base.png", base)
    _save(src / "near.png", base.resize((100, 100)).resize((140, 140)))
    _save(src / "distinct.png", Image.fromarray(_checkerboard()))
    (src / "notes.txt").parent.mkdir(parents=True, exist_ok=True)
    (src / "notes.txt").write_text("hello", encoding="utf-8")

    exit_code = main([
        "arrange",
        str(src),
        str(dst),
        "--perceptual-dedup",
        "12",
        "--non-image",
        "keep",
        "--no-manifest",
    ])

    assert exit_code == 0
    assert not (dst / "near.png").exists()
    assert (dst / "distinct.png").exists()
    assert (dst / "notes.txt").exists()


def test_arrange_dst_subdir_rejects_paths(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_image(src / "new.jpg", (0, 255, 0))

    with pytest.raises(SystemExit, match="single folder name"):
        main(["arrange", str(src), str(tmp_path / "dst"), "--dst-subdir", "nested/name"])


def test_arrange_dry_run_does_not_write_manifest(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_image(src / "new.jpg", (0, 255, 0))

    exit_code = main(["arrange", str(src), str(dst), "--rename-seq", "--dry-run"])

    assert exit_code == 0
    assert not dst.exists()


def test_process_writes_manifest_and_no_manifest_suppresses_it(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    quiet_dst = tmp_path / "quiet"
    dry_dst = tmp_path / "dry"
    _write_image(src / "image.jpg", (0, 255, 0))

    assert main(["process", str(src), str(dst), "--size", "8", "--to-format", "png"]) == 0
    manifest = yaml.safe_load((dst / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["command"] == "process"
    assert manifest["options"]["resize_mode"] == "long"
    assert manifest["options"]["size"] == 8
    assert manifest["options"]["to_format"] == "png"

    assert main(["process", str(src), str(quiet_dst), "--size", "8", "--no-manifest"]) == 0
    assert not (quiet_dst / "manifest.yaml").exists()

    assert main(["process", str(src), str(dry_dst), "--size", "8", "--dry-run"]) == 0
    assert not dry_dst.exists()


def test_process_opencv_caps_long_side_and_records_encoder_options(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _save(src / "large.jpg", Image.fromarray(_blocks(128)))
    _save(src / "small.jpg", Image.fromarray(_blocks(16)))

    assert main([
        "process",
        str(src),
        str(dst),
        "--size",
        "64",
        "--resize-mode",
        "cap-long",
        "--backend",
        "opencv",
        "--jpeg-quality",
        "100",
    ]) == 0

    with Image.open(dst / "large.jpg") as large:
        assert large.size == (64, 64)
    with Image.open(dst / "small.jpg") as small:
        assert small.size == (16, 16)
    manifest = yaml.safe_load((dst / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["options"]["backend"] == "opencv"
    assert manifest["options"]["jpeg_quality"] == 100
