from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from cvsuite.common.core import BBox, ImageRecord, Record
from cvsuite.label.core.annotate import overlay
from cvsuite.label.core.annotate import preview


def test_prepare_qt_environment_replaces_missing_font_dir(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-fonts"
    valid = tmp_path / "fonts"
    valid.mkdir()

    monkeypatch.setenv("QT_QPA_FONTDIR", str(missing))
    monkeypatch.setattr(preview, "_QT_FONT_CANDIDATES", (valid,))

    preview._prepare_qt_environment()

    assert os.environ["QT_QPA_FONTDIR"] == str(valid)


def test_prepare_qt_environment_unsets_missing_font_dir_without_fallback(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-fonts"
    monkeypatch.setenv("QT_QPA_FONTDIR", str(missing))
    monkeypatch.setattr(preview, "_QT_FONT_CANDIDATES", ())

    preview._prepare_qt_environment()

    assert "QT_QPA_FONTDIR" not in os.environ


def test_render_overlay_accepts_prompt_and_confidence_flags(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 24), color=(255, 255, 255)).save(image_path)
    record = Record(
        image=ImageRecord(path=image_path, width=32, height=24),
        boxes=[
            BBox(
                cx=0.5,
                cy=0.5,
                w=0.4,
                h=0.4,
                cls=0,
                label="ground",
                prompt="rust patch",
                score=0.87,
            )
        ],
    )

    canvas = overlay.render_overlay(record, ["ground"], show_confidence=True, show_prompt=True)

    assert canvas is not None
    assert canvas.shape[:2] == (24, 32)
