from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from cvsuite.common.core import Classification, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.prep.cli import main
from cvsuite.prep.core.orient.deep_orientation import (
    OrientationPrediction,
    predict_dataset,
)


def _make_pattern_image(path: Path, *, fmt: str, exif_orientation: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (4, 2))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((3, 0), (0, 255, 0))
    image.putpixel((0, 1), (0, 0, 255))
    image.putpixel((3, 1), (255, 255, 0))
    save_kwargs: dict[str, object] = {"format": fmt}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation
        save_kwargs["exif"] = exif.tobytes()
    image.save(path, **save_kwargs)
    image.close()


def test_orient_uses_exif_and_resets_orientation(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_pattern_image(src / "rotated.jpg", fmt="JPEG", exif_orientation=6)

    exit_code = main(["orient", str(src), str(dst)])

    assert exit_code == 0
    out_path = dst / "rotated.jpg"
    assert out_path.exists()
    with Image.open(out_path) as output:
        assert output.size == (2, 4)
        assert output.getexif().get(274) == 1


def test_orient_try_hardlink_for_unchanged_exif_image(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src_path = src / "upright.jpg"
    _make_pattern_image(src_path, fmt="JPEG", exif_orientation=1)

    main(["orient", str(src), str(dst), "--try-hardlink"])

    out_path = dst / "upright.jpg"
    assert out_path.exists()
    assert src_path.stat().st_ino == out_path.stat().st_ino


def test_orient_uses_model_fallback_with_stubbed_predictor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    upright = src / "upright.jpg"
    first = src / "first.png"
    second = src / "second.png"
    _make_pattern_image(upright, fmt="JPEG", exif_orientation=1)
    _make_pattern_image(first, fmt="PNG")
    _make_pattern_image(second, fmt="PNG")

    calls: list[tuple[int, int]] = []

    def _fake_predict_dataset(dataset, *, device_hint, precision, batch_size, weights_dir):
        calls.append((len(dataset.records), batch_size))
        # iter_files yields sorted order: first.png, second.png, upright.jpg
        return [
            OrientationPrediction(class_index=1, ccw_degrees=270, scores=(0.01, 0.96, 0.02, 0.01)),
            OrientationPrediction(class_index=1, ccw_degrees=270, scores=(0.01, 0.96, 0.02, 0.01)),
            OrientationPrediction(class_index=0, ccw_degrees=0, scores=(0.9, 0.05, 0.03, 0.02)),
        ]

    monkeypatch.setattr(
        "cvsuite.prep.core.orient.core.predict_dataset",
        _fake_predict_dataset,
    )

    main(["orient", str(src), str(dst), "--batch", "2"])

    assert calls == [(3, 2)]
    assert (dst / "upright.jpg").exists()
    for out_path in (dst / "first.png", dst / "second.png"):
        with Image.open(out_path) as output:
            assert output.size == (2, 4)
            assert output.getpixel((0, 0)) == (0, 0, 255)


def test_orient_rejects_invalid_batch_size(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    with pytest.raises(SystemExit, match="--batch must be >= 1."):
        main(["orient", str(src), str(dst), "--batch", "0"])


def test_predict_dataset_routes_through_common_fm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class _Runner:
        def run(self, dataset, *, no_resume: bool = False):
            captured["dataset"] = dataset
            captured["no_resume"] = no_resume
            for idx, rec in enumerate(dataset.records):
                rec.classification = Classification(
                    label="rotate_90_clockwise" if idx == 0 else "correct",
                    score=0.9,
                    probs={
                        "correct": 0.1 if idx == 0 else 0.9,
                        "rotate_90_clockwise": 0.9 if idx == 0 else 0.05,
                        "rotate_180": 0.0,
                        "rotate_90_counter_clockwise": 0.0,
                    },
                    meta={
                        "class_index": 1 if idx == 0 else 0,
                        "correction_degrees_ccw": 270 if idx == 0 else 0,
                    },
                )
            return dataset

    monkeypatch.setattr(
        "cvsuite.prep.core.orient.deep_orientation.FMRunner.from_dataset",
        lambda dataset: _Runner(),
    )

    dataset = VisionDataset(
        records=[
            Record(image=ImageRecord(path=tmp_path / "first.png", width=4, height=2)),
            Record(image=ImageRecord(path=tmp_path / "second.png", width=4, height=2)),
        ],
        root=tmp_path,
    )
    predictions = predict_dataset(
        dataset,
        device_hint="cpu",
        precision="fp32",
        batch_size=8,
        weights_dir=tmp_path / "weights",
        no_resume=True,
    )

    dataset = captured["dataset"]
    assert dataset.fm_request is not None
    assert dataset.fm_request.task == "classify"
    assert dataset.fm_request.provider == "deep_orientation"
    assert dataset.fm_request.device == "cpu"
    assert dataset.fm_request.precision == "fp32"
    assert dataset.fm_request.batch_size == 8
    assert captured["no_resume"] is True
    assert [prediction.ccw_degrees for prediction in predictions] == [270, 0]
