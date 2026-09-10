from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from PIL import Image
import pytest
import yaml

from cvsuite.common.core import BBox, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.label.transform.filter.commands import apply as apply_cmd
from cvsuite.label.transform.filter.commands import logistic_opt as train_logistic_cmd


def _write_image(path: Path, *, color: tuple[int, int, int] = (255, 255, 255)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=color).save(path)
    return path


def _dataset(tmp_path: Path, *, score: float = 0.4) -> VisionDataset:
    image_path = _write_image(tmp_path / "img.jpg")
    return VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=20, height=20),
                boxes=[BBox(cx=0.5, cy=0.5, w=0.4, h=0.4, cls=0, label="dog", score=score)],
            )
        ],
        classes=["dog"],
        root=tmp_path,
    )


def test_filter_apply_rejects_mixed_config_and_inline_thresholds(tmp_path: Path) -> None:
    cfg_path = tmp_path / "filter.yaml"
    cfg_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="either --config or inline threshold flags"):
        apply_cmd.run(
            _dataset(tmp_path),
            Namespace(
                config=cfg_path,
                det_conf=0.2,
                seg_conf=None,
                nms_iou=None,
                cross_class_iou=None,
                class_det_conf=[],
                class_seg_conf=[],
                with_rgb=False,
            ),
        )


def test_filter_apply_requires_config_or_inline_thresholds(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="requires --config or at least one inline threshold flag"):
        apply_cmd.run(
            _dataset(tmp_path),
            Namespace(
                config=None,
                det_conf=None,
                seg_conf=None,
                nms_iou=None,
                cross_class_iou=None,
                class_det_conf=[],
                class_seg_conf=[],
                with_rgb=False,
            ),
        )


def test_filter_apply_inline_threshold_filters_dataset(tmp_path: Path) -> None:
    result = apply_cmd.run(
        _dataset(tmp_path, score=0.4),
        Namespace(
            config=None,
            det_conf=0.5,
            seg_conf=None,
            nms_iou=None,
            cross_class_iou=None,
            class_det_conf=[],
            class_seg_conf=[],
            with_rgb=False,
        ),
    )

    assert len(result.records) == 1
    assert result.records[0].boxes == []


def test_filter_apply_uses_logistic_filter_for_logistic_config(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "logistic.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "global": {"det_conf": 0.1},
                "per_class": {
                    "dog": {
                        "logistic_gate": {
                            "features": ["score"],
                            "weights": [1.0],
                            "bias": 0.0,
                            "threshold_type": "logit",
                            "threshold_value": 0.0,
                        }
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    class _DummyThresholdFilter:
        def __init__(self, cfg) -> None:
            calls.append("threshold")

        def filter_record(self, rec, classes, rgb=None):
            return rec

    class _DummyLogisticFilter:
        def __init__(self, cfg) -> None:
            calls.append("logistic")

        def filter_record(self, rec, classes, rgb=None):
            return rec

    monkeypatch.setattr(apply_cmd, "ThresholdFilter", _DummyThresholdFilter)
    monkeypatch.setattr(apply_cmd, "LogisticFilter", _DummyLogisticFilter)

    result = apply_cmd.run(
        _dataset(tmp_path),
        Namespace(
            config=cfg_path,
            det_conf=None,
            seg_conf=None,
            nms_iou=None,
            cross_class_iou=None,
            class_det_conf=[],
            class_seg_conf=[],
            with_rgb=False,
        ),
    )

    assert len(result.records) == 1
    assert calls == ["logistic"]


def test_filter_train_logistic_writes_results_and_returns_filtered_dataset(monkeypatch, tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    gt_path = tmp_path / "gt.yaml"
    out_cfg = tmp_path / "learned.yaml"
    captured = {}

    class _DummyLogisticFilter:
        def __init__(self, cfg) -> None:
            captured["cfg"] = cfg

        def filter_record(self, rec, classes, rgb=None):
            captured["filter_called"] = True
            return rec

    class _FakeOptimizer:
        def __init__(self, cfg) -> None:
            captured["optimizer_cfg"] = cfg

        def run(self, gt_ds, permissive_ds, results) -> int:
            captured["gt_ds"] = gt_ds
            captured["permissive_ds"] = permissive_ds
            Path(self.cfg.out_cfg).write_text("{}", encoding="utf-8")
            return 0

        @property
        def cfg(self):
            return captured["optimizer_cfg"]

    monkeypatch.setattr(train_logistic_cmd.router, "ingest", lambda src, from_hint="auto": VisionDataset(records=[]))
    monkeypatch.setattr(train_logistic_cmd, "LogisticOptimizer", _FakeOptimizer)
    monkeypatch.setattr(train_logistic_cmd, "LogisticFilter", _DummyLogisticFilter)

    result = train_logistic_cmd.run(
        dataset,
        Namespace(
            gt_src=gt_path,
            config=None,
            out_config=out_cfg,
            match_iou=0.5,
            epochs=5,
            lr=0.1,
            l2=1e-3,
            alpha=0.05,
            max_prune_iters=2,
            min_features=1,
            always_keep="score",
            seed=7,
            device="cpu",
            beta=2.0,
            conf_metric="fbeta",
            val_split=0.2,
            tune_nms_iou=False,
            nms_grid="0.2,0.3",
            tune_cross_class_iou=False,
            cross_grid="0.1,0.2",
            with_rgb=False,
            log_dir=None,
            target_dir=tmp_path,
        ),
    )

    assert len(result.records) == 1
    assert captured["gt_ds"].records == []
    assert captured["permissive_ds"] is dataset
    assert captured["filter_called"] is True
    assert out_cfg.exists()
    assert (tmp_path / "filter_results.yaml").exists()
