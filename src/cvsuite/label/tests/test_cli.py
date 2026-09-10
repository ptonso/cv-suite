from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from cvsuite.label import cli as label_cli


@pytest.mark.parametrize(
    "argv, needle",
    [
        (["./x", "to-coco", "--help"], "cvsuite label to-coco"),
        (["./x", "sample", "--help"], "cvsuite label sample"),
        (["to-coco", "--help"], "cvsuite label to-coco"),
        (["./x", "sample", "--count", "3", "to-coco", "./o", "--help"], "cvsuite label to-coco"),
    ],
)
def test_label_cli_help_routes_to_specific_parser(capsys, argv, needle) -> None:
    assert label_cli.main(argv) == 0
    assert needle in capsys.readouterr().out


def test_label_cli_help_without_command_shows_branch_usage(capsys) -> None:
    assert label_cli.main(["--help"]) == 0
    assert "usage: cvsuite label <src>" in capsys.readouterr().out


def test_label_cli_to_results_max_flag_is_not_swallowed_by_max_depth(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    src.mkdir()
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(src / "a.jpg")
    captured = {}

    def _fake_ingest_many(srcs, from_hint="auto", keypoints_file=None, max_depth=-1):
        captured["max_depth"] = max_depth
        from cvsuite.common.core import VisionDataset

        return VisionDataset(records=[], classes=[])

    def _fake_to_results_run(dataset, args):
        captured["max"] = args.max
        return dataset

    monkeypatch.setattr("cvsuite.label.cli.router.ingest_many", _fake_ingest_many)
    monkeypatch.setattr("cvsuite.label.output.commands.to_results.run", _fake_to_results_run)

    assert label_cli.main([str(src), "--max-depth", "2", "to-results", str(tmp_path / "out"), "--max", "3"]) == 0
    assert captured["max"] == 3
    assert captured["max_depth"] == 2


def test_label_cli_accepts_dataset_return_from_transform_and_output(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    src.mkdir()
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(src / "a.jpg")
    audit_path = tmp_path / "audit.yaml"

    def _fake_rebox_run(dataset, args):
        return dataset

    monkeypatch.setattr("cvsuite.label.transform.ops.commands.rebox.run", _fake_rebox_run)

    exit_code = label_cli.main(
        [
            str(src),
            "--from",
            "images",
            "ops",
            "rebox",
            "audit",
            str(audit_path),
        ]
    )

    assert exit_code == 0
    assert audit_path.exists()


def test_label_cli_routes_to_ops_crop_dets_flags(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    src.mkdir()
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(src / "a.jpg")
    captured = {}

    def _fake_crop_run(dataset, args):
        captured["dataset"] = dataset
        captured["imgsz"] = args.imgsz
        captured["pad_type"] = args.pad_type
        captured["margin_frac"] = args.margin_frac
        return dataset

    monkeypatch.setattr("cvsuite.label.transform.ops.commands.crop_dets.run", _fake_crop_run)

    exit_code = label_cli.main(
        [
            str(src),
            "--from",
            "images",
            "ops",
            "crop-dets",
            "--imgsz",
            "96",
            "--pad-type",
            "gray",
            "--margin-frac",
            "0.25",
            "audit",
            str(tmp_path / "audit.yaml"),
        ]
    )

    assert exit_code == 0
    assert captured["imgsz"] == 96
    assert captured["pad_type"] == "gray"
    assert captured["margin_frac"] == pytest.approx(0.25)


def test_label_cli_routes_to_ops_inverse_crop_dets(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    src.mkdir()
    crops = tmp_path / "crops"
    crops.mkdir()
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(src / "a.jpg")
    captured = {}

    def _fake_inverse_run(dataset, args):
        captured["dataset"] = dataset
        captured["modified_crops"] = args.modified_crops
        return dataset

    monkeypatch.setattr("cvsuite.label.transform.ops.commands.inverse_crop_dets.run", _fake_inverse_run)

    exit_code = label_cli.main(
        [
            str(src),
            "--from",
            "images",
            "ops",
            "inverse-crop-dets",
            str(crops),
            "audit",
            str(tmp_path / "audit.yaml"),
        ]
    )

    assert exit_code == 0
    assert captured["modified_crops"] == crops


def test_label_cli_rejects_sample_hardlink_for_non_image_outputs() -> None:
    with pytest.raises(SystemExit, match="sample --hardlink"):
        label_cli.main(
            [
                "dataset.yaml",
                "sample",
                "--count",
                "1",
                "--hardlink",
                "audit",
            ]
        )


def test_label_help_hides_removed_filter_and_cache_commands(capsys) -> None:
    assert label_cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "filter" in out
    assert "filter-logistic-opt" not in out
    assert "filter-threshold" not in out
    assert "gsam-filter-labelme" not in out
    assert "gsam-logistic-opt" not in out
    assert "--clean-cache" not in out


def test_label_filter_help_lists_train_logistic_and_apply(capsys) -> None:
    assert label_cli.main(["filter", "--help"]) == 0
    out = capsys.readouterr().out
    assert "train-logistic" in out
    assert "apply" in out


def test_label_ground_help_lists_model_metadata(capsys) -> None:
    assert label_cli.main(["ground", "--help"]) == 0
    out = capsys.readouterr().out
    assert "Local providers:" in out
    assert "  sam3: default-model-id=sam3 params=unknown" in out
    assert "  gsam: default-model-id=gsam params=composite" in out
    assert "  locate_anything: default-model-id=nvidia/LocateAnything-3B params=3B" in out


def test_label_ocr_help_lists_model_metadata(capsys) -> None:
    assert label_cli.main(["ocr", "--help"]) == 0
    out = capsys.readouterr().out
    assert "Local providers:" in out
    assert "  paddleocr: default-model-id=paddleocr params=unknown" in out


def test_label_cli_routes_filter_train_logistic_then_output(monkeypatch, tmp_path: Path) -> None:
    events = []

    def _fake_ingest_many(srcs, from_hint="auto", keypoints_file=None, max_depth=-1):
        events.append(("ingest", srcs, from_hint, keypoints_file, max_depth))
        return {"stage": "ingested"}

    def _fake_train(dataset, args):
        events.append(("train", dataset, args.gt_src, args.out_config, getattr(args, "config", None)))
        return {"stage": "filtered"}

    def _fake_output(dataset, args):
        events.append(("output", dataset, args.dst))
        return 0

    monkeypatch.setattr("cvsuite.label.cli.router.ingest_many", _fake_ingest_many)
    monkeypatch.setattr("cvsuite.label.transform.filter.commands.logistic_opt.run", _fake_train)
    monkeypatch.setattr("cvsuite.label.output.commands.audit.run", _fake_output)

    src = tmp_path / "images"
    gt = tmp_path / "gt.yaml"
    out_cfg = tmp_path / "filter.yaml"
    audit = tmp_path / "audit.yaml"

    exit_code = label_cli.main(
        [
            str(src),
            "--from",
            "images",
            "filter",
            "train-logistic",
            "--gt-src",
            str(gt),
            "--out-config",
            str(out_cfg),
            "audit",
            str(audit),
        ]
    )

    assert exit_code == 0
    assert events[0] == ("ingest", [src], "images", None, -1)
    assert events[1] == ("train", {"stage": "ingested"}, gt, out_cfg, None)
    assert events[2] == ("output", {"stage": "filtered"}, audit)


def test_label_cli_routes_filter_apply_inline_thresholds(monkeypatch, tmp_path: Path) -> None:
    events = []

    def _fake_ingest_many(srcs, from_hint="auto", keypoints_file=None, max_depth=-1):
        events.append(("ingest", srcs, from_hint, max_depth))
        return {"stage": "ingested"}

    def _fake_apply(dataset, args):
        events.append(
            (
                "apply",
                dataset,
                args.config,
                args.det_conf,
                args.class_det_conf,
                args.class_seg_conf,
            )
        )
        return {"stage": "filtered"}

    def _fake_output(dataset, args):
        events.append(("output", dataset, args.dst))
        return 0

    monkeypatch.setattr("cvsuite.label.cli.router.ingest_many", _fake_ingest_many)
    monkeypatch.setattr("cvsuite.label.transform.filter.commands.apply.run", _fake_apply)
    monkeypatch.setattr("cvsuite.label.output.commands.audit.run", _fake_output)

    src = tmp_path / "images"
    audit = tmp_path / "audit.yaml"

    exit_code = label_cli.main(
        [
            str(src),
            "--from",
            "images",
            "filter",
            "apply",
            "--det-conf",
            "0.25",
            "--class-det-conf",
            "dog=0.6",
            "--class-seg-conf",
            "dog=0.7",
            "audit",
            str(audit),
        ]
    )

    assert exit_code == 0
    assert events[0] == ("ingest", [src], "images", -1)
    assert events[1] == ("apply", {"stage": "ingested"}, None, 0.25, ["dog=0.6"], ["dog=0.7"])
    assert events[2] == ("output", {"stage": "filtered"}, audit)
