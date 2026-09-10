from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from cvsuite.common.core import BBox, ImageRecord, Polygon, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.io import read_dataset, write_dataset
from cvsuite.label.transform.ground.commands import run as ground_command
from cvsuite.label.transform.ground import core as ground_core
from cvsuite.label.transform.ocr.commands import run as ocr_command


def test_parse_prompt_items_keeps_raw_string_as_single_prompt() -> None:
    assert ground_core.parse_prompt_items("fire, smoke") == ["fire, smoke"]


def test_parse_prompt_items_reads_yaml_list(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text("- cat\n- dog\n- cat\n- ' '\n", encoding="utf-8")
    assert ground_core.parse_prompt_items(str(prompt_file)) == ["cat", "dog"]


def test_parse_prompt_spec_accepts_yaml_label_mapping(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text("label: cat\n", encoding="utf-8")
    spec = ground_core.parse_prompt_spec(str(prompt_file))
    assert spec.prompts == ["cat"]
    assert spec.prompt_to_label == {"cat": "label"}
    assert spec.labels == ["label"]


def test_parse_prompt_spec_reads_yaml_mapping(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text("metal:\n  - can\n  - tin can\nrust:\n  - rust patch\n", encoding="utf-8")
    spec = ground_core.parse_prompt_spec(str(prompt_file))
    assert spec.prompts == ["can", "tin can", "rust patch"]
    assert spec.prompt_to_label == {"can": "metal", "tin can": "metal", "rust patch": "rust"}
    assert spec.labels == ["metal", "rust"]


def test_parse_prompt_spec_rejects_prompt_reused_across_labels(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text("metal:\n  - can\nrust:\n  - can\n", encoding="utf-8")
    with pytest.raises(ValueError, match="assigned to multiple labels"):
        ground_core.parse_prompt_spec(str(prompt_file))


def test_parse_prompt_spec_rejects_invalid_yaml_payload(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text("7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level list of strings or a label-to-prompts mapping"):
        ground_core.parse_prompt_items(str(prompt_file))


def test_ground_attach_defaults_to_sam3() -> None:
    parser = ground_command.argparse.ArgumentParser()
    ground_command.attach(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--prompt", "thing"])

    args = parser.parse_args(["--provider", "sam3", "--prompt", "thing"])
    assert args.provider == "sam3"
    assert args.device == "auto"
    assert args.threshold == ground_core.DEFAULT_THRESHOLD
    assert args.iou_threshold == ground_core.DEFAULT_IOU_THRESHOLD


def test_ground_attach_accepts_gdino() -> None:
    parser = ground_command.argparse.ArgumentParser()
    ground_command.attach(parser)
    args = parser.parse_args(["--provider", "gdino", "--prompt", "thing"])
    assert args.provider == "gdino"


def test_ground_attach_accepts_locate_anything() -> None:
    parser = ground_command.argparse.ArgumentParser()
    ground_command.attach(parser)
    args = parser.parse_args(["--provider", "locate_anything", "--prompt", "thing"])
    assert args.provider == "locate_anything"


def test_ground_attach_accepts_llmdet_model_id() -> None:
    parser = ground_command.argparse.ArgumentParser()
    ground_command.attach(parser)
    args = parser.parse_args(
        [
            "--provider",
            "llmdet",
            "--model-id",
            "iSEE-Laboratory/llmdet_large",
            "--prompt",
            "thing",
        ]
    )
    assert args.provider == "llmdet"
    assert args.model_id == "iSEE-Laboratory/llmdet_large"


def test_ground_attach_allows_yolo_e_reference_without_prompt(tmp_path: Path) -> None:
    parser = ground_command.argparse.ArgumentParser()
    ground_command.attach(parser)
    args = parser.parse_args(["--provider", "yolo_e", "--reference-folder", str(tmp_path)])
    assert args.provider == "yolo_e"
    assert args.prompt is None
    assert args.reference_folder == tmp_path


def _ground_args(**overrides) -> Namespace:
    base = dict(
        provider="yolo_e",
        prompt=None,
        reference_folder=None,
        device="auto",
        batch=1,
        precision="fp32",
        config=None,
        model_id=None,
        threshold=0.3,
        iou_threshold=0.4,
        no_resume=False,
    )
    base.update(overrides)
    return Namespace(**base)


def test_ground_run_rejects_both_prompt_and_reference(tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], root=tmp_path)
    args = _ground_args(prompt="thing", reference_folder=tmp_path)
    with pytest.raises(ValueError, match="exactly one of --prompt or --reference-folder"):
        ground_command.run(dataset, args)


def test_ground_run_rejects_neither_prompt_nor_reference(tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], root=tmp_path)
    args = _ground_args(prompt=None, reference_folder=None)
    with pytest.raises(ValueError, match="exactly one of --prompt or --reference-folder"):
        ground_command.run(dataset, args)


def test_ground_run_rejects_reference_folder_for_gsam(tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], root=tmp_path)
    args = _ground_args(provider="gsam", reference_folder=tmp_path)
    with pytest.raises(ValueError, match="only supported by provider"):
        ground_command.run(dataset, args)


def test_ground_run_rejects_model_id_for_non_llmdet(tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], root=tmp_path)
    args = _ground_args(provider="gsam", prompt="thing", model_id="iSEE-Laboratory/llmdet_base")
    with pytest.raises(ValueError, match="--model-id is only supported"):
        ground_command.run(dataset, args)


def test_ground_run_forwards_llmdet_model_id(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=10, height=10))], root=tmp_path)
    captured = {}

    def _fake_run_ground(ds: VisionDataset, **kwargs) -> VisionDataset:
        captured.update(kwargs)
        return ds

    monkeypatch.setattr("cvsuite.label.transform.ground.core.run_ground", _fake_run_ground)

    args = _ground_args(provider="llmdet", prompt="thing", model_id="iSEE-Laboratory/llmdet_large")
    out = ground_command.run(dataset, args)

    assert out is dataset
    assert captured["model"] == "llmdet"
    assert captured["model_id"] == "iSEE-Laboratory/llmdet_large"


def test_ground_run_forwards_locate_anything_model_id(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=10, height=10))], root=tmp_path)
    captured = {}

    def _fake_run_ground(ds: VisionDataset, **kwargs) -> VisionDataset:
        captured.update(kwargs)
        return ds

    monkeypatch.setattr("cvsuite.label.transform.ground.core.run_ground", _fake_run_ground)

    args = _ground_args(provider="locate_anything", prompt="thing", model_id="nvidia/LocateAnything-3B")
    out = ground_command.run(dataset, args)

    assert out is dataset
    assert captured["model"] == "locate_anything"
    assert captured["model_id"] == "nvidia/LocateAnything-3B"


def test_run_ground_reference_rejects_multiple_reference_images(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], root=tmp_path)
    multi = VisionDataset(
        records=[
            Record(image=ImageRecord(path=tmp_path / "r1.jpg", width=10, height=10)),
            Record(image=ImageRecord(path=tmp_path / "r2.jpg", width=10, height=10)),
        ],
        root=tmp_path,
    )
    monkeypatch.setattr("cvsuite.label.core.router.ingest", lambda src: multi)

    with pytest.raises(ValueError, match="exactly one annotated reference image"):
        ground_core.run_ground_reference(
            dataset,
            model="yolo_e",
            reference_folder=tmp_path,
            device="cpu",
            precision="fp32",
            batch_size=1,
            config_path=None,
        )


def test_run_ground_reference_populates_fm_request(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], root=tmp_path)
    reference = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=tmp_path / "ref.jpg", width=20, height=10),
                boxes=[BBox(cx=0.5, cy=0.5, w=0.4, h=0.4, cls=0, label="bolt", group_id=1)],
            )
        ],
        classes=["bolt"],
        root=tmp_path,
    )
    monkeypatch.setattr("cvsuite.label.core.router.ingest", lambda src: reference)
    monkeypatch.setattr(VisionDataset, "to_json", lambda self, p: Path(str(p)).write_text("{}", encoding="utf-8"))
    captured = {}

    class _Runner:
        def run(self, ds: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = ds.fm_request
            return ds

    monkeypatch.setattr("cvsuite.label.transform.ground.core.FMRunner.from_dataset", lambda ds: _Runner())

    out = ground_core.run_ground_reference(
        dataset,
        model="yolo_e",
        reference_folder=tmp_path,
        threshold=0.4,
        device="gpu",
        precision="fp32",
        batch_size=1,
        config_path=None,
    )

    assert out is dataset
    request = captured["request"]
    assert request.provider == "yolo_e"
    assert request.device == "cuda"
    assert request.model_args["confidence_threshold"] == 0.4
    ref_path = Path(request.model_args["reference_dataset_path"])
    assert ref_path.name == "reference.json"


def test_ground_attach_rejects_clip() -> None:
    parser = ground_command.argparse.ArgumentParser()
    ground_command.attach(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--provider", "clip", "--prompt", "thing"])


def test_ground_attach_accepts_cuda_device_alias() -> None:
    parser = ground_command.argparse.ArgumentParser()
    ground_command.attach(parser)
    args = parser.parse_args(["--provider", "sam3", "--prompt", "thing", "--device", "cuda"])
    assert args.device == "cuda"


def test_ground_run_filters_only_new_annotations(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=tmp_path / "a.jpg", width=100, height=100),
                boxes=[
                    BBox(cx=0.5, cy=0.5, w=0.2, h=0.2, cls=0, label="existing", score=0.95, group_id=1),
                ],
            )
        ],
        classes=["existing"],
        root=tmp_path,
    )

    def _fake_run_ground(ds: VisionDataset, **kwargs) -> VisionDataset:
        assert kwargs["model"] == "gsam"
        assert kwargs["prompts"] == ["prompt one"]
        rec = ds.records[0]
        rec.attributes[ground_core.GROUND_PROMPTS_KEY] = ["prompt one"]
        rec.boxes.append(BBox(cx=0.2, cy=0.2, w=0.1, h=0.1, cls=1, label="prompt one", score=0.2, group_id=2))
        rec.polys.append(
            Polygon(points=[(0.1, 0.1), (0.2, 0.1), (0.1, 0.2)], cls=1, label="prompt one", score=0.2, group_id=2)
        )
        rec.boxes.append(BBox(cx=0.7, cy=0.7, w=0.1, h=0.1, cls=1, label="prompt one", score=0.9, group_id=3))
        rec.polys.append(
            Polygon(points=[(0.6, 0.6), (0.8, 0.6), (0.6, 0.8)], cls=1, label="prompt one", score=0.9, group_id=3)
        )
        return ds

    monkeypatch.setattr("cvsuite.label.transform.ground.core.run_ground", _fake_run_ground)

    args = Namespace(
        provider="gsam",
        prompt="prompt one",
        device="cpu",
        batch=1,
        precision="fp32",
        config=None,
        threshold=0.5,
        no_resume=False,
    )
    out = ground_command.run(dataset, args)
    rec = out.records[0]

    assert [box.group_id for box in rec.boxes] == [1, 3]
    assert [poly.group_id for poly in rec.polys] == [3]
    assert ground_core.GROUND_PROMPTS_KEY not in rec.attributes


def test_ground_run_suppresses_overlapping_new_groups_within_same_label(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=100, height=100))],
        root=tmp_path,
    )
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text("ground:\n  - prompt one\n  - prompt two\n", encoding="utf-8")

    def _fake_run_ground(ds: VisionDataset, **kwargs) -> VisionDataset:
        rec = ds.records[0]
        rec.attributes[ground_core.GROUND_PROMPTS_KEY] = ["prompt one", "prompt two"]
        rec.boxes.append(BBox(cx=0.50, cy=0.50, w=0.30, h=0.30, cls=0, label="prompt one", score=0.65, group_id=1))
        rec.polys.append(
            Polygon(points=[(0.35, 0.35), (0.65, 0.35), (0.65, 0.65), (0.35, 0.65)], cls=0, label="prompt one", score=0.65, group_id=1)
        )
        rec.boxes.append(BBox(cx=0.52, cy=0.52, w=0.30, h=0.30, cls=1, label="prompt two", score=0.92, group_id=2))
        rec.polys.append(
            Polygon(points=[(0.37, 0.37), (0.67, 0.37), (0.67, 0.67), (0.37, 0.67)], cls=1, label="prompt two", score=0.92, group_id=2)
        )
        return ds

    monkeypatch.setattr("cvsuite.label.transform.ground.core.run_ground", _fake_run_ground)

    args = Namespace(
        provider="sam3",
        prompt=str(prompt_file),
        device="cuda",
        batch=1,
        precision="fp32",
        config=None,
        threshold=0.3,
        no_resume=False,
    )
    out = ground_command.run(dataset, args)

    assert len(out.records) == 1
    rec = out.records[0]
    assert [box.group_id for box in rec.boxes] == [2]
    assert [box.label for box in rec.boxes] == ["ground"]
    assert [box.prompt for box in rec.boxes] == ["prompt two"]
    assert [poly.group_id for poly in rec.polys] == [2]
    assert out.classes == ["ground"]
    assert ground_core.GROUND_PROMPTS_KEY not in rec.attributes


def test_ground_run_keeps_overlapping_new_groups_for_distinct_labels(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=100, height=100))],
        root=tmp_path,
    )
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text("metal:\n  - prompt one\nrust:\n  - prompt two\n", encoding="utf-8")

    def _fake_run_ground(ds: VisionDataset, **kwargs) -> VisionDataset:
        rec = ds.records[0]
        rec.attributes[ground_core.GROUND_PROMPTS_KEY] = ["prompt one", "prompt two"]
        rec.boxes.append(BBox(cx=0.50, cy=0.50, w=0.30, h=0.30, cls=0, label="prompt one", score=0.65, group_id=1))
        rec.boxes.append(BBox(cx=0.52, cy=0.52, w=0.30, h=0.30, cls=1, label="prompt two", score=0.92, group_id=2))
        return ds

    monkeypatch.setattr("cvsuite.label.transform.ground.core.run_ground", _fake_run_ground)

    args = Namespace(
        provider="sam3",
        prompt=str(prompt_file),
        device="cuda",
        batch=1,
        precision="fp32",
        config=None,
        threshold=0.3,
        no_resume=False,
    )
    out = ground_command.run(dataset, args)
    rec = out.records[0]

    assert [box.group_id for box in rec.boxes] == [1, 2]
    assert [box.label for box in rec.boxes] == ["metal", "rust"]
    assert [box.prompt for box in rec.boxes] == ["prompt one", "prompt two"]
    assert out.classes == ["metal", "rust"]


def test_run_ground_rejects_sam3_cpu(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=10, height=10))],
        root=tmp_path,
    )

    with pytest.raises(ValueError, match="sam3 does not support CPU execution"):
        ground_core.run_ground(
            dataset,
            model="sam3",
            prompts=["thing"],
            device="cpu",
            precision="fp32",
            batch_size=1,
            config_path=None,
        )


def test_run_ground_normalizes_gpu_alias_to_cuda(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=10, height=10))],
        root=tmp_path,
    )
    captured = {}

    class _Runner:
        def run(self, ds: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = ds.fm_request
            captured["no_resume"] = no_resume
            return ds

    monkeypatch.setattr("cvsuite.label.transform.ground.core.FMRunner.from_dataset", lambda ds: _Runner())

    out = ground_core.run_ground(
        dataset,
        model="gsam",
        prompts=["thing"],
        device="gpu",
        precision="fp32",
        batch_size=1,
        config_path=None,
        no_resume=True,
    )

    assert out is dataset
    assert captured["request"] is not None
    assert captured["request"].device == "cuda"
    assert captured["no_resume"] is True


def test_run_ground_forwards_threshold_to_sam3_confidence_threshold(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=10, height=10))],
        root=tmp_path,
    )
    captured = {}

    class _Runner:
        def run(self, ds: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = ds.fm_request
            captured["no_resume"] = no_resume
            return ds

    monkeypatch.setattr("cvsuite.label.transform.ground.core.FMRunner.from_dataset", lambda ds: _Runner())

    out = ground_core.run_ground(
        dataset,
        model="sam3",
        prompts=["thing"],
        threshold=0.125,
        device="cuda",
        precision="fp32",
        batch_size=1,
        config_path=None,
        no_resume=True,
    )

    assert out is dataset
    assert captured["request"] is not None
    assert captured["request"].model_args == {"confidence_threshold": 0.125}
    assert captured["no_resume"] is True


def test_run_ground_forwards_llmdet_model_id(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=10, height=10))],
        root=tmp_path,
    )
    captured = {}

    class _Runner:
        def run(self, ds: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = ds.fm_request
            return ds

    monkeypatch.setattr("cvsuite.label.transform.ground.core.FMRunner.from_dataset", lambda ds: _Runner())

    out = ground_core.run_ground(
        dataset,
        model="llmdet",
        prompts=["thing"],
        model_id="iSEE-Laboratory/llmdet_large",
        device="cpu",
        precision="fp32",
        batch_size=1,
        config_path=None,
    )

    assert out is dataset
    assert captured["request"].model_args == {"model_id": "iSEE-Laboratory/llmdet_large"}


def test_run_ground_forwards_locate_anything_model_id(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=10, height=10))],
        root=tmp_path,
    )
    captured = {}

    class _Runner:
        def run(self, ds: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = ds.fm_request
            return ds

    monkeypatch.setattr("cvsuite.label.transform.ground.core.FMRunner.from_dataset", lambda ds: _Runner())

    out = ground_core.run_ground(
        dataset,
        model="locate_anything",
        prompts=["thing"],
        model_id="nvidia/LocateAnything-3B",
        device="cpu",
        precision="fp32",
        batch_size=1,
        config_path=None,
    )

    assert out is dataset
    assert captured["request"].model_args == {"model_id": "nvidia/LocateAnything-3B"}


def test_run_ground_rejects_unknown_llmdet_model_id(tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], root=tmp_path)
    with pytest.raises(ValueError, match="Unsupported llmdet --model-id"):
        ground_core.run_ground(
            dataset,
            model="llmdet",
            prompts=["thing"],
            model_id="not-real",
            device="cpu",
            precision="fp32",
            batch_size=1,
            config_path=None,
        )


def test_run_ground_rejects_unknown_locate_anything_model_id(tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], root=tmp_path)
    with pytest.raises(ValueError, match="Unsupported locate_anything --model-id"):
        ground_core.run_ground(
            dataset,
            model="locate_anything",
            prompts=["thing"],
            model_id="not-real",
            device="cpu",
            precision="fp32",
            batch_size=1,
            config_path=None,
        )


def test_ocr_provider_alias_normalizes_to_paddleocr() -> None:
    assert ocr_command.normalize_provider("paddleOCR") == "paddleocr"


def test_ocr_run_populates_fm_request(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "a.jpg", width=10, height=10))],
        root=tmp_path,
    )
    config_path = tmp_path / "ocr.yaml"
    config_path.write_text("lang: en\n", encoding="utf-8")
    captured = {}

    class _Runner:
        def run(self, ds: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = ds.fm_request
            captured["no_resume"] = no_resume
            return ds

    monkeypatch.setattr("cvsuite.label.transform.ocr.commands.run.FMRunner.from_dataset", lambda ds: _Runner())

    args = Namespace(provider="paddleOCR", device="cpu", batch=2, precision="fp32", config=config_path, no_resume=True)
    out = ocr_command.run(dataset, args)

    assert out is dataset
    assert captured["request"] is not None
    assert captured["request"].task == "ocr"
    assert captured["request"].provider == "paddleocr"
    assert captured["request"].batch_size == 2
    assert captured["request"].config_path == config_path
    assert captured["no_resume"] is True


def test_labelme_write_and_ingest_preserve_ground_prompt_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 48), color=(255, 255, 255)).save(image_path)
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=64, height=48),
                boxes=[
                    BBox(
                        cx=0.5,
                        cy=0.5,
                        w=0.25,
                        h=0.25,
                        cls=0,
                        label="ground",
                        prompt="rust patch",
                        score=0.91,
                        group_id=7,
                    )
                ],
            )
        ],
        classes=["ground"],
        root=tmp_path,
    )

    out_dir = tmp_path / "labelme_out"
    write_dataset(dataset, out_dir, format="labelme", embed_image=False)
    json_files = sorted(out_dir.rglob("*.json"))
    assert json_files

    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    shape = payload["shapes"][0]
    assert shape["label"] == "ground"
    assert shape["group_id"] == 7
    assert shape["flags"] == {}
    assert shape["attributes"]["prompt"] == "rust patch"
    assert shape["score"] == pytest.approx(0.91)

    reloaded = read_dataset(out_dir, format="labelme")
    box = reloaded.records[0].boxes[0]
    assert box.label == "ground"
    assert box.prompt == "rust patch"
    assert box.score == pytest.approx(0.91)
    assert box.group_id == 7


def test_labelme_ingest_migrates_legacy_non_boolean_flags(tmp_path: Path) -> None:
    src_dir = tmp_path / "legacy_labelme" / "train"
    src_dir.mkdir(parents=True, exist_ok=True)
    image_path = src_dir / "legacy.jpg"
    Image.new("RGB", (64, 48), color=(255, 255, 255)).save(image_path)

    (src_dir / "legacy.json").write_text(
        json.dumps(
            {
                "imagePath": image_path.name,
                "imageHeight": 48,
                "imageWidth": 64,
                "shapes": [
                    {
                        "label": "ground",
                        "shape_type": "rectangle",
                        "points": [[16, 12], [32, 24]],
                        "flags": {
                            "verified": True,
                            "score": 0.77,
                            "prompt": "pressure gauge",
                            "model": "sam3",
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = read_dataset(src_dir.parent, format="labelme")
    box = dataset.records[0].boxes[0]
    assert box.attributes["flags"] == {"verified": True}
    assert box.prompt == "pressure gauge"
    assert box.model == "sam3"
    assert box.score == pytest.approx(0.77)

    out_dir = tmp_path / "migrated_labelme"
    write_dataset(dataset, out_dir, format="labelme", embed_image=False)
    payload = json.loads(next(out_dir.rglob("*.json")).read_text(encoding="utf-8"))
    migrated_shape = payload["shapes"][0]
    assert migrated_shape["flags"] == {"verified": True}
    assert migrated_shape["attributes"]["prompt"] == "pressure gauge"
    assert migrated_shape["attributes"]["model"] == "sam3"
    assert migrated_shape["score"] == pytest.approx(0.77)


def test_labelme_ingest_preserves_new_shape_and_record_attributes(tmp_path: Path) -> None:
    src_dir = tmp_path / "new_labelme" / "train"
    src_dir.mkdir(parents=True, exist_ok=True)
    image_path = src_dir / "new.jpg"
    Image.new("RGB", (64, 48), color=(255, 255, 255)).save(image_path)

    (src_dir / "new.json").write_text(
        json.dumps(
            {
                "imagePath": image_path.name,
                "imageHeight": 48,
                "imageWidth": 64,
                "attributes": {"cvsuite": {"inverse_crop_dets": {"applied": ["crop-1"]}}},
                "shapes": [
                    {
                        "label": "ground",
                        "shape_type": "rectangle",
                        "points": [[16, 12], [32, 24]],
                        "score": 0.88,
                        "flags": {"verified": False},
                        "attributes": {
                            "prompt": "rust patch",
                            "cvsuite": {"crop_dets": {"crop_id": "crop-1"}},
                            "flags": {"edited": True},
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = read_dataset(src_dir.parent, format="labelme")
    record = dataset.records[0]
    box = record.boxes[0]
    assert record.attributes["cvsuite"]["inverse_crop_dets"]["applied"] == ["crop-1"]
    assert box.prompt == "rust patch"
    assert box.score == pytest.approx(0.88)
    assert box.attributes["cvsuite"]["crop_dets"]["crop_id"] == "crop-1"
    assert box.attributes["flags"] == {"verified": False, "edited": True}

    out_dir = tmp_path / "roundtrip_new_labelme"
    write_dataset(dataset, out_dir, format="labelme", embed_image=False)
    payload = json.loads(next(out_dir.rglob("*.json")).read_text(encoding="utf-8"))
    shape = payload["shapes"][0]
    assert payload["attributes"]["cvsuite"]["inverse_crop_dets"]["applied"] == ["crop-1"]
    assert shape["attributes"]["prompt"] == "rust patch"
    assert shape["attributes"]["cvsuite"]["crop_dets"]["crop_id"] == "crop-1"
    assert shape["flags"] == {"verified": False, "edited": True}
    assert shape["score"] == pytest.approx(0.88)
