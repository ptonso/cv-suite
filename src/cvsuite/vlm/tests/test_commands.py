from __future__ import annotations

import builtins
import io as bytes_io
import json
from pathlib import Path
import sys
import tarfile

from PIL import Image
import pytest

from cvsuite.common.core import VisionDataset
from cvsuite.vlm import cli as vlm_cli
from cvsuite.vlm.core import inspect as inspect_core
from cvsuite.vlm.core import io as vlm_io
from cvsuite.vlm.core import router
from cvsuite.vlm.core.adapters.flat_json_io import FlatJsonIO
from cvsuite.vlm.core.adapters.shards_io import ShardsIO
from cvsuite.vlm.core.adapters.vqa_style_io import VQAStyleIO


def _make_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (12, 8), color=color)
    image.save(path)
    image.close()


def _fake_runner_factory(captured: dict):
    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = dataset.fm_request
            captured["no_resume"] = no_resume
            for record in dataset.records:
                for qa in record.vqas:
                    if qa.answer and qa.model:
                        continue
                    qa.answer = f"ans:{qa.question}"
                    qa.model = captured["request"].model if captured["request"] is not None else "qwen"
            return dataset

    return _Runner()


def test_router_detects_flat_json_and_ingests_ground_truth(tmp_path: Path) -> None:
    root = tmp_path / "flat"
    _make_image(root / "0001.jpg")
    (root / "0001.json").write_text(
        json.dumps(
            {
                "version": "1",
                "task": "vlm_sample",
                "sample_key": "0001",
                "image_id": 1,
                "predominant_label": "car",
                "split": "train",
                "qa_pairs": [
                    {
                        "question_id": 7,
                        "question": "What color is the car?",
                        "ground_truth": "red",
                        "answers": ["red", "dark red"],
                        "label": "color",
                        "meta": {"answer_type": "other"},
                    }
                ],
                "meta": {"source_dataset": "demo"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert router.detect_format(root) == "json"
    dataset = router.ingest(root)
    record = dataset.records[0]
    qa = record.vqas[0]
    assert record.attributes[vlm_io.SAMPLE_KEY_ATTR] == "0001"
    assert record.attributes[vlm_io.IMAGE_ID_ATTR] == 1
    assert record.attributes[vlm_io.PREDOMINANT_LABEL_ATTR] == "car"
    assert qa.meta["id"] == 7
    assert qa.meta["ground_truth"] == "red"
    assert qa.meta["expected_answers"] == ["red", "dark red"]
    assert qa.meta["source"] == "json"
    assert qa.meta["source_meta"] == {"answer_type": "other"}


def test_router_detects_recursive_flat_json_layout(tmp_path: Path) -> None:
    root = tmp_path / "recursive_flat"
    _make_image(root / "images" / "train" / "0001.jpg")
    (root / "labels" / "train" / "0001.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train" / "0001.json").write_text(
        json.dumps(
            {
                "qa_pairs": [
                    {
                        "question": "What is visible?",
                        "ground_truth": "tree",
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert router.detect_format(root) == "json"
    dataset = router.ingest(root)
    assert len(dataset.records) == 1
    record = dataset.records[0]
    assert record.image.path == Path("images/train/0001.jpg")
    assert record.rel_image_path == Path("images/train/0001.jpg")
    assert record.attributes[vlm_io.SAMPLE_KEY_ATTR] == "images__train__0001"
    assert record.vqas[0].meta["ground_truth"] == "tree"


def test_router_detects_shards_and_manifest(tmp_path: Path) -> None:
    shard_root = tmp_path / "shards_ds"
    shard_dir = shard_root / "shards"
    shard_dir.mkdir(parents=True)
    image_bytes = bytes_io.BytesIO()
    Image.new("RGB", (10, 10), color=(0, 255, 0)).save(image_bytes, format="JPEG")
    sample_payload = {
        "sample_key": "sample",
        "image_id": 3,
        "original_filename": "sample.jpg",
        "qa_pairs": [{"question_id": 1, "question": "What is shown?", "ground_truth": "tree"}],
    }
    with tarfile.open(shard_dir / "shard-000000.tar", mode="w") as handle:
        img_info = tarfile.TarInfo("sample.jpg")
        img_bytes = image_bytes.getvalue()
        img_info.size = len(img_bytes)
        handle.addfile(img_info, bytes_io.BytesIO(img_bytes))
        json_bytes = json.dumps(sample_payload).encode("utf-8")
        json_info = tarfile.TarInfo("sample.json")
        json_info.size = len(json_bytes)
        handle.addfile(json_info, bytes_io.BytesIO(json_bytes))

    manifest_root = tmp_path / "manifest_ds"
    _make_image(manifest_root / "images" / "train" / "img.jpg")
    (manifest_root / "vqa.json").write_text(
        json.dumps(
            {
                "version": "1",
                "task": "vlm",
                "images_root": "images",
                "items": [
                    {
                        "image": "train/img.jpg",
                        "split": "train",
                        "qas": [{"id": 9, "prompt": "What is visible?", "ground_truth": "tree"}],
                        "meta": {"image_id": 9},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert router.detect_format(shard_root) == "shards"
    assert router.detect_format(manifest_root) == "vqa-style"


def test_router_ingests_raw_vqa_questions_without_annotations(tmp_path: Path) -> None:
    root = tmp_path / "raw_vqa"
    _make_image(root / "train2014" / "COCO_train2014_000000263006.jpg")
    (root / "nested" / "v2_OpenEnded_mscoco_train2014_questions.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "nested" / "v2_OpenEnded_mscoco_train2014_questions.json").write_text(
        json.dumps(
            {
                "data_subtype": "train2014",
                "data_type": "mscoco",
                "questions": [
                    {
                        "image_id": 263006,
                        "question": "How many people are pictured?",
                        "question_id": 263006000,
                    },
                    {
                        "image_id": 263006,
                        "question": "What color is the water?",
                        "question_id": 263006001,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert router.detect_format(root) == "vqa-style"
    dataset = router.ingest(root)
    assert len(dataset.records) == 1
    record = dataset.records[0]
    assert record.image.path == Path("train2014/COCO_train2014_000000263006.jpg")
    assert record.split == "train2014"
    assert record.attributes[vlm_io.IMAGE_ID_ATTR] == 263006
    assert [qa.question for qa in record.vqas] == ["How many people are pictured?", "What color is the water?"]
    assert all(qa.answer == "" for qa in record.vqas)


def test_router_ingests_raw_vqa_annotations_and_predictions(tmp_path: Path) -> None:
    root = tmp_path / "raw_vqa"
    _make_image(root / "images" / "train2014" / "COCO_train2014_000000263006.jpg")
    (root / "json").mkdir(parents=True, exist_ok=True)
    (root / "json" / "questions.json").write_text(
        json.dumps(
            {
                "data_subtype": "train2014",
                "data_type": "mscoco",
                "questions": [
                    {
                        "image_id": 263006,
                        "question": "How many people are pictured?",
                        "question_id": 263006000,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "json" / "annotations.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "answer_type": "number",
                        "answers": [{"answer": "2"}, {"answer": "2"}],
                        "image_id": 263006,
                        "multiple_choice_answer": "2",
                        "question_id": 263006000,
                        "question_type": "how many people are",
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "json" / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "question_id": 263006000,
                    "answer": "2",
                    "model": "demo-vlm",
                    "score": 0.9,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = router.ingest(root)
    record = dataset.records[0]
    qa = record.vqas[0]
    assert qa.answer == "2"
    assert qa.model == "demo-vlm"
    assert qa.score == 0.9
    assert qa.meta["ground_truth"] == "2"
    assert qa.meta["expected_answers"] == ["2", "2"]
    assert qa.meta["source"] == "vqa-style"
    assert qa.meta["source_meta"]["answer_type"] == "number"
    assert qa.meta["source_meta"]["question_type"] == "how many people are"


def test_vlm_cli_images_to_ask_to_json(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    _make_image(src / "nested" / "a.png")
    captured: dict = {}
    monkeypatch.setattr("cvsuite.vlm.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _fake_runner_factory(captured))

    dst = tmp_path / "json_out"
    exit_code = vlm_cli.main(
        [
            str(src),
            "ask",
            "--provider",
            "qwen",
            "--prompt",
            json.dumps({"color": ["What color is the object?"]}),
            "--model-id",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "--no-resume",
            "--model-arg",
            "max_new_tokens=32",
            "to-json",
            str(dst),
        ]
    )

    assert exit_code == 0
    assert captured["request"].task == "vlm"
    assert captured["request"].model == "qwen"
    assert captured["request"].prompt == json.dumps({"color": ["What color is the object?"]})
    assert captured["request"].model_args["model_id"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert captured["request"].model_args["max_new_tokens"] == 32
    assert captured["no_resume"] is True

    sidecars = sorted(dst.glob("*.json"))
    assert len(sidecars) == 1
    payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert payload["qa_pairs"][0]["question"] == "What color is the object?"
    assert payload["qa_pairs"][0]["answer"] == "ans:What color is the object?"
    assert payload["qa_pairs"][0]["label"] == "color"
    assert payload["qa_pairs"][0]["meta"]["source"] == "ask"


def test_vlm_cli_images_to_ask_to_json_records_max_gpu_memory(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    _make_image(src / "nested" / "a.png")
    captured: dict = {}
    monkeypatch.setattr("cvsuite.vlm.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _fake_runner_factory(captured))

    dst = tmp_path / "json_out"
    exit_code = vlm_cli.main(
        [
            str(src),
            "ask",
            "--provider",
            "qwen",
            "--prompt",
            "What color is the object?",
            "--device",
            "auto",
            "--max-gpu-memory",
            "14GiB",
            "to-json",
            str(dst),
        ]
    )

    assert exit_code == 0
    assert captured["request"].max_gpu_memory == "14GiB"


def test_vlm_cli_rejects_max_gpu_memory_without_auto_device(tmp_path: Path) -> None:
    src = tmp_path / "images"
    _make_image(src / "nested" / "a.png")
    dst = tmp_path / "json_out"

    with pytest.raises(SystemExit, match="requires `--device auto`"):
        vlm_cli.main(
            [
                str(src),
                "ask",
                "--provider",
                "qwen",
                "--prompt",
                "What color is the object?",
                "--device",
                "gpu",
                "--max-gpu-memory",
                "14GiB",
                "to-json",
                str(dst),
            ]
        )


def test_vlm_cli_images_to_ask_to_json_normalizes_qwen_model_namespace(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    _make_image(src / "nested" / "a.png")
    captured: dict = {}
    monkeypatch.setattr("cvsuite.vlm.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _fake_runner_factory(captured))

    dst = tmp_path / "json_out"
    exit_code = vlm_cli.main(
        [
            str(src),
            "ask",
            "--prompt",
            "What color is the object?",
            "--model",
            "qwen",
            "--model-id",
            "Qwen2.5-VL-3B-Instruct",
            "to-json",
            str(dst),
        ]
    )

    assert exit_code == 0
    assert captured["request"].model == "qwen"
    assert captured["request"].model_args["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_vlm_cli_images_to_ask_to_json_normalizes_qwen3_model_namespace(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    _make_image(src / "nested" / "a.png")
    captured: dict = {}
    monkeypatch.setattr("cvsuite.vlm.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _fake_runner_factory(captured))

    dst = tmp_path / "json_out"
    exit_code = vlm_cli.main(
        [
            str(src),
            "ask",
            "--prompt",
            "What color is the object?",
            "--model",
            "qwen",
            "--model-id",
            "Qwen3-VL-30B-A3B-Instruct",
            "to-json",
            str(dst),
        ]
    )

    assert exit_code == 0
    assert captured["request"].model == "qwen"
    assert captured["request"].model_args["model_id"] == "Qwen/Qwen3-VL-30B-A3B-Instruct"


def test_vlm_cli_images_to_ask_to_json_minicpm_alias(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    _make_image(src / "nested" / "a.png")
    captured: dict = {}
    monkeypatch.setattr("cvsuite.vlm.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _fake_runner_factory(captured))

    dst = tmp_path / "json_out"
    exit_code = vlm_cli.main(
        [
            str(src),
            "ask",
            "--prompt",
            "What is visible?",
            "--model",
            "minicpm-v",
            "--model-id",
            "openbmb/MiniCPM-V-4",
            "to-json",
            str(dst),
        ]
    )

    assert exit_code == 0
    assert captured["request"].model == "minicpm_v"
    assert captured["request"].model_args["model_id"] == "openbmb/MiniCPM-V-4"


def test_vlm_cli_caption_to_vqa_style(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    _make_image(src / "frame.jpg")
    captured: dict = {}
    monkeypatch.setattr("cvsuite.vlm.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _fake_runner_factory(captured))

    dst = tmp_path / "manifest_out"
    exit_code = vlm_cli.main([str(src), "caption", "--provider", "qwen", "to-vqa-style", str(dst)])

    assert exit_code == 0
    assert captured["request"].prompt == vlm_io.CAPTION_PROMPT
    manifest = json.loads((dst / "vqa.json").read_text(encoding="utf-8"))
    assert manifest["items"][0]["qas"][0]["prompt"] == vlm_io.CAPTION_PROMPT
    assert manifest["items"][0]["qas"][0]["answer"] == f"ans:{vlm_io.CAPTION_PROMPT}"
    assert (dst / "images" / "frame.jpg").exists()


def test_vlm_cli_ask_to_json_skip_images(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    _make_image(src / "nested" / "a.png")
    captured: dict = {}
    monkeypatch.setattr("cvsuite.vlm.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _fake_runner_factory(captured))

    dst = tmp_path / "json_out"
    exit_code = vlm_cli.main(
        [
            str(src),
            "ask",
            "--provider",
            "qwen",
            "--prompt",
            "What color is the object?",
            "to-json",
            str(dst),
            "--skip-images",
        ]
    )

    assert exit_code == 0
    payload = json.loads((dst / "a.json").read_text(encoding="utf-8"))
    assert payload["qa_pairs"][0]["answer"] == "ans:What color is the object?"
    assert payload["original_filename"] == "a.png"
    assert not (dst / "a.png").exists()


def test_vlm_cli_caption_to_vqa_style_skip_images(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "images"
    _make_image(src / "frame.jpg")
    captured: dict = {}
    monkeypatch.setattr("cvsuite.vlm.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _fake_runner_factory(captured))

    dst = tmp_path / "manifest_out"
    exit_code = vlm_cli.main([str(src), "caption", "--provider", "qwen", "to-vqa-style", str(dst), "--skip-images"])

    assert exit_code == 0
    manifest = json.loads((dst / "vqa.json").read_text(encoding="utf-8"))
    assert manifest["images_root"] == "images"
    assert manifest["items"][0]["image"] == "frame.jpg"
    assert not (dst / "images").exists()


def test_vlm_cli_vqa_help_does_not_import_torch(monkeypatch, capsys) -> None:
    for name in [
        "torch",
        "cvsuite.common.fm",
        "cvsuite.common.fm.core",
        "cvsuite.vlm.transform.core.runtime",
        "cvsuite.vlm.transform.commands.vqa",
    ]:
        sys.modules.pop(name, None)

    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise AssertionError("`cvsuite vlm vqa -h` should not import torch.")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    monkeypatch.setattr(
        "cvsuite.vlm.transform.core.runtime.resolve_supported_model_catalog",
        lambda model: (
            (
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "Qwen/Qwen2.5-VL-32B-Instruct",
                "Qwen/Qwen2.5-VL-72B-Instruct",
            ),
            "live Hugging Face query",
        ),
    )

    exit_code = vlm_cli.main(["vqa", "--model", "qwen", "-h"])

    assert exit_code == 0
    help_text = capsys.readouterr().out
    assert "--model" in help_text
    assert "--model-id" in help_text
    assert "--precision" in help_text
    assert "--max-gpu-memory" in help_text
    assert "nf4" in help_text
    assert 'device_map="auto"' in help_text
    assert "CPU RAM then disk fallback" in help_text
    assert "NF4 currently requires `--device cuda`" in help_text
    assert "Local providers:" in help_text
    assert "  qwen: default-model-id=Qwen/Qwen2.5-VL-3B-Instruct params=3B" in help_text
    assert "Qwen/Qwen2.5-VL-72B-Instruct" not in help_text
    assert "live Hugging Face query" not in help_text


def test_vlm_cli_top_level_model_help_lists_qwen_metadata(capsys) -> None:
    exit_code = vlm_cli.main(["--model", "qwen", "-h"])
    assert exit_code == 0
    help_text = capsys.readouterr().out
    assert "usage: cvsuite vlm <transform> --provider qwen [--model-id MODEL_ID] ..." in help_text
    assert "Local providers:" in help_text
    assert "  qwen: default-model-id=Qwen/Qwen2.5-VL-3B-Instruct params=3B" in help_text
    assert "live Hugging Face query" not in help_text
    assert "bitsandbytes" in help_text
    assert "auto-installs" in help_text
    assert "--max-gpu-memory" in help_text
    assert 'device_map="auto"' in help_text
    assert "Qwen/Qwen2.5-VL-72B-Instruct" not in help_text


def test_vlm_cli_top_level_model_help_lists_minicpm_metadata(capsys) -> None:
    exit_code = vlm_cli.main(["--model", "minicpm-v", "-h"])
    assert exit_code == 0
    help_text = capsys.readouterr().out
    assert "usage: cvsuite vlm <transform> --provider minicpm-v [--model-id MODEL_ID] ..." in help_text
    assert "  minicpm-v: default-model-id=openbmb/MiniCPM-V-4 params=8B" in help_text
    assert "openbmb/MiniCPM-V-4_5" not in help_text
    assert "live Hugging Face query" not in help_text


def test_vlm_cli_runtime_args_expose_new_model_wrappers() -> None:
    from cvsuite.vlm.transform.core import runtime

    assert "llava" in runtime.SUPPORTED_MODELS
    assert "blip" in runtime.SUPPORTED_MODELS
    assert "cogvlm" in runtime.SUPPORTED_MODELS
    assert "internvl" in runtime.SUPPORTED_MODELS
    assert "minicpm-v" in runtime.SUPPORTED_MODELS


def test_vlm_cli_json_to_vqa_to_shards(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "flat"
    _make_image(src / "0001.jpg")
    (src / "0001.json").write_text(
        json.dumps(
            {
                "sample_key": "0001",
                "qa_pairs": [{"question_id": 1, "question": "What is visible?", "ground_truth": "tree"}],
            }
        ),
        encoding="utf-8",
    )
    captured: dict = {}
    monkeypatch.setattr("cvsuite.vlm.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _fake_runner_factory(captured))

    dst = tmp_path / "sharded"
    exit_code = vlm_cli.main([str(src), "vqa", "--provider", "qwen", "to-shards", str(dst), "--target-shard-size-mb", "1", "--max-samples-per-shard", "1"])
    assert exit_code == 0

    shard_paths = sorted((dst / "shards").glob("*.tar"))
    assert len(shard_paths) == 1
    with tarfile.open(shard_paths[0], mode="r") as handle:
        payload_file = handle.extractfile("0001.json")
        assert payload_file is not None
        payload = json.loads(payload_file.read().decode("utf-8"))
    assert payload["qa_pairs"][0]["answer"] == "ans:What is visible?"
    assert payload["qa_pairs"][0]["ground_truth"] == "tree"


def test_vlm_cli_shards_filters_raw_vqa_splits_by_prefix(tmp_path: Path) -> None:
    src = tmp_path / "raw_vqa"
    _make_image(src / "train2014" / "COCO_train2014_000000000001.jpg")
    _make_image(src / "val2014" / "COCO_val2014_000000000002.jpg")
    _make_image(src / "test2015" / "COCO_test2015_000000000003.jpg")
    (src / "json").mkdir(parents=True, exist_ok=True)
    (src / "json" / "train_questions.json").write_text(
        json.dumps(
            {
                "data_subtype": "train2014",
                "questions": [{"image_id": 1, "question": "train?", "question_id": 101}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (src / "json" / "val_questions.json").write_text(
        json.dumps(
            {
                "data_subtype": "val2014",
                "questions": [{"image_id": 2, "question": "val?", "question_id": 102}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (src / "json" / "test_questions.json").write_text(
        json.dumps(
            {
                "data_subtype": "test2015",
                "questions": [{"image_id": 3, "question": "test?", "question_id": 103}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    dst = tmp_path / "sharded"
    exit_code = vlm_cli.main(
        [
            str(src),
            "to-shards",
            str(dst),
            "--splits",
            "train,val",
            "--target-shard-size-mb",
            "1",
            "--max-samples-per-shard",
            "10",
        ]
    )
    assert exit_code == 0

    dataset = ShardsIO.ingest(dst)
    assert {record.split for record in dataset.records} == {"train2014", "val2014"}
    assert all(record.split != "test2015" for record in dataset.records)


def test_vlm_cli_vqa_requires_pending_questions(tmp_path: Path) -> None:
    src = tmp_path / "images"
    _make_image(src / "a.jpg")
    try:
        vlm_cli.main([str(src), "vqa", "--provider", "qwen", "to-json", str(tmp_path / "out")])
    except SystemExit as exc:
        assert "pending question" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for missing questions")


def test_roundtrip_writers_keep_vqa_fields(tmp_path: Path) -> None:
    root = tmp_path / "flat"
    _make_image(root / "0001.jpg")
    (root / "0001.json").write_text(
        json.dumps(
            {
                "sample_key": "0001",
                "image_id": 11,
                "predominant_label": "vehicle",
                "qa_pairs": [
                    {
                        "question_id": 5,
                        "question": "What color is it?",
                        "answer": "red",
                        "ground_truth": "red",
                        "answers": ["red"],
                        "label": "color",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = router.ingest(root)

    manifest_out = tmp_path / "manifest"
    VQAStyleIO.write(dataset, manifest_out, hardlink=False)
    manifest_dataset = VQAStyleIO.ingest(manifest_out)
    manifest_qa = manifest_dataset.records[0].vqas[0]
    assert manifest_qa.answer == "red"
    assert manifest_qa.meta["ground_truth"] == "red"

    shard_out = tmp_path / "shards"
    ShardsIO.write(dataset, shard_out, target_shard_size_mb=1, max_samples_per_shard=10)
    shard_dataset = ShardsIO.ingest(shard_out)
    shard_qa = shard_dataset.records[0].vqas[0]
    assert shard_qa.answer == "red"
    assert shard_qa.meta["expected_answers"] == ["red"]

    flat_out = tmp_path / "flat_out"
    FlatJsonIO.write(dataset, flat_out, hardlink=False)
    flat_dataset = FlatJsonIO.ingest(flat_out)
    flat_qa = flat_dataset.records[0].vqas[0]
    assert flat_qa.answer == "red"
    assert flat_qa.meta["label"] == "color"


def test_inspect_render_returns_canvas(tmp_path: Path) -> None:
    try:
        import numpy as _np  # noqa: F401
    except ModuleNotFoundError:
        return

    src = tmp_path / "images"
    _make_image(src / "img.jpg")
    dataset = router.ingest(src)
    dataset.records[0].vqas = []
    dataset.records[0].vqas.append(
        vlm_io.qa_from_payload(
            {"question": "What is visible?", "answer": "tree", "ground_truth": "tree", "answers": ["tree"]},
            source="json",
            question_keys=("question",),
        )
    )
    canvas = inspect_core.render_record(dataset, dataset.records[0], max_height=320, panel_width=320)
    assert canvas.ndim == 3
    assert canvas.shape[0] <= 320
    assert canvas.shape[1] > canvas.shape[0]
