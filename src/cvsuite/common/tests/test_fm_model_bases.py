from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cvsuite.common.core.enums import Task
from cvsuite.common.core import FMRequest, ImageRecord, Record, VQA
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.providers.base import BatchResult
from cvsuite.common.fm.providers.bases import (
    BaseClassificationModel,
    BaseCreateGenerationModel,
    BaseEditGenerationModel,
    BaseGenerationModel,
    BaseGroundModel,
    BaseOCRModel,
    BaseVLMModel,
    RecordImageJob,
)
from cvsuite.common.fm.providers.bases.gen import GEN_OUTPUT_KEY_ATTR, GEN_PROMPT_ATTR, GEN_PROMPT_INDEX_ATTR, GEN_SOURCE_IMAGE_ATTR
from cvsuite.common.fm.providers.registry import (
    MODEL_MODULES,
    allowed_model_ids_for_provider,
    default_model_id_for_provider,
    resolve_model_module,
)


def _ctx(
    tmp_path: Path,
    *,
    request: FMRequest | None = None,
    options: object | None = None,
    prompt: str = "",
    batch_size: int = 2,
):
    request = request or FMRequest(provider="dummy", task="classify")
    return SimpleNamespace(
        request=request,
        options=options or SimpleNamespace(),
        prompt=prompt,
        batch_size=batch_size,
        caller_cwd=tmp_path,
        config_path=tmp_path / "config.yaml",
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
    )


class _DummyClassificationModel(BaseClassificationModel[object, object]):
    model_name = "dummy-classify"


class _DummyGroundModel(BaseGroundModel[object, object]):
    model_name = "dummy-ground"


class _DummyOCRModel(BaseOCRModel[object, object]):
    model_name = "dummy-ocr"


class _DummyVLMModel(BaseVLMModel[object, object]):
    model_name = "dummy-vlm"

    def load_runtime(self, dataset, ctx):
        return object()

    def process_batch(self, dataset, batch, runtime, ctx):
        return BatchResult()


class _DummyGenerationModel(BaseGenerationModel[object, object, object]):
    model_name = "dummy-gen"

    def build_jobs(self, dataset, ctx):
        return []

    def job_id(self, job, dataset, ctx):
        return "dummy"

    def load_runtime(self, dataset, ctx):
        return object()

    def process_batch(self, dataset, batch, runtime, ctx):
        return BatchResult()


class _DummyCreateGenerationModel(BaseCreateGenerationModel[object]):
    model_name = "dummy-gen-create"

    def load_runtime(self, dataset, ctx):
        return object()

    def process_batch(self, dataset, batch, runtime, ctx):
        return BatchResult()


class _DummyEditGenerationModel(BaseEditGenerationModel[object]):
    model_name = "dummy-gen-edit"

    def load_runtime(self, dataset, ctx):
        return object()

    def process_batch(self, dataset, batch, runtime, ctx):
        return BatchResult()


def test_model_registry_covers_all_current_models() -> None:
    assert MODEL_MODULES == {
        "blip": "cvsuite.common.fm.providers.vlm.blip",
        "clip": "cvsuite.common.fm.providers.classify.clip",
        "cogvlm": "cvsuite.common.fm.providers.vlm.cogvlm",
        "deep_orientation": "cvsuite.common.fm.providers.classify.deep_orientation",
        "dim_edit": "cvsuite.common.fm.providers.edit.dim_edit",
        "flux": "cvsuite.common.fm.providers.create.flux",
        "flux2_klein": "cvsuite.common.fm.providers.edit.flux2_klein",
        "gdino": "cvsuite.common.fm.providers.ground.gdino",
        "gsam": "cvsuite.common.fm.providers.ground.gsam",
        "instruct_pix2pix": "cvsuite.common.fm.providers.edit.instruct_pix2pix",
        "internvl": "cvsuite.common.fm.providers.vlm.internvl",
        "locate_anything": "cvsuite.common.fm.providers.ground.locate_anything",
        "llmdet": "cvsuite.common.fm.providers.ground.llmdet",
        "llava": "cvsuite.common.fm.providers.vlm.llava",
        "minicpm_v": "cvsuite.common.fm.providers.vlm.minicpm_v",
        "ovis_u1_3b": "cvsuite.common.fm.providers.edit.ovis_u1_3b",
        "paddleocr": "cvsuite.common.fm.providers.ocr.paddleocr",
        "paligemma": "cvsuite.common.fm.providers.vlm.paligemma",
        "qwen": "cvsuite.common.fm.providers.vlm.qwen",
        "qwen_image": "cvsuite.common.fm.providers.create.qwen_image",
        "qwen_image_edit": "cvsuite.common.fm.providers.edit.qwen_image_edit",
        "rex_omni": "cvsuite.common.fm.providers.ground.rex_omni",
        "sana": "cvsuite.common.fm.providers.create.sana",
        "sam3": "cvsuite.common.fm.providers.ground.sam3",
        "siglip2": "cvsuite.common.fm.providers.classify.siglip2",
        "stable_diffusion": "cvsuite.common.fm.providers.create.stable_diffusion",
        "step1x_edit": "cvsuite.common.fm.providers.edit.step1x_edit",
        "yolo_e": "cvsuite.common.fm.providers.ground.yolo_e",
    }
    assert resolve_model_module("clip") == "cvsuite.common.fm.providers.classify.clip"
    assert default_model_id_for_provider("llmdet", family="ground") == "iSEE-Laboratory/llmdet_base"
    assert allowed_model_ids_for_provider("llmdet", family="ground") == (
        "iSEE-Laboratory/llmdet_tiny",
        "iSEE-Laboratory/llmdet_base",
        "iSEE-Laboratory/llmdet_large",
    )
    assert default_model_id_for_provider("locate_anything", family="ground") == "nvidia/LocateAnything-3B"
    assert allowed_model_ids_for_provider("locate_anything", family="ground") == ("nvidia/LocateAnything-3B",)


def test_model_registry_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown FM provider"):
        resolve_model_module("not-a-model")


def test_base_classification_model_applies_threshold_and_meta(tmp_path: Path) -> None:
    model = _DummyClassificationModel()
    dataset = VisionDataset(records=[Record(image=ImageRecord(path=Path("img.jpg"), width=32, height=32))])
    ctx = _ctx(
        tmp_path,
        request=FMRequest(provider="dummy-classify", task="classify", device="cpu", precision="fp16"),
        options=SimpleNamespace(threshold=0.6),
    )

    modified = model.store_classification_result(
        dataset,
        RecordImageJob(record_idx=0, image_path=tmp_path / "img.jpg"),
        ctx=ctx,
        label="cat",
        score=0.55,
        probs={"cat": 0.55, "dog": 0.45},
        meta=model.build_classification_record_meta(ctx, candidate_labels=["cat", "dog"]),
        candidate_labels_default=["cat", "dog"],
    )

    assert modified == 0
    assert dataset.records[0].classification is not None
    assert dataset.records[0].classification.label is None
    assert dataset.records[0].classification.meta["threshold"] == pytest.approx(0.6)
    assert dataset.records[0].classification.meta["candidate_labels"] == ["cat", "dog"]
    assert dataset.records[0].attributes["fm_tasks"] == ["classify"]
    assert model.build_classification_fm_meta(ctx, candidate_labels=["cat", "dog"]) == {
        "task": "classify",
        "provider": "dummy-classify",
        "device": "cpu",
        "precision": "fp16",
        "batch_size": 2,
        "candidate_labels": ["cat", "dog"],
        "threshold": 0.6,
    }


def test_base_ground_model_helpers_manage_classes_shapes_and_tasks(tmp_path: Path) -> None:
    model = _DummyGroundModel()
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=Path("img.jpg"), width=100, height=50),
                attributes={"ground_prompts": ["metal can"]},
            )
        ]
    )
    ctx = _ctx(tmp_path, request=FMRequest(provider="dummy-ground", task="ground", device="cuda", precision="bf16"))
    record = dataset.records[0]
    class_to_id = {name: idx for idx, name in enumerate(dataset.classes)}
    cls_id = model.ensure_class_id(dataset, "metal can", class_to_id)
    gid = model.next_group_id(record)

    model.append_ground_box(
        record,
        box_xyxy=pytest.importorskip("numpy").array([10.0, 5.0, 50.0, 25.0]),
        width=100.0,
        height=50.0,
        cls_id=cls_id,
        label="metal can",
        score=0.9,
        group_id=gid,
        prompt="metal can",
    )
    poly_count = model.append_ground_polygons(
        record,
        polys=[[(10.0, 5.0), (50.0, 5.0), (50.0, 25.0)]],
        width=100.0,
        height=50.0,
        cls_id=cls_id,
        label="metal can",
        score=0.9,
        group_id=gid,
        prompt="metal can",
    )
    model.finalize_record_task(record, set_det_when_boxes=True)
    model.finalize_dataset_task(dataset, has_boxes_any=True, has_masks_any=True, set_det_when_boxes=True)

    assert cls_id == 0
    assert dataset.classes == ["metal can"]
    assert poly_count == 1
    assert record.boxes[0].label == "metal can"
    assert record.boxes[0].prompt == "metal can"
    assert record.polys[0].label == "metal can"
    assert record.polys[0].group_id == gid
    assert record.task == Task.seg
    assert dataset.task == Task.seg
    assert model.build_ground_fm_meta(dataset, ctx, task_name="sam3") == {
        "task": "sam3",
        "provider": "dummy-ground",
        "device": "cuda",
        "precision": "bf16",
        "prompts": ["metal can"],
    }


def test_base_ocr_model_helpers_add_boxes_and_tasks(tmp_path: Path) -> None:
    model = _DummyOCRModel()
    dataset = VisionDataset(records=[Record(image=ImageRecord(path=Path("img.jpg"), width=20, height=10))])
    ctx = _ctx(tmp_path, request=FMRequest(provider="dummy-ocr", task="ocr", device="cpu", precision="fp32"))
    record = dataset.records[0]
    cls_id = model.ensure_text_class(dataset)

    model.append_ocr_box(
        record,
        poly=[(1.0, 1.0), (9.0, 1.0), (9.0, 5.0), (1.0, 5.0)],
        width=20.0,
        height=10.0,
        cls_id=cls_id,
        text="HELLO",
        score=0.91,
        group_id=1,
        attributes={"polygon": [(1.0, 1.0), (9.0, 1.0)]},
    )
    model.finalize_record_task(record)
    model.finalize_dataset_task(dataset)

    assert dataset.classes == ["text"]
    assert record.task == Task.det
    assert dataset.task == Task.det
    assert record.boxes[0].kind == "ocr"
    assert record.boxes[0].label == "text"
    assert record.boxes[0].text == "HELLO"
    assert model.build_ocr_fm_meta(ctx) == {
        "task": "ocr",
        "provider": "dummy-ocr",
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 2,
    }


def test_base_vlm_model_collects_jobs_tracks_questions_and_appends_answers(tmp_path: Path) -> None:
    model = _DummyVLMModel()
    image_path = tmp_path / "img.jpg"
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=image_path.name, width=16, height=16))],
        root=tmp_path,
    )
    ctx = _ctx(
        tmp_path,
        request=FMRequest(provider="dummy-vlm", task="vlm", device="auto", precision="fp16"),
        prompt='{"binary": "Is there a can?"}',
        batch_size=1,
    )

    jobs = model.build_jobs(dataset, ctx)
    modified = model.append_answers(dataset, jobs, [" yes "])

    assert len(jobs) == 1
    assert jobs[0].question == "Is there a can?"
    assert jobs[0].label == "binary"
    assert model.job_id(jobs[0], dataset, ctx).startswith("record:0:prompt:")
    assert model.question_meta == [{"question": "Is there a can?", "label": "binary"}]
    assert modified == [0]
    assert dataset.records[0].vqas == [
        VQA(
            question="Is there a can?",
            answer="yes",
            score=None,
            model="dummy-vlm",
            meta={"label": "binary"},
        )
    ]
    assert dataset.records[0].attributes["fm_tasks"] == ["vlm"]
    assert model.build_vlm_meta(
        runtime=object(),
        ctx=ctx,
        hf_model_id="dummy/hf",
        precision=None,
    ) == {
        "task": "vlm",
        "provider": "dummy-vlm",
        "hf_model": "dummy/hf",
        "device": "auto",
        "precision": "fp16",
        "batch_size": 1,
        "weights": str(tmp_path / "weights"),
        "config": str(tmp_path / "config.yaml"),
        "questions": [{"question": "Is there a can?", "label": "binary"}],
        "managed_device_map_active": False,
        "memory_budget_source": "none",
        "effective_max_memory": None,
        "offload_cap_enabled": False,
        "offload_scope": None,
        "offload_folder": None,
        "allocator_hint_auto_applied": False,
        "max_gpu_memory": None,
    }


def test_base_generation_model_validates_request_and_builds_output_dataset(tmp_path: Path) -> None:
    model = _DummyGenerationModel()
    dataset = VisionDataset(
        records=[],
        meta={"gen_mode": "edit", "seed": 7},
        fm_request=FMRequest(provider="dummy-gen", task="gen", device="cpu", precision="fp32"),
    )
    ctx = _ctx(tmp_path, request=dataset.fm_request, batch_size=3)

    assert model.validate_generation_request(dataset, dataset.fm_request) == "edit"

    result = model.build_generated_dataset(
        dataset,
        image_path=tmp_path / "out.png",
        width=64,
        height=32,
        extra_meta={"scheduler": "ddim"},
    )

    assert result.root == tmp_path
    assert result.records[0].image.path == tmp_path / "out.png"
    assert result.meta["gen_mode"] == "edit"
    assert result.meta["seed"] == 7
    assert result.meta["scheduler"] == "ddim"
    assert result.fm_request == dataset.fm_request
    assert model.build_generation_fm_meta(
        dataset,
        config_path=tmp_path / "cfg.yaml",
        weights_dir=tmp_path / "weights",
        request=dataset.fm_request,
        batch_size=3,
    ) == {
        "task": "gen",
        "provider": "dummy-gen",
        "mode": "edit",
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 3,
        "weights": str(tmp_path / "weights"),
        "config": str(tmp_path / "cfg.yaml"),
    }


def test_base_generation_model_rejects_invalid_request_mode() -> None:
    model = _DummyGenerationModel()
    dataset = VisionDataset(records=[], meta={"gen_mode": "blend"})
    request = FMRequest(provider="dummy-gen", task="gen")

    with pytest.raises(ValueError, match="Unsupported generation mode"):
        model.validate_generation_request(dataset, request)

    with pytest.raises(ValueError, match="expects FMRequest.task='gen'"):
        model.validate_generation_request(dataset, FMRequest(provider="dummy-gen", task="vlm"))


def test_base_create_generation_model_accepts_only_create_mode(tmp_path: Path) -> None:
    model = _DummyCreateGenerationModel()
    create_dataset = VisionDataset(records=[], meta={"gen_mode": "create"})
    edit_dataset = VisionDataset(records=[], meta={"gen_mode": "edit"})
    request = FMRequest(provider="dummy-gen-create", task="gen")

    assert model.validate_generation_request(create_dataset, request) == "create"
    with pytest.raises(ValueError, match="Unsupported generation mode"):
        model.validate_generation_request(edit_dataset, request)


def test_base_edit_generation_model_accepts_only_edit_mode(tmp_path: Path) -> None:
    model = _DummyEditGenerationModel()
    create_dataset = VisionDataset(records=[], meta={"gen_mode": "create"})
    edit_dataset = VisionDataset(records=[], meta={"gen_mode": "edit"})
    request = FMRequest(provider="dummy-gen-edit", task="gen")

    assert model.validate_generation_request(edit_dataset, request) == "edit"
    with pytest.raises(ValueError, match="Unsupported generation mode"):
        model.validate_generation_request(create_dataset, request)


def test_base_edit_generation_model_build_jobs_prefers_preserved_source_image(tmp_path: Path) -> None:
    model = _DummyEditGenerationModel()
    source = tmp_path / "source.jpg"
    source.write_bytes(b"src")
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=Path("filled_0000_source_0001.png"), width=24, height=18),
                split="train",
                attributes={
                    GEN_PROMPT_ATTR: "fill the masked area",
                    GEN_PROMPT_INDEX_ATTR: 0,
                    GEN_OUTPUT_KEY_ATTR: "filled_0000_source_0001",
                    GEN_SOURCE_IMAGE_ATTR: "source.jpg",
                },
            )
        ],
        root=tmp_path,
        meta={"gen_mode": "edit"},
    )
    ctx = _ctx(tmp_path, request=FMRequest(provider="dummy-gen-edit", task="gen"))

    jobs = model.build_jobs(dataset, ctx)

    assert len(jobs) == 1
    assert jobs[0].source_image_path == source
