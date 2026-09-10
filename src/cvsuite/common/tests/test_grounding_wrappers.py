from __future__ import annotations

from contextlib import nullcontext
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from cvsuite.common.core.enums import Task
from cvsuite.common.core import FMRequest, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.providers.ground import sam3 as sam3_module
from cvsuite.common.fm.providers.ground.sam3 import (
    Sam3Job,
    Sam3Model,
    Sam3Options,
    Sam3Runtime,
    _patch_sam3_cpu_build_precompute,
    _wrap_sam3_fused_addmm_act,
)

try:
    from cvsuite.common.fm.providers.ground.gsam import GSAMConfig, GSAMJob, GSAMModel, GSAMRuntime
    from cvsuite.common.fm.providers.ground.llmdet import (
        DEFAULT_MODEL_ID as LLMDET_DEFAULT_MODEL_ID,
        LLMDetConfig,
        LLMDetJob,
        LLMDetModel,
        LLMDetOptions,
        LLMDetRuntime,
        _resolve_model_id as resolve_llmdet_model_id,
    )
    from cvsuite.common.fm.providers.ground import locate_anything as locate_anything_module
    from cvsuite.common.fm.providers.ground.locate_anything import (
        DEFAULT_MODEL_ID as LOCATE_ANYTHING_DEFAULT_MODEL_ID,
        LocateAnythingConfig,
        LocateAnythingJob,
        LocateAnythingModel,
        LocateAnythingOptions,
        LocateAnythingRuntime,
    )
except ModuleNotFoundError as exc:
    if exc.name == "transformers":
        pytest.skip("transformers is required for grounding wrapper unit tests", allow_module_level=True)
    raise

from cvsuite.common.fm.providers.ground import rex_omni as rex_omni_module
from cvsuite.common.fm.providers.ground.rex_omni import RexOmniModel, RexOmniOptions
from cvsuite.common.fm.providers.ground import yolo_e as yolo_e_module
from cvsuite.common.fm.providers.ground.yolo_e import (
    YoloEConfig,
    YoloEJob,
    YoloEModel,
    YoloEOptions,
    YoloERuntime,
)


def _write_image(path: Path) -> None:
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(path)


def test_sam3_process_batch_uses_exact_ground_prompt_as_label(tmp_path: Path) -> None:
    image_path = tmp_path / "sam3.jpg"
    _write_image(image_path)
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=16, height=16),
                attributes={"ground_prompts": ["metal can"]},
            )
        ],
        fm_request=FMRequest(provider="sam3", task="ground"),
    )

    class _Processor:
        def set_image(self, image):
            return {"image": image}

        def set_text_prompt(self, state, prompt):
            assert prompt == "metal can"
            return {
                "boxes": np.array([[1.0, 1.0, 8.0, 8.0]], dtype=np.float32),
                "scores": np.array([0.8], dtype=np.float32),
                "masks": np.array([np.ones((8, 8), dtype=np.uint8)]),
            }

    runtime = Sam3Runtime(device=torch.device("cpu"), autocast_ctx=nullcontext(), processor=_Processor())
    result = Sam3Model().process_batch(dataset, [Sam3Job(record_idx=0, image_path=image_path)], runtime, None)

    rec = dataset.records[0]
    assert result.modified_record_indices == [0]
    assert rec.boxes[0].label == "metal can"
    assert rec.boxes[0].prompt == "metal can"
    assert rec.polys[0].label == "metal can"
    assert rec.polys[0].prompt == "metal can"
    assert dataset.classes == ["metal can"]
    assert dataset.task == Task.seg


def test_sam3_process_batch_warns_with_max_score_when_prompt_falls_below_threshold(tmp_path: Path) -> None:
    image_path = tmp_path / "sam3-miss.jpg"
    _write_image(image_path)
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=16, height=16),
                attributes={"ground_prompts": ["fire alarm"]},
            )
        ],
        fm_request=FMRequest(provider="sam3", task="ground"),
    )

    class _Processor:
        def __init__(self) -> None:
            self.confidence_threshold = 0.2

        def set_image(self, image):
            return {"image": image}

        def set_text_prompt(self, state, prompt):
            assert prompt == "fire alarm"
            assert self.confidence_threshold == -1.0
            return {
                "boxes": np.array([[1.0, 1.0, 8.0, 8.0]], dtype=np.float32),
                "scores": np.array([0.12], dtype=np.float32),
                "masks": np.array([np.ones((8, 8), dtype=np.uint8)]),
            }

    processor = _Processor()
    runtime = Sam3Runtime(device=torch.device("cpu"), autocast_ctx=nullcontext(), processor=processor)
    result = Sam3Model().process_batch(dataset, [Sam3Job(record_idx=0, image_path=image_path)], runtime, None)

    assert result.modified_record_indices == [0]
    assert dataset.records[0].boxes == []
    assert any("max_score=0.120" in warning for warning in result.warnings)
    assert any("threshold=0.200" in warning for warning in result.warnings)
    assert processor.confidence_threshold == pytest.approx(0.2)


def test_sam3_load_runtime_rejects_cpu_request(monkeypatch) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="sam3", task="ground", device="cpu"))
    ctx = SimpleNamespace(request=dataset.fm_request, weights_dir=None)

    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.build_sam3_image_model", lambda **kwargs: object())
    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.Sam3Processor", lambda *args, **kwargs: object())

    with pytest.raises(RuntimeError, match="sam3 does not support CPU execution"):
        Sam3Model().load_runtime(dataset, ctx)


def test_sam3_load_runtime_rejects_auto_without_cuda(monkeypatch) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="sam3", task="ground", device="auto"))
    ctx = SimpleNamespace(request=dataset.fm_request, weights_dir=None)

    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.build_sam3_image_model", lambda **kwargs: object())
    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.Sam3Processor", lambda *args, **kwargs: object())
    monkeypatch.setattr(torch.version, "cuda", "12.6", raising=False)
    monkeypatch.setattr(sam3_module.utils, "select_device", lambda pref: torch.device("cpu"))

    with pytest.raises(RuntimeError, match="sam3 requires CUDA"):
        Sam3Model().load_runtime(dataset, ctx)


def test_sam3_load_runtime_uses_loaded_model_dtype_for_autocast(monkeypatch) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="sam3", task="ground", device="cuda", precision="bf16"))
    ctx = SimpleNamespace(request=dataset.fm_request, weights_dir=None, options=Sam3Options())
    model = torch.nn.Linear(4, 4, bias=False).to(dtype=torch.float32)
    sentinel = object()

    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.build_sam3_image_model", lambda **kwargs: model)
    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.Sam3Processor", lambda *args, **kwargs: object())
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(sam3_module, "_cuda_arch_list", lambda: {"sm_80"})
    monkeypatch.setattr(sam3_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(sam3_module.utils, "maybe_autocast_for_module", lambda device, precision, module: sentinel)

    runtime = Sam3Model().load_runtime(dataset, ctx)

    assert runtime.autocast_ctx is sentinel


def test_sam3_load_runtime_forwards_confidence_threshold_to_processor(monkeypatch) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="sam3", task="ground", device="cuda"))
    ctx = SimpleNamespace(request=dataset.fm_request, weights_dir=None, options=Sam3Options(confidence_threshold=0.125))
    model = object()
    captured = {}

    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.build_sam3_image_model", lambda **kwargs: model)
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(sam3_module, "_cuda_arch_list", lambda: {"sm_80"})
    monkeypatch.setattr(sam3_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(sam3_module.utils, "maybe_autocast_for_module", lambda device, precision, module: nullcontext())

    def _fake_processor(model_arg, **kwargs):
        captured["model"] = model_arg
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.Sam3Processor", _fake_processor)

    Sam3Model().load_runtime(dataset, ctx)

    assert captured["model"] is model
    assert captured["kwargs"]["confidence_threshold"] == pytest.approx(0.125)


def test_sam3_load_runtime_prefers_durable_weights_cache_for_hf_assets(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="sam3", task="ground", device="cuda"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        stage_dir=tmp_path / "stage",
        options=Sam3Options(),
    )
    model = object()
    captured = {}

    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.build_sam3_image_model", lambda **kwargs: model)
    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.Sam3Processor", lambda *args, **kwargs: object())
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(sam3_module, "_cuda_arch_list", lambda: {"sm_80"})
    monkeypatch.setattr(sam3_module, "_patch_sam3_fused_addmm_dtype_compat", lambda: None)
    monkeypatch.setattr(sam3_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(sam3_module.utils, "maybe_autocast_for_module", lambda device, precision, module: nullcontext())
    monkeypatch.setattr(sam3_module.utils, "ensure_hf_caches", lambda path: captured.setdefault("cache_root", path))
    monkeypatch.delenv("SAM3_CACHE", raising=False)

    Sam3Model().load_runtime(dataset, ctx)

    assert captured["cache_root"] == ctx.weights_dir
    assert os.environ["SAM3_CACHE"] == str(ctx.weights_dir)


def test_patch_sam3_cpu_build_precompute_disables_cuda_only_constructor_warmups(monkeypatch) -> None:
    captured = {}

    class _FakePositionEmbeddingSine:
        def __init__(self, *args, **kwargs):
            captured["precompute_resolution"] = kwargs.get("precompute_resolution")

    class _FakeTransformerDecoder:
        def __init__(self, *args, **kwargs):
            captured["resolution"] = kwargs.get("resolution")
            captured["stride"] = kwargs.get("stride")

    fake_position_module = types.SimpleNamespace(PositionEmbeddingSine=_FakePositionEmbeddingSine)
    fake_decoder_module = types.SimpleNamespace(TransformerDecoder=_FakeTransformerDecoder)
    monkeypatch.setattr(
        sam3_module.importlib,
        "import_module",
        lambda name: {
            "sam3.model.position_encoding": fake_position_module,
            "sam3.model.decoder": fake_decoder_module,
        }[name],
    )

    with _patch_sam3_cpu_build_precompute():
        _FakePositionEmbeddingSine(precompute_resolution=1008)
        _FakeTransformerDecoder(resolution=1008, stride=14)

    assert captured["precompute_resolution"] is None
    assert captured["resolution"] is None
    assert captured["stride"] is None


def test_sam3_build_model_cpu_disables_upstream_cuda_precompute(monkeypatch) -> None:
    captured = {}

    class _FakePositionEmbeddingSine:
        def __init__(self, *args, **kwargs):
            captured["precompute_resolution"] = kwargs.get("precompute_resolution")

    class _FakeTransformerDecoder:
        def __init__(self, *args, **kwargs):
            captured["resolution"] = kwargs.get("resolution")
            captured["stride"] = kwargs.get("stride")

    fake_position_module = types.SimpleNamespace(PositionEmbeddingSine=_FakePositionEmbeddingSine)
    fake_decoder_module = types.SimpleNamespace(TransformerDecoder=_FakeTransformerDecoder)
    monkeypatch.setattr(
        sam3_module.importlib,
        "import_module",
        lambda name: {
            "sam3.model.position_encoding": fake_position_module,
            "sam3.model.decoder": fake_decoder_module,
        }[name],
    )

    def _fake_build_model(**kwargs):
        captured["build_device"] = kwargs["device"]
        _FakePositionEmbeddingSine(precompute_resolution=1008)
        _FakeTransformerDecoder(resolution=1008, stride=14)
        return object()

    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.build_sam3_image_model", _fake_build_model)

    Sam3Model()._build_sam3_model(device_arg="cpu")

    assert captured["build_device"] == "cpu"
    assert captured["precompute_resolution"] is None
    assert captured["resolution"] is None
    assert captured["stride"] is None


def test_sam3_load_runtime_auto_dispatches_model(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="sam3", task="ground", device="auto"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        stage_dir=tmp_path / "stage",
        model_cache=tmp_path / "cache",
        options=Sam3Options(),
    )
    model = torch.nn.Linear(4, 4, bias=False)
    captured = {}

    def _fake_build_model(**kwargs):
        captured["build_kwargs"] = kwargs
        return model

    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.build_sam3_image_model", _fake_build_model)
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(sam3_module, "_cuda_arch_list", lambda: {"sm_80"})
    monkeypatch.setattr(sam3_module, "_patch_sam3_fused_addmm_dtype_compat", lambda: None)
    monkeypatch.setattr(sam3_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(sam3_module.utils, "maybe_autocast_for_module", lambda device, precision, module: nullcontext())
    monkeypatch.setattr(Sam3Model, "ensure_optional_dependency", lambda self, **kwargs: captured.setdefault("dependency", kwargs))

    def _fake_build_hf_load_kwargs(self, ctx_arg, device_arg, precision_arg, **kwargs):
        placement = SimpleNamespace(
            device_map="auto",
            max_memory={0: 1234, "cpu": 5678},
            offload_folder=str(tmp_path / "stage" / "offload"),
            offload_cap_enabled=True,
            managed_device_map_active=True,
            memory_budget_source="inferred",
        )
        self._last_hf_load_placement = placement
        return {}, placement

    monkeypatch.setattr(Sam3Model, "build_hf_load_kwargs", _fake_build_hf_load_kwargs)
    def _fake_infer_device_map(model_arg, *, max_memory):
        captured["max_memory"] = max_memory
        return {"": 0}

    monkeypatch.setattr(sam3_module, "_accelerate_infer_auto_device_map", _fake_infer_device_map)

    def _fake_dispatch(model_arg, *, device_map, offload_dir):
        captured["device_map"] = device_map
        captured["offload_dir"] = offload_dir
        model_arg.hf_device_map = {"": 0}
        return model_arg

    monkeypatch.setattr(sam3_module, "_accelerate_dispatch_model", _fake_dispatch)
    monkeypatch.setattr(Sam3Model, "emit_hf_auto_device_report", lambda self, model_arg, placement=None: captured.setdefault("reported", True))

    def _fake_processor(model_arg, **kwargs):
        captured["processor_model"] = model_arg
        captured["processor_kwargs"] = kwargs
        return object()

    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.Sam3Processor", _fake_processor)

    runtime = Sam3Model().load_runtime(dataset, ctx)

    assert isinstance(runtime.device, torch.device)
    assert runtime.device.type == "cuda"
    assert captured["build_kwargs"]["device"] == "cpu"
    assert captured["max_memory"] == {0: 1234, "cpu": 5678}
    assert captured["device_map"] == {"": 0}
    assert captured["offload_dir"] == str(tmp_path / "stage" / "offload")
    assert captured["processor_model"] is model
    assert captured["processor_kwargs"]["device"] == "cuda"
    assert captured["reported"] is True


def test_sam3_load_runtime_auto_falls_back_to_cpu_for_mixed_cpu_gpu_map(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="sam3", task="ground", device="auto"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        stage_dir=tmp_path / "stage",
        model_cache=tmp_path / "cache",
        options=Sam3Options(),
    )
    model = torch.nn.Linear(4, 4, bias=False)
    captured = {}

    def _fake_build_model(**kwargs):
        captured.setdefault("build_devices", []).append(kwargs["device"])
        return model

    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.build_sam3_image_model", _fake_build_model)
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(sam3_module, "_cuda_arch_list", lambda: {"sm_80"})
    monkeypatch.setattr(sam3_module, "_patch_sam3_fused_addmm_dtype_compat", lambda: None)
    monkeypatch.setattr(sam3_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(sam3_module.utils, "maybe_autocast_for_module", lambda device, precision, module: nullcontext())
    monkeypatch.setattr(Sam3Model, "ensure_optional_dependency", lambda self, **kwargs: captured.setdefault("dependency", kwargs))

    def _fake_build_hf_load_kwargs(self, ctx_arg, device_arg, precision_arg, **kwargs):
        placement = SimpleNamespace(
            device_map="auto",
            max_memory={0: 1234, "cpu": 5678},
            offload_folder=str(tmp_path / "stage" / "offload"),
            offload_cap_enabled=True,
            managed_device_map_active=True,
            memory_budget_source="inferred",
        )
        self._last_hf_load_placement = placement
        return {}, placement

    monkeypatch.setattr(Sam3Model, "build_hf_load_kwargs", _fake_build_hf_load_kwargs)
    monkeypatch.setattr(sam3_module, "_accelerate_infer_auto_device_map", lambda model_arg, *, max_memory: {"backbone": 0, "decoder": "cpu"})
    monkeypatch.setattr(
        Sam3Model,
        "emit_runtime_warning",
        lambda self, dataset_arg, message: captured.setdefault("warning", message),
    )
    monkeypatch.setattr(
        Sam3Model,
        "emit_hf_auto_device_report",
        lambda self, model_arg, placement=None: captured.setdefault("reported", True),
    )

    def _fake_processor(model_arg, **kwargs):
        captured["processor_model"] = model_arg
        captured["processor_kwargs"] = kwargs
        return object()

    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.Sam3Processor", _fake_processor)

    runtime = Sam3Model().load_runtime(dataset, ctx)

    assert runtime.device.type == "cpu"
    assert runtime.managed_auto_fallback == "cpu"
    assert captured["build_devices"] == ["cpu"]
    assert "Falling back to CPU execution" in captured["warning"]
    assert captured["processor_model"] is model
    assert captured["processor_kwargs"]["device"] == "cpu"
    assert "reported" not in captured


def test_sam3_load_runtime_surfaces_gated_hf_access_error(monkeypatch) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="sam3", task="ground", device="cuda"))
    ctx = SimpleNamespace(request=dataset.fm_request, weights_dir=None, options=Sam3Options())

    class _Unauthorized(Exception):
        def __init__(self) -> None:
            super().__init__(
                "Cannot access gated repo for url https://huggingface.co/facebook/sam3/resolve/main/config.json."
            )
            self.response = SimpleNamespace(status_code=401)

    monkeypatch.setattr(
        "cvsuite.common.fm.providers.ground.sam3.build_sam3_image_model",
        lambda **kwargs: (_ for _ in ()).throw(_Unauthorized()),
    )
    monkeypatch.setattr("cvsuite.common.fm.providers.ground.sam3.Sam3Processor", lambda *args, **kwargs: object())
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(sam3_module, "_cuda_arch_list", lambda: {"sm_80"})
    monkeypatch.setattr(sam3_module, "_patch_sam3_fused_addmm_dtype_compat", lambda: None)
    monkeypatch.setattr(sam3_module.utils, "select_device", lambda pref: torch.device("cuda"))

    with pytest.raises(RuntimeError, match="facebook/sam3") as excinfo:
        Sam3Model().load_runtime(dataset, ctx)

    assert "HF_TOKEN/HUGGINGFACE_HUB_TOKEN" in str(excinfo.value)
    assert ".env" in str(excinfo.value)


def test_wrap_sam3_fused_addmm_act_restores_linear_weight_dtype() -> None:
    linear = torch.nn.Linear(4, 3, bias=False).to(dtype=torch.float32)

    def _fake_addmm_act(activation, linear_module, mat1):
        assert linear_module is linear
        return torch.ones((2, 3), dtype=torch.bfloat16)

    wrapped = _wrap_sam3_fused_addmm_act(_fake_addmm_act)
    output = wrapped(None, linear, torch.ones((2, 4), dtype=torch.float32))

    assert output.dtype == torch.float32


def test_sam3_patch_updates_fused_and_vitdet_modules(monkeypatch) -> None:
    fused_mod = types.ModuleType("sam3.perflib.fused")
    vitdet_mod = types.ModuleType("sam3.model.vitdet")

    def _fake_addmm_act(activation, linear_module, mat1):
        return torch.ones((2, 3), dtype=torch.bfloat16)

    fused_mod.addmm_act = _fake_addmm_act
    vitdet_mod.addmm_act = _fake_addmm_act

    monkeypatch.setattr(sam3_module, "_SAM3_FUSED_ADDMM_PATCHED", False)
    monkeypatch.setattr(sam3_module.importlib, "import_module", lambda name: {"sam3.perflib.fused": fused_mod, "sam3.model.vitdet": vitdet_mod}[name])

    sam3_module._patch_sam3_fused_addmm_dtype_compat()

    assert fused_mod.addmm_act is vitdet_mod.addmm_act
    assert getattr(fused_mod.addmm_act, "_cvsuite_preserves_output_dtype", False) is True


def test_gsam_process_batch_uses_exact_ground_prompt_as_label(tmp_path: Path) -> None:
    image_path = tmp_path / "gsam.jpg"
    _write_image(image_path)
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=16, height=16),
                attributes={"ground_prompts": ["rust patch"]},
            )
        ],
        fm_request=FMRequest(provider="gsam", task="ground"),
    )

    class _Processor:
        def __call__(self, images=None, text=None, return_tensors=None):
            if images is not None:
                return {"pixel_values": torch.ones((1, 3, 8, 8), dtype=torch.float32)}
            assert text == "rust patch."
            return {"input_ids": torch.tensor([[1]], dtype=torch.int64)}

        def post_process_grounded_object_detection(self, outputs, input_ids, threshold, text_threshold, target_sizes):
            return [
                {
                    "boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]], dtype=torch.float32),
                    "scores": torch.tensor([0.9], dtype=torch.float32),
                }
            ]

    class _GDino:
        def __call__(self, **inputs):
            return {"ok": True}

    class _Predictor:
        def set_image(self, image):
            return None

        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
            return np.ones((1, 1, 8, 8), dtype=np.uint8), None, None

    runtime = GSAMRuntime(
        cfg=GSAMConfig(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        autocast_ctx=nullcontext(),
        processor=_Processor(),
        gdino=_GDino(),
        sam_predictor=_Predictor(),
        resolved_cfg_path=None,
        resolved_ckpt_path=None,
    )
    result = GSAMModel().process_batch(dataset, [GSAMJob(record_idx=0, image_path=image_path)], runtime, None)

    rec = dataset.records[0]
    assert result.modified_record_indices == [0]
    assert rec.boxes[0].label == "rust patch"
    assert rec.boxes[0].prompt == "rust patch"
    assert rec.polys[0].label == "rust patch"
    assert rec.polys[0].prompt == "rust patch"
    assert dataset.classes == ["rust patch"]
    assert dataset.task == Task.seg


def test_llmdet_resolves_default_and_override_model_ids() -> None:
    assert resolve_llmdet_model_id(LLMDetConfig(), LLMDetOptions()) == LLMDET_DEFAULT_MODEL_ID
    assert (
        resolve_llmdet_model_id(
            LLMDetConfig(model_id="iSEE-Laboratory/llmdet_tiny"),
            LLMDetOptions(model_id="iSEE-Laboratory/llmdet_large"),
        )
        == "iSEE-Laboratory/llmdet_large"
    )
    with pytest.raises(ValueError, match="Unsupported LLMDet model_id"):
        resolve_llmdet_model_id(LLMDetConfig(), LLMDetOptions(model_id="not-real"))


class _FakeRexOmniWrapper:
    def __init__(self, captured: dict):
        self._captured = captured

    def __call__(self, **kwargs):
        self._captured.update(kwargs)
        self.model = SimpleNamespace(generate=lambda **kw: None)
        return self


def test_rex_omni_load_runtime_forwards_requested_precision_dtype(monkeypatch) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(provider="rex_omni", task="ground", device="cuda", precision="bf16"),
    )
    ctx = SimpleNamespace(request=dataset.fm_request, config_path=None, weights_dir=None, options=RexOmniOptions())

    captured: dict[str, object] = {}
    monkeypatch.setitem(sys.modules, "rex_omni", types.SimpleNamespace(RexOmniWrapper=_FakeRexOmniWrapper(captured)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)
    monkeypatch.setattr(rex_omni_module.utils, "select_device", lambda pref, force_cpu=False: torch.device("cuda"))

    runtime = RexOmniModel().load_runtime(dataset, ctx)

    assert captured["model_path"] == "IDEA-Research/Rex-Omni"
    assert captured["torch_dtype"] is torch.bfloat16
    assert captured["device_map"] == "cuda"
    assert captured["attn_implementation"] == "sdpa"
    assert runtime.device.type == "cuda"
    assert isinstance(runtime.model.model.generate, rex_omni_module._ScoreCapture)


def test_rex_omni_load_runtime_falls_back_to_fp32_off_gpu(monkeypatch) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(provider="rex_omni", task="ground", device="cpu", precision="bf16"),
    )
    ctx = SimpleNamespace(request=dataset.fm_request, config_path=None, weights_dir=None, options=RexOmniOptions())

    captured: dict[str, object] = {}
    monkeypatch.setitem(sys.modules, "rex_omni", types.SimpleNamespace(RexOmniWrapper=_FakeRexOmniWrapper(captured)))

    RexOmniModel().load_runtime(dataset, ctx)

    assert captured["torch_dtype"] is torch.float32
    assert captured["device_map"] == "cpu"


def test_rex_omni_box_confidences_align_coordinate_tokens() -> None:
    import math

    raw = (
        "<|object_ref_start|>widget<|object_ref_end|>"
        "<|box_start|><10><20><30><40>, <5><6>, <1><2><3><4><|box_end|>"
        "<|object_ref_start|>cable<|object_ref_end|>"
        "<|box_start|><7><8><9><11><|box_end|><|im_end|>"
    )
    first_comma = raw.index(",")
    pieces = list(raw)
    logprobs = [
        math.log(0.9) if (ch.isdigit() and idx < first_comma) else math.log(0.4) if ch.isdigit() else -50.0
        for idx, ch in enumerate(raw)
    ]

    confs = rex_omni_module._box_confidences(raw, pieces, logprobs)

    assert confs is not None
    assert set(confs) == {"widget", "cable"}
    assert confs["widget"] == pytest.approx([0.9, 0.4])  # point segment <5><6> skipped
    assert confs["cable"] == pytest.approx([0.4])

    assert rex_omni_module._box_confidences("different text", pieces, logprobs) is None


def test_rex_omni_process_batch_scores_boxes_from_token_logprobs(monkeypatch, tmp_path: Path) -> None:
    import math

    image_path = tmp_path / "rex.jpg"
    _write_image(image_path)
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=16, height=16),
                attributes={"ground_prompts": ["widget", "cable"]},
            )
        ],
        fm_request=FMRequest(provider="rex_omni", task="ground"),
    )

    raw = "<|object_ref_start|>widget<|object_ref_end|><|box_start|><0><0><499><499><|box_end|><|im_end|>"

    class _FakeInference:
        processor = SimpleNamespace(tokenizer=object())

        def inference(self, images=None, task=None, categories=None):
            assert task == "detection"
            assert categories == ["widget", "cable"]
            return [
                {
                    "success": True,
                    "raw_output": raw,
                    "extracted_predictions": {
                        "widget": [{"type": "box", "coords": [0.0, 0.0, 8.0, 8.0]}],
                        # cable box has no counterpart in raw_output -> alignment fallback
                        "cable": [{"type": "box", "coords": [1.0, 1.0, 4.0, 4.0]}],
                    },
                }
            ]

    monkeypatch.setattr(
        rex_omni_module,
        "_decode_token_logprobs",
        lambda capture, tokenizer: (list(raw), [math.log(0.7) if ch.isdigit() else -50.0 for ch in raw]),
    )

    runtime = rex_omni_module.RexOmniRuntime(
        cfg=rex_omni_module.RexOmniConfig(),
        device=torch.device("cpu"),
        model=_FakeInference(),
        score_capture=object(),
    )
    result = RexOmniModel().process_batch(
        dataset, [rex_omni_module.RexOmniJob(record_idx=0, image_path=image_path)], runtime, None
    )

    rec = dataset.records[0]
    assert result.modified_record_indices == [0]
    assert [box.label for box in rec.boxes] == ["widget", "cable"]
    assert rec.boxes[0].score == pytest.approx(0.7)
    assert rec.boxes[1].score == pytest.approx(1.0)
    assert any("confidence alignment failed for 'cable'" in w for w in result.warnings)


def test_llmdet_process_batch_uses_prompt_as_label(tmp_path: Path) -> None:
    image_path = tmp_path / "llmdet.jpg"
    _write_image(image_path)
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=16, height=16),
                attributes={"ground_prompts": ["rust patch", "widget"]},
            )
        ],
        fm_request=FMRequest(provider="llmdet", task="ground"),
    )

    # input_ids: [CLS] rust patch . widget . [SEP] -> label spans [1, 2] and [4]
    input_ids = torch.tensor([[101, 7, 8, 1012, 9, 1012, 102]], dtype=torch.int64)
    # query0 peaks on "widget" tokens, query1 on "rust patch" tokens
    logits = torch.full((1, 2, 7), -10.0, dtype=torch.float32)
    logits[0, 0, 4] = 2.0
    logits[0, 1, 1] = 1.5

    class _Processor:
        tokenizer = SimpleNamespace(
            convert_tokens_to_ids=lambda token: 1012,
            all_special_ids=[101, 102],
        )

        def __call__(self, images=None, text=None, return_tensors=None):
            assert images is not None
            assert text == [["rust patch", "widget"]]
            return {
                "pixel_values": torch.ones((1, 3, 8, 8), dtype=torch.float32),
                "input_ids": input_ids,
            }

        def post_process_grounded_object_detection(self, outputs, input_ids, threshold, text_threshold, target_sizes):
            assert threshold == 0.4
            assert text_threshold == 0.3
            assert target_sizes == [(16, 16)]
            return [
                {
                    "boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0], [2.0, 2.0, 6.0, 6.0]], dtype=torch.float32),
                    "scores": torch.sigmoid(torch.tensor([2.0, 1.5], dtype=torch.float32)),
                }
            ]

    class _LLMDet:
        def __call__(self, **inputs):
            assert "pixel_values" in inputs
            assert "input_ids" in inputs
            return SimpleNamespace(logits=logits)

    runtime = LLMDetRuntime(
        cfg=LLMDetConfig(),
        model_id=LLMDET_DEFAULT_MODEL_ID,
        device=torch.device("cpu"),
        dtype=torch.float32,
        autocast_ctx=nullcontext(),
        processor=_Processor(),
        model=_LLMDet(),
    )
    result = LLMDetModel().process_batch(dataset, [LLMDetJob(record_idx=0, image_path=image_path)], runtime, None)

    rec = dataset.records[0]
    assert result.modified_record_indices == [0]
    assert rec.boxes[0].label == "widget"
    assert rec.boxes[0].prompt == "widget"
    assert rec.boxes[0].score == pytest.approx(float(torch.sigmoid(torch.tensor(2.0))))
    assert rec.boxes[1].label == "rust patch"
    assert rec.boxes[1].prompt == "rust patch"
    assert rec.attributes["fm_tasks"] == ["llmdet"]
    assert dataset.classes == ["rust patch", "widget"]
    assert dataset.task == Task.det


def test_locate_anything_parse_boxes_scales_normalized_coords() -> None:
    boxes, warnings = locate_anything_module._parse_boxes(
        "ok <box><100><200><900><800></box>",
        width=200.0,
        height=50.0,
    )

    np.testing.assert_allclose(boxes[0], np.asarray([20.0, 10.0, 180.0, 40.0]))
    assert warnings == []


def test_locate_anything_parse_boxes_drops_malformed_boxes() -> None:
    boxes, warnings = locate_anything_module._parse_boxes(
        "<box><900><200><100><800></box> <box><100><200><900><800></box>",
        width=200.0,
        height=50.0,
    )

    np.testing.assert_allclose(boxes[0], np.asarray([20.0, 10.0, 180.0, 40.0]))
    assert len(boxes) == 1
    assert any("inverted" in w for w in warnings)


def test_locate_anything_process_batch_uses_prompt_as_label(tmp_path: Path) -> None:
    image_path = tmp_path / "locate-anything.jpg"
    _write_image(image_path)
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=20, height=10),
                attributes={"ground_prompts": ["widget"]},
            )
        ],
        fm_request=FMRequest(provider="locate_anything", task="ground"),
    )
    captured: dict[str, object] = {}

    class _Processor:
        def py_apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            captured["messages"] = messages
            assert tokenize is False
            assert add_generation_prompt is True
            return "chat"

        def process_vision_info(self, messages):
            return [messages[0]["content"][0]["image"]], None

        def __call__(self, text=None, images=None, videos=None, return_tensors=None):
            assert text == ["chat"]
            assert len(images) == 1
            assert videos is None
            assert return_tensors == "pt"
            return {
                "pixel_values": torch.ones((1, 3, 4, 4), dtype=torch.float32),
                "input_ids": torch.ones((1, 2), dtype=torch.int64),
                "attention_mask": torch.ones((1, 2), dtype=torch.int64),
                "image_grid_hws": np.asarray([[1, 1]], dtype=np.int32),
            }

    class _Model:
        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return ["<box><100><200><500><600></box>"]

    precision = locate_anything_module.utils.resolve_precision(torch.device("cpu"), "fp32", allow_nf4=True)
    runtime = LocateAnythingRuntime(
        cfg=LocateAnythingConfig(max_new_tokens=32),
        model_id=LOCATE_ANYTHING_DEFAULT_MODEL_ID,
        device=torch.device("cpu"),
        model_device=torch.device("cpu"),
        dtype=torch.float32,
        precision=precision,
        autocast_ctx=nullcontext(),
        tokenizer=object(),
        processor=_Processor(),
        model=_Model(),
    )

    result = LocateAnythingModel().process_batch(
        dataset,
        [LocateAnythingJob(record_idx=0, image_path=image_path)],
        runtime,
        None,
    )

    rec = dataset.records[0]
    assert result.modified_record_indices == [0]
    assert rec.boxes[0].label == "widget"
    assert rec.boxes[0].prompt == "widget"
    assert rec.boxes[0].score is None
    assert rec.boxes[0].cx == pytest.approx(0.3)
    assert rec.boxes[0].cy == pytest.approx(0.4)
    assert rec.boxes[0].w == pytest.approx(0.4)
    assert rec.boxes[0].h == pytest.approx(0.4)
    assert rec.attributes["fm_tasks"] == ["locate-anything"]
    assert dataset.classes == ["widget"]
    assert dataset.task == Task.det
    assert captured["generate_kwargs"]["max_new_tokens"] == 32
    assert captured["generate_kwargs"]["generation_mode"] == "hybrid"
    assert "Locate all the instances that match the following description: widget." in str(captured["messages"])


class _FakeLocateAnythingModel:
    def __init__(self) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1))

    def eval(self) -> None:
        return None

    def parameters(self):
        return iter([self._param])


def _locate_anything_ctx(tmp_path: Path, request: FMRequest) -> SimpleNamespace:
    return SimpleNamespace(
        request=request,
        config_path=None,
        options=LocateAnythingOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        model_cache=tmp_path / "cache" / "locate_anything",
        stage_dir=tmp_path / "stage",
    )


class _FakeLocateAnythingSubConfig:
    pass


class _FakeLocateAnythingHFConfig:
    def __init__(self) -> None:
        self.text_config = _FakeLocateAnythingSubConfig()
        self.vision_config = _FakeLocateAnythingSubConfig()


def _fake_locate_anything_source(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    return locate_anything_module.utils.HFModelSource(
        load_arg=LOCATE_ANYTHING_DEFAULT_MODEL_ID,
        cache_dir=str(tmp_path / "weights"),
        revision=None,
        local_files_only=True,
        snapshot_path=snapshot,
        repo_cache_dir=tmp_path / "weights" / "models--nvidia--LocateAnything-3B",
    )


def _patch_locate_anything_load_deps(monkeypatch, tmp_path: Path, captured: dict[str, object]):
    config = _FakeLocateAnythingHFConfig()
    source = _fake_locate_anything_source(tmp_path)
    monkeypatch.setattr(locate_anything_module.utils, "materialize_hf_model_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(locate_anything_module.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(locate_anything_module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    def _fake_processor_from_pretrained(*args, **kwargs):
        captured["processor_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(locate_anything_module.AutoProcessor, "from_pretrained", _fake_processor_from_pretrained)
    return config


def _patch_locate_anything_nf4_precision(monkeypatch) -> SimpleNamespace:
    qcfg = SimpleNamespace(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    precision = locate_anything_module.utils.ResolvedPrecision(
        requested="nf4",
        normalized="nf4",
        effective="nf4",
        compute_dtype=torch.float16,
        quantization_mode="nf4",
        quantization_config=qcfg,
    )
    monkeypatch.setattr(LocateAnythingModel, "resolve_runtime_precision", lambda self, dataset, ctx, device: precision)
    return qcfg


def _write_locate_anything_remote_patch_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "image_processing_locateanything.py").write_text(
        'AutoImageProcessor.register("LocateAnythingImageProcessor", LocateAnythingImageProcessor)\n',
        encoding="utf-8",
    )
    (root / "modeling_qwen2.py").write_text(
        "from transformers.modeling_utils import PreTrainedModel\n\n"
        "class Qwen2ForCausalLM(Qwen2PreTrainedModel):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (root / "generate_utils.py").write_text(
        "box_avg.append(torch.tensor(out_ref, dtype=x0.dtype, device=x0.device))\n",
        encoding="utf-8",
    )


def test_locate_anything_remote_code_patches_are_idempotent(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    stage_remote = tmp_path / "stage" / "hf_modules" / "transformers_modules" / "nvidia" / "LocateAnything"
    _write_locate_anything_remote_patch_fixture(snapshot)
    _write_locate_anything_remote_patch_fixture(stage_remote)
    source = locate_anything_module.utils.HFModelSource(
        load_arg=LOCATE_ANYTHING_DEFAULT_MODEL_ID,
        cache_dir=None,
        revision=None,
        local_files_only=True,
        snapshot_path=snapshot,
        repo_cache_dir=None,
    )

    locate_anything_module._patch_locate_anything_remote_code(source, tmp_path / "stage")
    locate_anything_module._patch_locate_anything_remote_code(source, tmp_path / "stage")

    for root in (snapshot, stage_remote):
        image_text = (root / "image_processing_locateanything.py").read_text(encoding="utf-8")
        qwen_text = (root / "modeling_qwen2.py").read_text(encoding="utf-8")
        generate_text = (root / "generate_utils.py").read_text(encoding="utf-8")
        assert "slow_image_processor_class=LocateAnythingImageProcessor" in image_text
        assert 'AutoImageProcessor.register("LocateAnythingImageProcessor", LocateAnythingImageProcessor)' not in image_text
        assert qwen_text.count("from transformers.generation import GenerationMixin") == 1
        assert "class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):" in qwen_text
        assert "torch.as_tensor(out_ref, dtype=x0.dtype, device=x0.device)" in generate_text
        assert "torch.tensor(out_ref, dtype=x0.dtype, device=x0.device)" not in generate_text


def test_locate_anything_load_runtime_forwards_requested_precision(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(provider="locate_anything", task="ground", device="cuda", precision="fp16"),
    )
    ctx = _locate_anything_ctx(tmp_path, dataset.fm_request)
    captured: dict[str, object] = {}
    hf_config = _patch_locate_anything_load_deps(monkeypatch, tmp_path, captured)

    monkeypatch.setattr(locate_anything_module.utils, "select_device", lambda pref, force_cpu=False: torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _fake_model_from_pretrained(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeLocateAnythingModel()

    monkeypatch.setattr(locate_anything_module.AutoModel, "from_pretrained", _fake_model_from_pretrained)
    monkeypatch.setattr(locate_anything_module.utils, "maybe_autocast", lambda device, precision: nullcontext())

    LocateAnythingModel().load_runtime(dataset, ctx)

    assert captured["kwargs"]["dtype"] is torch.float16
    assert "torch_dtype" not in captured["kwargs"]
    assert captured["kwargs"]["device_map"] == {"": "cuda"}
    assert captured["kwargs"]["trust_remote_code"] is True
    assert captured["kwargs"]["config"] is hf_config
    assert captured["processor_kwargs"]["use_fast"] is False
    assert hf_config._attn_implementation == "sdpa"
    assert hf_config.text_config._attn_implementation == "sdpa"
    assert hf_config.vision_config._attn_implementation == "sdpa"


def test_locate_anything_load_runtime_forwards_nf4_quant_config(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(provider="locate_anything", task="ground", device="cuda", precision="nf4"),
    )
    ctx = _locate_anything_ctx(tmp_path, dataset.fm_request)
    captured: dict[str, object] = {}
    _patch_locate_anything_load_deps(monkeypatch, tmp_path, captured)
    expected_qcfg = _patch_locate_anything_nf4_precision(monkeypatch)

    monkeypatch.setattr(locate_anything_module.utils, "select_device", lambda pref, force_cpu=False: torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _fake_model_from_pretrained(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeLocateAnythingModel()

    monkeypatch.setattr(locate_anything_module.AutoModel, "from_pretrained", _fake_model_from_pretrained)
    monkeypatch.setattr(locate_anything_module.utils, "maybe_autocast", lambda device, precision: nullcontext())

    LocateAnythingModel().load_runtime(dataset, ctx)

    qcfg = captured["kwargs"]["quantization_config"]
    assert qcfg is expected_qcfg
    assert qcfg.load_in_4bit is True
    assert qcfg.bnb_4bit_quant_type == "nf4"
    assert qcfg.bnb_4bit_use_double_quant is True
    assert captured["kwargs"]["device_map"] == {"": "cuda"}


def test_locate_anything_load_runtime_rejects_auto_nf4(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(provider="locate_anything", task="ground", device="auto", precision="nf4"),
    )
    ctx = _locate_anything_ctx(tmp_path, dataset.fm_request)
    _patch_locate_anything_load_deps(monkeypatch, tmp_path, {})
    _patch_locate_anything_nf4_precision(monkeypatch)

    monkeypatch.setattr(locate_anything_module.utils, "select_device", lambda pref, force_cpu=False: torch.device("cuda"))

    with pytest.raises(RuntimeError, match="`--device auto --precision nf4` is not supported yet"):
        LocateAnythingModel().load_runtime(dataset, ctx)


def test_locate_anything_load_runtime_with_auto_infers_offload_kwargs(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(provider="locate_anything", task="ground", device="auto", precision="fp16"),
    )
    ctx = _locate_anything_ctx(tmp_path, dataset.fm_request)
    captured: dict[str, object] = {}
    gib = 1024 ** 3
    _patch_locate_anything_load_deps(monkeypatch, tmp_path, captured)

    monkeypatch.setattr(locate_anything_module.utils, "select_device", lambda pref, force_cpu=False: torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (8 * gib, 8 * gib))
    monkeypatch.setattr(locate_anything_module.utils, "available_cpu_memory_bytes", lambda: 10 * gib)

    def _fake_model_from_pretrained(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeLocateAnythingModel()

    monkeypatch.setattr(locate_anything_module.AutoModel, "from_pretrained", _fake_model_from_pretrained)
    monkeypatch.setattr(locate_anything_module.utils, "maybe_autocast", lambda device, precision: nullcontext())

    runtime = LocateAnythingModel().load_runtime(dataset, ctx)

    assert captured["kwargs"]["device_map"] == "auto"
    assert captured["kwargs"]["max_memory"] == {0: 7 * gib, "cpu": 8 * gib}
    assert captured["kwargs"]["offload_folder"] == str(tmp_path / "stage" / "offload")
    assert captured["kwargs"]["dtype"] is torch.float16
    assert runtime.precision.effective == "fp16"


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)

    def __len__(self):
        return len(self.xyxy)


def test_yolo_e_resolve_weights_uses_durable_cache_without_hf(monkeypatch, tmp_path: Path) -> None:
    ctx = SimpleNamespace(weights_dir=tmp_path / "weights", caller_cwd=tmp_path)

    def _fail_hf(*args, **kwargs):
        raise AssertionError("HF should not be used")

    monkeypatch.setattr(yolo_e_module, "_legacy_weight_roots", lambda _ctx: [])
    monkeypatch.setattr(yolo_e_module.utils, "materialize_hf_file", _fail_hf)

    path = Path(yolo_e_module._resolve_weights(YoloEConfig(), ctx))

    assert path == tmp_path / "weights" / "yoloe-26l-seg.pt"
    assert path.parent.is_dir()


def test_yolo_e_resolve_weights_adopts_legacy_ultralytics_download(monkeypatch, tmp_path: Path) -> None:
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_file = legacy_dir / "yoloe-26l-seg.pt"
    legacy_file.write_bytes(b"checkpoint")
    ctx = SimpleNamespace(weights_dir=tmp_path / "weights", caller_cwd=tmp_path)

    monkeypatch.setattr(yolo_e_module, "_legacy_weight_roots", lambda _ctx: [legacy_dir])

    path = Path(yolo_e_module._resolve_weights(YoloEConfig(), ctx))

    assert path == tmp_path / "weights" / "yoloe-26l-seg.pt"
    assert path.read_bytes() == b"checkpoint"


def test_yolo_e_load_runtime_runs_ultralytics_from_weights_dir(monkeypatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=tmp_path / "target.jpg"),
                attributes={"ground_prompts": ["widget"]},
            )
        ],
        fm_request=FMRequest(provider="yolo_e", task="ground", device="cpu"),
    )
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=YoloEOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        caller_cwd=tmp_path,
    )
    calls = []

    class _YOLOE:
        def __init__(self, path):
            calls.append(("init", Path.cwd(), path))

        def get_text_pe(self, names):
            calls.append(("text", Path.cwd(), tuple(names)))
            return object()

        def set_classes(self, names, text_pe):
            calls.append(("classes", Path.cwd(), tuple(names), text_pe is not None))

    monkeypatch.setattr(yolo_e_module, "_legacy_weight_roots", lambda _ctx: [])
    monkeypatch.setattr(yolo_e_module.utils, "select_device", lambda pref: torch.device("cpu"))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLOE=_YOLOE))

    runtime = YoloEModel().load_runtime(dataset, ctx)

    assert runtime.mode == "text"
    assert calls[0] == ("init", ctx.weights_dir, str(ctx.weights_dir / "yoloe-26l-seg.pt"))
    assert calls[1][:3] == ("text", ctx.weights_dir, ("widget",))
    assert calls[2][0:3] == ("classes", ctx.weights_dir, ("widget",))


class _FakeYoloEResult:
    def __init__(self, *, xyxy, conf, cls, orig_shape, mask_xy=None):
        self.boxes = _FakeBoxes(xyxy, conf, cls)
        self.orig_shape = orig_shape
        self.masks = SimpleNamespace(xy=mask_xy) if mask_xy is not None else None


class _FakeYoloEModel:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self._result]


def test_yolo_e_text_mode_uses_prompt_class_as_label(tmp_path: Path) -> None:
    image_path = tmp_path / "yoloe.jpg"
    _write_image(image_path)
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=16, height=16),
                attributes={"ground_prompts": ["widget"]},
            )
        ],
        fm_request=FMRequest(provider="yolo_e", task="ground"),
    )

    result = _FakeYoloEResult(
        xyxy=[[1.0, 1.0, 8.0, 8.0]],
        conf=[0.9],
        cls=[0],
        orig_shape=(16, 16),
        mask_xy=[np.array([[1.0, 1.0], [8.0, 1.0], [8.0, 8.0], [1.0, 8.0]], dtype=np.float32)],
    )
    model = _FakeYoloEModel(result)
    runtime = YoloERuntime(
        cfg=YoloEConfig(),
        model=model,
        device_arg="cpu",
        mode="text",
        class_names=["widget"],
        confidence=0.25,
        refer_image=None,
        visual_prompts=None,
        vp_predictor=None,
    )

    out = YoloEModel().process_batch(dataset, [YoloEJob(record_idx=0, image_path=image_path)], runtime, None)

    rec = dataset.records[0]
    assert out.modified_record_indices == [0]
    assert rec.boxes[0].label == "widget"
    assert rec.boxes[0].prompt == "widget"
    assert rec.polys[0].label == "widget"
    assert dataset.classes == ["widget"]
    assert dataset.task == Task.seg
    assert "yolo-e" in rec.attributes["fm_tasks"]
    assert "refer_image" not in model.calls[0]


def test_yolo_e_reference_mode_labels_from_reference_classes(tmp_path: Path) -> None:
    image_path = tmp_path / "target.jpg"
    _write_image(image_path)
    refer_path = tmp_path / "reference.jpg"
    _write_image(refer_path)
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=image_path, width=16, height=16))],
        fm_request=FMRequest(provider="yolo_e", task="ground"),
    )

    result = _FakeYoloEResult(
        xyxy=[[2.0, 2.0, 10.0, 10.0]],
        conf=[0.75],
        cls=[0],
        orig_shape=(16, 16),
    )
    model = _FakeYoloEModel(result)
    sentinel_predictor = object()
    runtime = YoloERuntime(
        cfg=YoloEConfig(),
        model=model,
        device_arg="cpu",
        mode="reference",
        class_names=["bolt"],
        confidence=0.25,
        refer_image=refer_path,
        visual_prompts={"bboxes": np.array([[1.0, 1.0, 5.0, 5.0]], dtype=np.float32), "cls": np.array([0])},
        vp_predictor=sentinel_predictor,
    )

    out = YoloEModel().process_batch(dataset, [YoloEJob(record_idx=0, image_path=image_path)], runtime, None)

    rec = dataset.records[0]
    assert out.modified_record_indices == [0]
    assert rec.boxes[0].label == "bolt"
    assert dataset.classes == ["bolt"]
    assert model.calls[0]["refer_image"] == str(refer_path)
    assert model.calls[0]["predictor"] is sentinel_predictor
