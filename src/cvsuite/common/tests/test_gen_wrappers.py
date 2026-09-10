from __future__ import annotations

from importlib import import_module
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cvsuite.common.core import FMRequest
from cvsuite.common.core import VisionDataset

try:
    from cvsuite.common.fm.providers.bases import BaseCreateGenerationModel, BaseEditGenerationModel
    from cvsuite.common.fm.providers.bases.diffusion import (
        BaseDiffusersCreateBackend,
        BaseDiffusersEditBackend,
        BaseDiffusersGenerationBackend,
    )
    from cvsuite.common.fm.providers.edit.dim_edit import patch_dim_modeling
    from cvsuite.common.fm.providers.bases.qwen_image import QwenImageDiffusersMixin
    from cvsuite.common.fm.providers.registry import MODEL_MODULES
except ModuleNotFoundError as exc:
    if exc.name and not exc.name.startswith("cvsuite"):
        pytest.skip(f"{exc.name} is required for gen wrapper unit tests", allow_module_level=True)
    raise

CREATE_MODELS = {
    "stable_diffusion": {
        "module": "cvsuite.common.fm.providers.create.stable_diffusion",
        "backend": "stable-diffusion",
        "hf_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "managed_auto": True,
        "nf4": True,
    },
    "flux": {
        "module": "cvsuite.common.fm.providers.create.flux",
        "backend": "flux",
        "hf_model": "black-forest-labs/FLUX.1-dev",
        "managed_auto": True,
        "nf4": True,
    },
    "qwen_image": {
        "module": "cvsuite.common.fm.providers.create.qwen_image",
        "backend": "qwen-image",
        "hf_model": "Qwen/Qwen-Image",
        "managed_auto": True,
        "nf4": True,
    },
    "sana": {
        "module": "cvsuite.common.fm.providers.create.sana",
        "backend": "sana",
        "hf_model": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        "managed_auto": True,
        "nf4": False,
    },
}

EDIT_MODELS = {
    "qwen_image_edit": {
        "module": "cvsuite.common.fm.providers.edit.qwen_image_edit",
        "backend": "qwen-image-edit",
        "hf_model": "Qwen/Qwen-Image-Edit",
        "managed_auto": True,
        "nf4": True,
    },
    "flux2_klein": {
        "module": "cvsuite.common.fm.providers.edit.flux2_klein",
        "backend": "flux2-klein",
        "hf_model": "black-forest-labs/FLUX.2-klein-4B",
        "managed_auto": True,
        "nf4": True,
    },
    "step1x_edit": {
        "module": "cvsuite.common.fm.providers.edit.step1x_edit",
        "backend": "step1x-edit",
        "hf_model": "stepfun-ai/Step1X-Edit-v1p1-diffusers",
        "managed_auto": False,
        "nf4": False,
    },
    "dim_edit": {
        "module": "cvsuite.common.fm.providers.edit.dim_edit",
        "backend": "dim-edit",
        "hf_model": "stdKonjac/DIM-4.6B-Edit",
        "managed_auto": False,
        "nf4": True,
    },
    "ovis_u1_3b": {
        "module": "cvsuite.common.fm.providers.edit.ovis_u1_3b",
        "backend": "ovis-u1",
        "hf_model": "AIDC-AI/Ovis-U1-3B",
        "managed_auto": False,
        "nf4": False,
    },
}


def _ctx(tmp_path: Path, model, *, request: FMRequest | None = None):
    return SimpleNamespace(
        request=request or FMRequest(provider=model.model_name, task="gen", device="cpu", precision="fp32"),
        options=model.parse_options({}),
        config_path=None,
        work_dir=tmp_path / "run",
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        stage_dir=tmp_path / "stage",
        caller_cwd=tmp_path,
        model_cache=tmp_path / "model-cache",
        model_name=model.model_name,
        batch_size=1,
        prompt="[]",
    )


@pytest.mark.parametrize("model_id, spec", list(CREATE_MODELS.items()) + list(EDIT_MODELS.items()))
def test_generation_wrappers_are_registered_under_base_aligned_packages(model_id: str, spec: dict[str, object]) -> None:
    assert MODEL_MODULES[model_id] == spec["module"]
    assert ".models.gen." not in MODEL_MODULES[model_id]
    if model_id in CREATE_MODELS:
        assert ".models.create." in MODEL_MODULES[model_id]
    else:
        assert ".models.edit." in MODEL_MODULES[model_id]


@pytest.mark.parametrize("model_id, spec", CREATE_MODELS.items())
def test_create_wrappers_expose_expected_defaults(model_id: str, spec: dict[str, object]) -> None:
    module = import_module(str(spec["module"]))
    model = module.MODEL

    assert isinstance(model, BaseCreateGenerationModel)
    assert model.model_name == model_id
    assert model.supports_managed_auto_device is spec["managed_auto"]
    assert model.supports_nf4_precision is spec["nf4"]

    options = model.options_cls()
    assert options.backend == spec["backend"]
    assert options.model_id == spec["hf_model"]


@pytest.mark.parametrize("model_id, spec", EDIT_MODELS.items())
def test_edit_wrappers_expose_expected_defaults(model_id: str, spec: dict[str, object]) -> None:
    module = import_module(str(spec["module"]))
    model = module.MODEL

    assert isinstance(model, BaseEditGenerationModel)
    assert model.model_name == model_id
    assert model.supports_managed_auto_device is spec["managed_auto"]
    assert model.supports_nf4_precision is spec["nf4"]

    options = model.options_cls()
    assert options.backend == spec["backend"]
    assert options.model_id == spec["hf_model"]


@pytest.mark.parametrize("model_id, spec", list(CREATE_MODELS.items()) + list(EDIT_MODELS.items()))
def test_generation_wrappers_load_runtime_with_default_backend_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_id: str,
    spec: dict[str, object],
) -> None:
    module = import_module(str(spec["module"]))
    model = module.MODEL
    dataset = VisionDataset(records=[], meta={"gen_mode": "create" if model_id in CREATE_MODELS else "edit"})
    ctx = _ctx(tmp_path, model)
    captured: dict[str, object] = {}

    class _FakeBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate_batch(self, **_kwargs):
            return []

    monkeypatch.setattr("cvsuite.common.fm.providers.bases.gen._load_object", lambda _spec: _FakeBackend)

    runtime = model.load_runtime(dataset, ctx)

    assert runtime.backend_spec == spec["backend"]
    assert runtime.config["backend"] == spec["backend"]
    assert runtime.config["model_id"] == spec["hf_model"]
    assert runtime.outputs_dir == ctx.work_dir / "images"
    assert captured["mode"] == dataset.meta["gen_mode"]
    assert captured["model"] is model


def test_flux_wrapper_emits_precision_fallback_warning_on_cpu(tmp_path: Path) -> None:
    model = import_module(str(CREATE_MODELS["flux"]["module"])).MODEL
    dataset = VisionDataset(records=[])
    ctx = _ctx(
        tmp_path,
        model,
        request=FMRequest(provider="flux", task="gen", device="cpu", precision="fp16"),
    )

    precision = model.resolve_runtime_precision(dataset, ctx, torch.device("cpu"))

    assert precision.effective == "fp32"
    assert dataset.meta["fm"]["warnings"]
    assert "requires CUDA" in dataset.meta["fm"]["warnings"][0]


def test_flux_wrapper_rejects_auto_nf4_requests() -> None:
    model = import_module(str(CREATE_MODELS["flux"]["module"])).MODEL

    with pytest.raises(RuntimeError, match="--device auto --precision nf4"):
        model.validate_auto_device_request(FMRequest(provider="flux", task="gen", device="auto", precision="nf4"))


def test_step1x_wrapper_rejects_managed_auto_device() -> None:
    model = import_module(str(EDIT_MODELS["step1x_edit"]["module"])).MODEL

    with pytest.raises(RuntimeError, match="`--device auto` is not implemented"):
        model.validate_auto_device_request(FMRequest(provider="step1x_edit", task="gen", device="auto", precision="fp32"))


def test_dim_edit_wrapper_accepts_auto_device_when_cuda_available(monkeypatch) -> None:
    model = import_module(str(EDIT_MODELS["dim_edit"]["module"])).MODEL

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model.validate_auto_device_request(FMRequest(provider="dim_edit", task="gen", device="auto", precision="nf4"))


def test_dim_edit_wrapper_rejects_auto_device_without_cuda(monkeypatch) -> None:
    model = import_module(str(EDIT_MODELS["dim_edit"]["module"])).MODEL

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was explicitly requested"):
        model.validate_auto_device_request(FMRequest(provider="dim_edit", task="gen", device="auto", precision="fp32"))


def test_dim_edit_wrapper_rejects_auto_max_gpu_memory(monkeypatch) -> None:
    model = import_module(str(EDIT_MODELS["dim_edit"]["module"])).MODEL

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="--max-gpu-memory"):
        model.validate_auto_device_request(
            FMRequest(provider="dim_edit", task="gen", device="auto", precision="fp32", max_gpu_memory="12GiB")
        )


def test_diffusers_generation_backends_skip_unset_optional_call_kwargs() -> None:
    job = SimpleNamespace(prompt="industrial machine", prompt_index=0)
    create_backend = BaseDiffusersCreateBackend.__new__(BaseDiffusersCreateBackend)
    create_backend.config = {
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 30,
        "guidance_scale": 7.0,
        "max_sequence_length": None,
        "true_cfg_scale": None,
    }

    create_kwargs = create_backend.build_call_kwargs(job)

    assert "max_sequence_length" not in create_kwargs
    assert "true_cfg_scale" not in create_kwargs

    create_backend.config["max_sequence_length"] = "512"
    create_backend.config["true_cfg_scale"] = "4.0"

    create_kwargs = create_backend.build_call_kwargs(job)

    assert create_kwargs["max_sequence_length"] == 512
    assert create_kwargs["true_cfg_scale"] == pytest.approx(4.0)

    edit_backend = BaseDiffusersEditBackend.__new__(BaseDiffusersEditBackend)
    edit_backend.config = {
        "width": None,
        "height": None,
        "num_inference_steps": 30,
        "guidance_scale": 7.0,
        "strength": None,
        "max_sequence_length": None,
        "true_cfg_scale": None,
    }

    edit_kwargs = edit_backend.build_call_kwargs(job, SimpleNamespace(width=640, height=480))

    assert "max_sequence_length" not in edit_kwargs
    assert "true_cfg_scale" not in edit_kwargs


def test_diffusers_backend_does_not_pass_explicit_device_map_dicts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    backend = BaseDiffusersGenerationBackend.__new__(BaseDiffusersGenerationBackend)
    backend.device = torch.device("cuda")
    backend.ctx = SimpleNamespace(
        request=FMRequest(provider="qwen_image", task="gen", device="gpu", precision="fp32"),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        stage_dir=tmp_path / "stage",
        model_cache=tmp_path / "model-cache",
    )

    kwargs, placement = backend.placement_load_kwargs()

    assert placement.device_map == {"": "cuda"}
    assert "device_map" not in kwargs
    assert "max_memory" not in kwargs
    assert "offload_folder" not in kwargs


def test_diffusers_backend_preserves_managed_auto_device_map(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    backend = BaseDiffusersGenerationBackend.__new__(BaseDiffusersGenerationBackend)
    backend.device = torch.device("cuda")
    backend.ctx = SimpleNamespace(
        request=FMRequest(
            model="qwen_image",
            task="gen",
            device="auto",
            precision="fp32",
            max_gpu_memory="20GiB",
        ),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        stage_dir=tmp_path / "stage",
        model_cache=tmp_path / "model-cache",
    )

    kwargs, placement = backend.placement_load_kwargs()

    assert placement.device_map == "auto"
    assert kwargs["device_map"] == "balanced"
    assert kwargs["max_memory"][0] == "20GiB"
    assert "offload_folder" in kwargs


def test_diffusers_backend_can_ignore_legacy_config_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeModel:
        @classmethod
        def extract_init_dict(cls, config_dict, **_kwargs):
            return config_dict, {}, {}

    class _FakePipeline:
        @property
        def _execution_device(self):
            return torch.device("cpu")

        def __init__(self):
            self.transformer = torch.nn.Linear(1, 1)
            self.transformer.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(FakeModel=_FakeModel, FakePipeline=_FakePipeline))
    backend = BaseDiffusersGenerationBackend.__new__(BaseDiffusersGenerationBackend)
    backend.diffusers_config_ignored_attrs = {"FakeModel": ("legacy_attr", "existing")}
    backend.diffusers_execution_device_components = {"FakePipeline": "transformer"}

    backend.patch_diffusers_config_compat()

    init_dict, _, _ = _FakeModel.extract_init_dict({"keep": 1, "legacy_attr": 2, "existing": 3})

    assert init_dict == {"keep": 1}
    assert _FakePipeline()._execution_device.type == ("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "model_id, spec",
    [
        ("qwen_image", CREATE_MODELS["qwen_image"]),
        ("qwen_image_edit", EDIT_MODELS["qwen_image_edit"]),
    ],
)
def test_qwen_image_wrappers_ignore_legacy_transformer_config_attr(model_id: str, spec: dict[str, object]) -> None:
    module = import_module(str(spec["module"]))
    backend_spec = module.MODEL.backend_aliases[str(spec["backend"])]
    backend_module, backend_attr = backend_spec.split(":", 1)
    backend_cls = getattr(import_module(backend_module), backend_attr)

    assert backend_cls.diffusers_config_ignored_attrs == {"QwenImageTransformer2DModel": ("pooled_projection_dim",)}
    assert backend_cls.managed_auto_diffusers_components == {"transformer": "QwenImageTransformer2DModel"}
    assert backend_cls.managed_auto_transformers_components == {"text_encoder": "Qwen2_5_VLForConditionalGeneration"}


def test_diffusers_backend_loads_managed_auto_components_separately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class _FakeComponent:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((cls.__name__, args, kwargs))
            return cls()

    class _FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((cls.__name__, args, kwargs))
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(FakePipeline=_FakePipeline, FakeTransformer=_FakeComponent),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(FakeTextEncoder=_FakeComponent))

    backend = BaseDiffusersGenerationBackend.__new__(BaseDiffusersGenerationBackend)
    backend.pipeline_class_name = "FakePipeline"
    backend.diffusers_config_ignored_attrs = {}
    backend.diffusers_execution_device_components = {}
    backend.managed_auto_diffusers_components = {"transformer": "FakeTransformer"}
    backend.managed_auto_transformers_components = {"text_encoder": "FakeTextEncoder"}
    backend.precision = SimpleNamespace(compute_dtype=torch.float32)
    backend.source = SimpleNamespace(
        load_arg="repo",
        cache_dir=None,
        local_files_only=False,
        token=None,
        revision=None,
    )
    backend.model_wrapper = SimpleNamespace(emit_hf_auto_device_report=lambda *_args, **_kwargs: None)
    placement = SimpleNamespace(max_memory={0: 1024**3, "cpu": 2 * 1024**3}, offload_folder=str(tmp_path / "offload"))

    pipeline, returned_placement = backend.load_managed_auto_component_pipeline(placement)

    assert isinstance(pipeline, _FakePipeline)
    assert returned_placement is placement
    assert calls[0][0] == "_FakeComponent"
    assert calls[0][2]["subfolder"] == "transformer"
    assert calls[0][2]["device_map"] == "auto"
    assert str(calls[0][2]["offload_folder"]).endswith("/transformer")
    assert calls[1][2]["subfolder"] == "text_encoder"
    assert calls[1][2]["device_map"] == "auto"
    assert str(calls[1][2]["offload_folder"]).endswith("/text_encoder")
    assert calls[2][0] == "_FakePipeline"
    assert "transformer" in calls[2][2]
    assert "text_encoder" in calls[2][2]
    assert "device_map" not in calls[2][2]


def test_diffusers_backend_patches_concrete_managed_auto_pipeline_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Backend(QwenImageDiffusersMixin, BaseDiffusersGenerationBackend):
        pass

    class _FakeComponent:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class _ConcretePipeline:
        text_encoder = torch.nn.Embedding(4, 2)
        _execution_device = torch.device("meta")
        seen_device: torch.device | None = None

        def _get_qwen_prompt_embeds(self, prompt=None, device=None, dtype=None):
            self.__class__.seen_device = torch.device(device)
            return torch.ones(1, 1, 2, device=device), torch.ones(1, 1, dtype=torch.long, device=device)

    class _FactoryPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return _ConcretePipeline()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(FakePipeline=_FactoryPipeline, FakeTransformer=_FakeComponent),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(FakeTextEncoder=_FakeComponent))

    backend = _Backend.__new__(_Backend)
    backend.pipeline_class_name = "FakePipeline"
    backend.diffusers_config_ignored_attrs = {}
    backend.diffusers_execution_device_components = {}
    backend.managed_auto_diffusers_components = {"transformer": "FakeTransformer"}
    backend.managed_auto_transformers_components = {"text_encoder": "FakeTextEncoder"}
    backend.precision = SimpleNamespace(compute_dtype=torch.float32)
    backend.source = SimpleNamespace(
        load_arg="repo",
        cache_dir=None,
        local_files_only=False,
        token=None,
        revision=None,
    )
    backend.model_wrapper = SimpleNamespace(emit_hf_auto_device_report=lambda *_args, **_kwargs: None)
    placement = SimpleNamespace(max_memory={0: 1024**3, "cpu": 2 * 1024**3}, offload_folder=str(tmp_path / "offload"))

    pipeline, _placement = backend.load_managed_auto_component_pipeline(placement)
    prompt_embeds, prompt_mask = pipeline._get_qwen_prompt_embeds("hello", device=torch.device("meta"))

    assert _ConcretePipeline.seen_device == torch.device("cpu")
    assert prompt_embeds.device.type == "meta"
    assert prompt_mask.device.type == "meta"


def test_diffusers_component_auto_keeps_inferred_runtime_cuda_reserve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = BaseDiffusersGenerationBackend.__new__(BaseDiffusersGenerationBackend)
    backend.managed_auto_cuda_runtime_reserve_fraction = 0.25
    backend.managed_auto_cuda_runtime_reserve_bytes = 4 * 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _index: (15 * 1024**3, 16 * 1024**3))
    placement = SimpleNamespace(
        max_memory={0: 14 * 1024**3, "cpu": 64 * 1024**3},
        memory_budget_source="inferred",
        offload_folder=str(tmp_path / "offload"),
    )

    kwargs = backend._managed_component_load_kwargs(placement, "transformer")

    assert kwargs["max_memory"][0] == 11 * 1024**3


def test_diffusers_component_auto_preserves_explicit_cuda_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = BaseDiffusersGenerationBackend.__new__(BaseDiffusersGenerationBackend)
    backend.managed_auto_cuda_runtime_reserve_fraction = 0.25
    backend.managed_auto_cuda_runtime_reserve_bytes = 4 * 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _index: (15 * 1024**3, 16 * 1024**3))
    placement = SimpleNamespace(
        max_memory={0: 14 * 1024**3, "cpu": 64 * 1024**3},
        memory_budget_source="explicit",
        offload_folder=str(tmp_path / "offload"),
    )

    kwargs = backend._managed_component_load_kwargs(placement, "transformer")

    assert kwargs["max_memory"][0] == 14 * 1024**3


def test_qwen_prompt_encoder_uses_text_embedding_device_then_returns_requested_device() -> None:
    seen_devices: list[torch.device] = []

    class _FakeTextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 2)

        def get_input_embeddings(self):
            return self.embedding

    class _FakeQwenPipeline:
        text_encoder = _FakeTextEncoder()
        _execution_device = torch.device("meta")

        def _get_qwen_prompt_embeds(self, prompt=None, device=None, dtype=None):
            seen_devices.append(torch.device(device))
            return torch.ones(1, 1, 2, device=device), torch.ones(1, 1, dtype=torch.long, device=device)

    QwenImageDiffusersMixin.patch_prompt_encoder_input_device(_FakeQwenPipeline)

    prompt_embeds, prompt_mask = _FakeQwenPipeline()._get_qwen_prompt_embeds("hello", device=torch.device("meta"))

    assert seen_devices == [torch.device("cpu")]
    assert prompt_embeds.device.type == "meta"
    assert prompt_mask.device.type == "meta"


def test_qwen_image_disables_true_cfg_scale_without_negative_prompt() -> None:
    module = import_module(str(CREATE_MODELS["qwen_image"]["module"]))
    backend = module.QwenImageCreateBackend.__new__(module.QwenImageCreateBackend)
    backend.config = {
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 28,
        "guidance_scale": None,
        "max_sequence_length": "512",
        "true_cfg_scale": "4.0",
        "negative_prompt": None,
    }

    kwargs = backend.build_call_kwargs(SimpleNamespace(prompt="industrial machine", prompt_index=0))

    assert kwargs["max_sequence_length"] == 512
    assert kwargs["true_cfg_scale"] == pytest.approx(1.0)

    backend.config["negative_prompt"] = "low quality"
    kwargs = backend.build_call_kwargs(SimpleNamespace(prompt="industrial machine", prompt_index=0))

    assert kwargs["negative_prompt"] == "low quality"
    assert kwargs["true_cfg_scale"] == pytest.approx(4.0)


def test_qwen_image_managed_auto_encodes_vae_image_on_vae_device() -> None:
    module = import_module(str(CREATE_MODELS["qwen_image"]["module"]))
    backend = module.QwenImageCreateBackend.__new__(module.QwenImageCreateBackend)
    seen_devices: list[torch.device] = []
    execution_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    class _FakePipeline:
        _execution_device = execution_device
        latent_channels = 1

        class _Vae(torch.nn.Module):
            config = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def encode(self, image):
                seen_devices.append(image.device)
                return SimpleNamespace(latents=torch.ones(1, 1, 1, 1, 1, device=image.device))

        vae = _Vae()

        def _encode_vae_image(self, image, generator=None):
            return self.vae.encode(image).latents

    pipeline = _FakePipeline()

    assert backend.prepare_managed_auto_pipeline(pipeline) is pipeline
    image_latents = pipeline._encode_vae_image(torch.ones(1, 1, 1, 1, 1, device=execution_device))

    assert seen_devices == [torch.device("cpu")]
    assert image_latents.device == execution_device


def test_qwen_image_managed_auto_patches_vae_decode_input_device() -> None:
    module = import_module(str(CREATE_MODELS["qwen_image"]["module"]))
    backend = module.QwenImageCreateBackend.__new__(module.QwenImageCreateBackend)
    seen_devices: list[torch.device] = []

    class _FakeVae(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def decode(self, latents, return_dict=False):
            seen_devices.append(latents.device)
            return (latents,)

    class _FakePipeline:
        vae = _FakeVae()

    pipeline = _FakePipeline()

    assert backend.prepare_managed_auto_pipeline(pipeline) is pipeline
    result = pipeline.vae.decode(torch.ones(1), return_dict=False)

    assert seen_devices == [torch.device("cpu")]
    assert torch.equal(result[0], torch.ones(1))


def test_qwen_image_create_uses_qwen_guidance_defaults() -> None:
    module = import_module(str(CREATE_MODELS["qwen_image"]["module"]))
    options = module.QwenImageOptions()

    assert options.guidance_scale is None
    assert options.true_cfg_scale is None


def test_dim_edit_wrapper_exposes_effective_sampling_defaults() -> None:
    module = import_module(str(EDIT_MODELS["dim_edit"]["module"]))
    options = module.DIMEditOptions()

    assert options.seed == 233
    assert options.max_condition_length == 8192
    assert options.gen_resolution == 1024
    assert options.task_type == "MM-PAD"
    assert options.num_inference_steps == 30
    assert options.guidance_scale == pytest.approx(7.5)


def test_dim_flash_attention_patch_adds_sdpa_fallback(tmp_path: Path) -> None:
    dim_root = tmp_path / "DIM"
    models_dir = dim_root / "models"
    models_dir.mkdir(parents=True)
    modeling_path = models_dir / "modeling_dim.py"
    sana_pipeline_path = models_dir / "sana_pipeline.py"
    modeling_path.write_text(
        "import math\n"
        "import os\n"
        "import re\n"
        "\n"
        "class DIM:\n"
        "    def __init__(self, model_args):\n"
        "        if model_args.condition_type == 'LMToken':\n"
        "            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(\n"
        "                model_args.pretrained_model_name_or_path,\n"
        "                torch_dtype=\"auto\",\n"
        "                attn_implementation='flash_attention_2',\n"
        "            )\n"
        "            self.designer = Qwen2_5_VLForConditionalGeneration.from_pretrained(\n"
        "                \"Qwen/Qwen2.5-VL-7B-Instruct\",\n"
        "                torch_dtype=torch.bfloat16,\n"
        "            )\n"
        "    def from_pretrained(self, pretrained_model_name_or_path):\n"
        "        state_dict = load_file(os.path.join(pretrained_model_name_or_path, \"model.safetensors\"))\n"
        "        # remove text encoder (if any)\n"
        "        state_dict = {k: v for k, v in state_dict.items() if not k.startswith(\"decoder.text_encoder.\")}\n"
        "\n"
        "        # load weights, need to check carefully\n"
        "        info = self.load_state_dict(state_dict, strict=False)\n"
        "        print(f'Load pretrained weights from {pretrained_model_name_or_path}\\n\\n{info}\\n\\n')\n"
        "\n"
        "        assert not info.unexpected_keys, 'Only missing keys are allowed, got unexpected keys'\n"
        "\n"
        "    def generate(self, y, guidance_scale, num_inference_steps):\n"
        "        generator = torch.Generator(device=y.device).manual_seed(233)\n"
        "        scheduler = DPMS(\n"
        "            self.model,\n"
        "            condition=y,\n"
        "            uncondition=null_y,\n"
        "            guidance_type=guidance_type,\n"
        "            # cfg_scale=guidance_scale,\n"
        "            cfg_scale=7.5,\n"
        "            pag_scale=pag_guidance_scale,\n"
        "            pag_applied_layers=self.config.model.pag_applied_layers,\n"
        "            model_type=\"flow\",\n"
        "            model_kwargs=model_kwargs,\n"
        "            schedule=\"FLOW\",\n"
        "            with_latents_condition=self.model_args.with_latents_condition,\n"
        "            latents_condition=latents_condition\n"
        "        )\n"
        "        _latents_denoised = scheduler.sample(\n"
        "            z,\n"
        "            # steps=num_inference_steps,\n"
        "            steps=30,\n"
        "            order=2,\n"
        "            skip_type=\"time_uniform_flow\",\n"
        "            method=\"multistep\",\n"
        "            flow_shift=self.flow_shift,\n"
        "        )\n"
        "        return generator, _latents_denoised\n",
        encoding="utf-8",
    )
    sana_pipeline_path.write_text(
        "# SPDX-License-Identifier: Apache-2.0\n"
        "import warnings\n"
        "\n"
        "class SanaPipeline:\n"
        "    def __init__(self, config):\n"
        "        self.device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n"
        "        # 1. build vae and text encoder\n"
        "        self.vae = self.build_vae(config.vae)\n"
        "        self.tokenizer, self.text_encoder = self.build_text_encoder(config.text_encoder)\n"
        "\n"
        "        # 2. build Sana model\n"
        "        self.model = self.build_sana_model(config).to(self.device)\n"
        "\n"
        "        # 3. pre-compute null embedding\n"
        "        with torch.no_grad():\n"
        "            null_caption_token = self.tokenizer(\n"
        "                \"\", max_length=self.max_sequence_length, padding=\"max_length\", truncation=True, return_tensors=\"pt\"\n"
        "            ).to(self.device)\n"
        "            self.null_caption_embs = self.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[\n"
        "                0\n"
        "            ]\n",
        encoding="utf-8",
    )

    patch_dim_modeling(dim_root)

    modeling_text = modeling_path.read_text(encoding="utf-8")
    assert "import importlib.util" in modeling_text
    assert "import sys" in modeling_text
    assert "mllm_load_kwargs" in modeling_text
    assert "VT_DIM_MLLM_AUTO_DEVICE_REPORT" in modeling_text
    assert "VT_DIM_MLLM_NF4" in modeling_text
    assert "BitsAndBytesConfig" in modeling_text
    assert 'bnb_4bit_quant_type="nf4"' in modeling_text
    assert 'mllm_load_kwargs.setdefault("device_map"' in modeling_text
    assert '"torch_dtype": "auto"' in modeling_text
    assert "torch_dtype=torch.bfloat16" in modeling_text
    assert '"attn_implementation": "flash_attention_2"' in modeling_text
    assert 'else "sdpa"' in modeling_text
    assert "attn_implementation='flash_attention_2'" not in modeling_text
    assert "VT_DIM_SEED" in modeling_text
    assert "VT_DIM_GUIDANCE_SCALE" in modeling_text
    assert "VT_DIM_NUM_INFERENCE_STEPS" in modeling_text
    assert "manual_seed(233)" not in modeling_text
    assert "cfg_scale=7.5" not in modeling_text
    assert "steps=30" not in modeling_text

    sana_text = sana_pipeline_path.read_text(encoding="utf-8")
    assert "import os" in sana_text
    assert "VT_DIM_SANA_INIT_DEVICE" in sana_text
    assert "VT_DIM_SANA_DECODER_ONLY" in sana_text
    assert "VT_DIM_SANA_LAZY_TO_DEVICE" in sana_text

    # Second call should be idempotent (no further changes)
    text_before = modeling_path.read_text(encoding="utf-8")
    sana_before = sana_pipeline_path.read_text(encoding="utf-8")
    patch_dim_modeling(dim_root)
    assert modeling_path.read_text(encoding="utf-8") == text_before
    assert sana_pipeline_path.read_text(encoding="utf-8") == sana_before
