from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

try:
    from cvsuite.common.fm.providers.vlm import minicpm_v as minicpm_module
    from cvsuite.common.fm.providers.vlm.minicpm_v import (
        DEFAULT_HF_REVISION,
        MiniCPMVConfig,
        MiniCPMVModel,
        MiniCPMVOptions,
        MiniCPMVRuntime,
    )
except ModuleNotFoundError as exc:
    if exc.name == "transformers":
        pytest.skip("transformers is required for MiniCPM wrapper unit tests", allow_module_level=True)
    raise

from cvsuite.common.core import FMRequest, ImageRecord, Record, VQA
from cvsuite.common.core import VisionDataset


def test_minicpm_load_runtime_uses_dtype_and_pinned_revision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_dir = (
        tmp_path
        / "weights"
        / "models--openbmb--MiniCPM-V-4"
        / "snapshots"
        / DEFAULT_HF_REVISION
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "configuration_minicpm.py",
        "modeling_minicpmv.py",
        "modeling_navit_siglip.py",
        "resampler.py",
        "model.safetensors.index.json",
    ):
        (snapshot_dir / name).write_text("", encoding="utf-8")

    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="minicpm_v", task="vlm", device="cuda", precision="fp16"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=MiniCPMVOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
    )
    captured: dict[str, dict[str, object]] = {}
    processor = object()
    tokenizer = object()

    class _Model:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1))

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

    monkeypatch.setattr(minicpm_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(minicpm_module.utils, "select_dtype", lambda device, precision: torch.float16)
    monkeypatch.setattr(MiniCPMVModel, "resolve_device_map", staticmethod(lambda pref: {"": "cuda"}))

    def _fake_processor_from_pretrained(*args, **kwargs):
        captured["processor"] = kwargs
        return processor

    def _fake_tokenizer_from_pretrained(*args, **kwargs):
        captured["tokenizer"] = kwargs
        return tokenizer

    monkeypatch.setattr(minicpm_module.AutoProcessor, "from_pretrained", _fake_processor_from_pretrained)
    monkeypatch.setattr(minicpm_module.AutoTokenizer, "from_pretrained", _fake_tokenizer_from_pretrained)

    def _fake_model_from_pretrained(*args, **kwargs):
        captured["model"] = kwargs
        return _Model()

    monkeypatch.setattr(minicpm_module.AutoModel, "from_pretrained", _fake_model_from_pretrained)

    runtime = MiniCPMVModel().load_runtime(dataset, ctx)

    assert captured["model"]["dtype"] == torch.float16
    assert "torch_dtype" not in captured["model"]
    assert captured["model"]["revision"] == DEFAULT_HF_REVISION
    assert captured["model"]["trust_remote_code"] is True
    assert captured["model"]["local_files_only"] is True
    assert captured["processor"]["revision"] == DEFAULT_HF_REVISION
    assert captured["processor"]["local_files_only"] is True
    assert captured["processor"]["use_fast"] is False
    assert captured["tokenizer"]["revision"] == DEFAULT_HF_REVISION
    assert captured["tokenizer"]["local_files_only"] is True
    assert runtime.revision == DEFAULT_HF_REVISION


def test_minicpm_suppresses_legacy_image_processor_register(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, object, object, object, bool]] = []

    def _fake_register(
        config_class,
        image_processor_class=None,
        slow_image_processor_class=None,
        fast_image_processor_class=None,
        exist_ok=False,
    ):
        calls.append(
            (config_class, image_processor_class, slow_image_processor_class, fast_image_processor_class, exist_ok)
        )
        return "forwarded"

    legacy_image_processor_cls = type("MiniCPMVImageProcessor", (), {})
    passthrough_image_processor_cls = type("OtherImageProcessor", (), {})

    monkeypatch.setattr(minicpm_module.AutoImageProcessor, "register", staticmethod(_fake_register))

    with minicpm_module._suppress_legacy_minicpm_image_processor_register():
        legacy_result = minicpm_module.AutoImageProcessor.register("MiniCPMVImageProcessor", legacy_image_processor_cls)
        passthrough_result = minicpm_module.AutoImageProcessor.register(
            "OtherConfig",
            passthrough_image_processor_cls,
            exist_ok=True,
        )

    assert legacy_result is None
    assert passthrough_result == "forwarded"
    assert calls == [("OtherConfig", passthrough_image_processor_cls, None, None, True)]
    assert minicpm_module.AutoImageProcessor.register is _fake_register


def test_minicpm_load_runtime_does_not_force_default_revision_for_other_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="minicpm_v", task="vlm", device="cpu"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=MiniCPMVOptions(model_id="openbmb/MiniCPM-V-2_6"),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
    )
    captured: dict[str, dict[str, object]] = {}

    class _Model:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1))

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

    monkeypatch.setattr(minicpm_module.utils, "select_device", lambda pref: torch.device("cpu"))
    monkeypatch.setattr(minicpm_module.utils, "select_dtype", lambda device, precision: torch.float32)
    monkeypatch.setattr(MiniCPMVModel, "resolve_device_map", staticmethod(lambda pref: {"": "cpu"}))
    monkeypatch.setattr(minicpm_module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(minicpm_module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    def _fake_model_from_pretrained(*args, **kwargs):
        captured["model"] = kwargs
        return _Model()

    monkeypatch.setattr(minicpm_module.AutoModel, "from_pretrained", _fake_model_from_pretrained)

    MiniCPMVModel().load_runtime(dataset, ctx)

    assert "revision" not in captured["model"]
    assert "local_files_only" not in captured["model"]


def test_minicpm_load_runtime_rejects_auto_nf4(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(provider="minicpm_v", task="vlm", device="auto", precision="nf4", max_gpu_memory="14GiB"),
    )
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=MiniCPMVOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        model_cache=tmp_path / "cache" / "minicpm_v",
    )

    monkeypatch.setattr(minicpm_module.utils, "select_device", lambda pref: torch.device("cuda"))

    with pytest.raises(RuntimeError, match="`--device auto --precision nf4` is not supported yet"):
        MiniCPMVModel().load_runtime(dataset, ctx)


def test_minicpm_patch_remote_init_calls_post_init(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class _FakeMiniCPMV:
        def __init__(self, config) -> None:
            self.config = config

        def post_init(self) -> None:
            calls.append("post_init")
            self.all_tied_weights_keys = {}

    fake_module = SimpleNamespace(MiniCPMV=_FakeMiniCPMV)

    monkeypatch.setattr(minicpm_module, "init_hf_modules", lambda: None)
    monkeypatch.setattr(minicpm_module, "HF_MODULES_CACHE", str(tmp_path))
    monkeypatch.setattr(minicpm_module.importlib, "import_module", lambda name: fake_module)

    tmp_path.mkdir(parents=True, exist_ok=True)
    minicpm_module._patch_remote_minicpm_init(DEFAULT_HF_MODEL_ID, DEFAULT_HF_REVISION)

    instance = _FakeMiniCPMV(object())

    assert calls == ["post_init"]
    assert instance.all_tied_weights_keys == {}


def test_minicpm_process_batch_passes_processor_to_chat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"stub")

    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=8, height=8),
                vqas=[VQA(question="What is shown?", answer="")],
            )
        ],
        fm_request=FMRequest(provider="minicpm_v", task="vlm", device="cpu"),
    )
    model = MiniCPMVModel()
    ctx = SimpleNamespace(prompt="", caller_cwd=tmp_path)
    jobs = model.build_jobs(dataset, ctx)
    captured: dict[str, object] = {}

    class _ChatModel:
        def chat(self, image=None, msgs=None, tokenizer=None, processor=None, max_new_tokens=2048, sampling=True, stream=False):
            captured["image"] = image
            captured["msgs"] = msgs
            captured["tokenizer"] = tokenizer
            captured["processor"] = processor
            captured["max_new_tokens"] = max_new_tokens
            captured["sampling"] = sampling
            captured["stream"] = stream
            return "a cat"

    monkeypatch.setattr(MiniCPMVModel, "load_rgb_image", staticmethod(lambda path: "IMAGE"))

    runtime = MiniCPMVRuntime(
        cfg=MiniCPMVConfig(max_new_tokens=32),
        revision=DEFAULT_HF_REVISION,
        device=torch.device("cpu"),
        dtype=torch.float32,
        model=_ChatModel(),
        processor=object(),
        tokenizer=object(),
        model_device=torch.device("cpu"),
    )

    result = model.process_batch(dataset, jobs, runtime, ctx)

    assert result.warnings == []
    assert result.modified_record_indices == [0]
    assert captured["image"] == "IMAGE"
    assert captured["msgs"] == [{"role": "user", "content": "What is shown?"}]
    assert captured["processor"] is runtime.processor
    assert captured["tokenizer"] is runtime.tokenizer
    assert captured["max_new_tokens"] == 32
    assert captured["sampling"] is False
    assert captured["stream"] is False
    assert dataset.records[0].vqas[0].answer == "a cat"
    assert dataset.records[0].vqas[0].model == "minicpm_v"


def test_minicpm_process_batch_uses_batched_chat_for_multiple_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")

    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_a, width=8, height=8),
                vqas=[VQA(question="What is shown?", answer="")],
            ),
            Record(
                image=ImageRecord(path=image_b, width=8, height=8),
                vqas=[VQA(question="What is shown?", answer="")],
            ),
        ],
        fm_request=FMRequest(provider="minicpm_v", task="vlm", device="cpu"),
    )
    model = MiniCPMVModel()
    ctx = SimpleNamespace(prompt="", caller_cwd=tmp_path)
    jobs = model.build_jobs(dataset, ctx)
    captured: dict[str, object] = {}

    class _ChatModel:
        def chat(self, image=None, msgs=None, tokenizer=None, processor=None, max_new_tokens=2048, sampling=True, stream=False):
            captured["image"] = image
            captured["msgs"] = msgs
            captured["tokenizer"] = tokenizer
            captured["processor"] = processor
            captured["max_new_tokens"] = max_new_tokens
            captured["sampling"] = sampling
            captured["stream"] = stream
            return ["a cat", "a dog"]

    monkeypatch.setattr(MiniCPMVModel, "load_rgb_image", staticmethod(lambda path: f"IMAGE:{path.name}"))

    runtime = MiniCPMVRuntime(
        cfg=MiniCPMVConfig(max_new_tokens=32),
        revision=DEFAULT_HF_REVISION,
        device=torch.device("cpu"),
        dtype=torch.float32,
        model=_ChatModel(),
        processor=object(),
        tokenizer=object(),
        model_device=torch.device("cpu"),
    )

    result = model.process_batch(dataset, jobs, runtime, ctx)

    assert result.warnings == []
    assert result.modified_record_indices == [0, 1]
    assert captured["image"] is None
    assert captured["msgs"] == [
        [{"role": "user", "content": ["IMAGE:a.jpg", "What is shown?"]}],
        [{"role": "user", "content": ["IMAGE:b.jpg", "What is shown?"]}],
    ]
    assert captured["processor"] is runtime.processor
    assert captured["tokenizer"] is runtime.tokenizer
    assert captured["max_new_tokens"] == 32
    assert captured["sampling"] is False
    assert captured["stream"] is False
    assert dataset.records[0].vqas[0].answer == "a cat"
    assert dataset.records[0].vqas[0].model == "minicpm_v"
    assert dataset.records[1].vqas[0].answer == "a dog"
    assert dataset.records[1].vqas[0].model == "minicpm_v"
