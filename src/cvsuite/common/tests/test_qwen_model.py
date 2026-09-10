from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

try:
    from cvsuite.common.fm.providers.vlm import qwen as qwen_module
    from cvsuite.common.fm.providers.vlm.qwen import QwenModel, QwenOptions
except ModuleNotFoundError as exc:
    if exc.name == "transformers":
        pytest.skip("transformers is required for qwen wrapper unit tests", allow_module_level=True)
    raise

from cvsuite.common.core import FMRequest
from cvsuite.common.core import VisionDataset


def test_patch_quantized_parameter_constructor_compat_preserves_unknown_attrs() -> None:
    class _FakeParam:
        def __new__(cls, data=None, requires_grad: bool = False, known=None):
            value = object.__new__(cls)
            value.data = data
            value.requires_grad = requires_grad
            value.known = known
            return value

    patched = qwen_module.utils.patch_quantized_parameter_constructor_compat(_FakeParam)

    assert patched is True

    value = _FakeParam("payload", requires_grad=True, known="ok", _is_hf_initialized=True, extra_note="kept")

    assert value.data == "payload"
    assert value.requires_grad is True
    assert value.known == "ok"
    assert value._is_hf_initialized is True
    assert value.extra_note == "kept"


def test_qwen_load_runtime_sets_tokenizer_padding_left(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="qwen", task="vlm", device="cpu"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=QwenOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
    )

    class _Tokenizer:
        padding_side = "right"

    class _Processor:
        def __init__(self) -> None:
            self.tokenizer = _Tokenizer()

    class _Model:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1))

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

    processor = _Processor()
    monkeypatch.setattr(qwen_module.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(model_type="qwen2_5_vl"))
    monkeypatch.setattr(qwen_module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(qwen_module.AutoModelForImageTextToText, "from_pretrained", lambda *args, **kwargs: _Model())
    monkeypatch.setattr(qwen_module.utils, "maybe_autocast", lambda device, precision: nullcontext())

    runtime = QwenModel().load_runtime(dataset, ctx)

    assert runtime.processor is processor
    assert runtime.model_type == "qwen2_5_vl"
    assert processor.tokenizer.padding_side == "left"


def test_qwen_load_runtime_forwards_torch_dtype(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="qwen", task="vlm", device="cuda", precision="fp16"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=QwenOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
    )
    captured: dict[str, object] = {}

    class _Tokenizer:
        padding_side = "right"

    class _Processor:
        def __init__(self) -> None:
            self.tokenizer = _Tokenizer()

    class _Model:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1))

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

    processor = _Processor()
    config = SimpleNamespace(model_type="qwen3_vl_moe")
    monkeypatch.setattr(qwen_module.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(qwen_module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)

    def _fake_model_from_pretrained(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _Model()

    monkeypatch.setattr(qwen_module.AutoModelForImageTextToText, "from_pretrained", _fake_model_from_pretrained)
    monkeypatch.setattr(qwen_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(QwenModel, "resolve_device_map", staticmethod(lambda pref: {"": "cuda"}))
    monkeypatch.setattr(qwen_module.utils, "maybe_autocast", lambda device, precision: nullcontext())

    QwenModel().load_runtime(dataset, ctx)

    assert "kwargs" in captured
    assert captured["kwargs"]["config"] is config
    assert captured["kwargs"]["torch_dtype"] == torch.float16
    assert "dtype" not in captured["kwargs"]


def test_qwen_load_runtime_rejects_non_vl_qwen_checkpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="qwen", task="vlm", device="cpu"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=QwenOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
    )
    calls = {"processor": 0, "model": 0}

    monkeypatch.setattr(qwen_module.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(model_type="qwen2"))
    monkeypatch.setattr(
        qwen_module.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: calls.__setitem__("processor", calls["processor"] + 1),
    )
    monkeypatch.setattr(
        qwen_module.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: calls.__setitem__("model", calls["model"] + 1),
    )

    with pytest.raises(ValueError, match="does not support checkpoint"):
        QwenModel().load_runtime(dataset, ctx)

    assert calls["processor"] == 0
    assert calls["model"] == 0


def test_qwen_load_runtime_warns_and_falls_back_to_fp32_for_unknown_precision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="qwen", task="vlm", device="cpu", precision="mystery"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=QwenOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
    )
    captured: dict[str, object] = {}

    class _Tokenizer:
        padding_side = "right"

    class _Processor:
        def __init__(self) -> None:
            self.tokenizer = _Tokenizer()

    class _Model:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1))

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

    monkeypatch.setattr(qwen_module.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(model_type="qwen2_5_vl"))
    monkeypatch.setattr(qwen_module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: _Processor())

    def _fake_model_from_pretrained(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _Model()

    monkeypatch.setattr(qwen_module.AutoModelForImageTextToText, "from_pretrained", _fake_model_from_pretrained)
    monkeypatch.setattr(qwen_module.utils, "maybe_autocast", lambda device, precision: nullcontext())

    QwenModel().load_runtime(dataset, ctx)

    err = capsys.readouterr().err
    assert "falling back to fp32" in err
    assert captured["kwargs"]["torch_dtype"] == torch.float32


def test_qwen_load_runtime_nf4_auto_installs_bitsandbytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from transformers.utils import import_utils as hf_import_utils

    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="qwen", task="vlm", device="cuda", precision="nf4"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=QwenOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        model_cache=tmp_path / "cache" / "qwen",
    )
    captured: dict[str, object] = {}
    dependency_state = {"installed": False}

    class _Tokenizer:
        padding_side = "right"

    class _Processor:
        def __init__(self) -> None:
            self.tokenizer = _Tokenizer()

    class _Model:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1))

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

    monkeypatch.setattr(qwen_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(qwen_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(qwen_module.torch.cuda, "is_bf16_supported", lambda: False, raising=False)
    monkeypatch.setattr(
        qwen_module.utils,
        "optional_dependency_available",
        lambda name: dependency_state["installed"] if name == "bitsandbytes" else True,
    )
    monkeypatch.setattr(hf_import_utils, "_bitsandbytes_available", False, raising=False)
    hf_import_utils.is_bitsandbytes_available.cache_clear()
    monkeypatch.setattr(
        "cvsuite.common.fm.providers.base.fm_utils.ensure_bitsandbytes_quant_parameter_compat",
        lambda: ["Params4bit"],
    )
    monkeypatch.setattr(qwen_module.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(model_type="qwen3_vl"))
    monkeypatch.setattr(qwen_module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: _Processor())

    def _fake_install(cmd, check, cwd, capture_output, text):
        captured["install_cmd"] = cmd
        captured["install_cwd"] = cwd
        dependency_state["installed"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def _fake_model_from_pretrained(*args, **kwargs):
        assert hf_import_utils.is_bitsandbytes_available(check_library_only=True) is True
        captured["model_kwargs"] = kwargs
        return _Model()

    monkeypatch.setattr("cvsuite.common.fm.providers.base.subprocess.run", _fake_install)
    monkeypatch.setattr(qwen_module.AutoModelForImageTextToText, "from_pretrained", _fake_model_from_pretrained)
    monkeypatch.setattr(qwen_module.utils, "maybe_autocast", lambda device, precision: nullcontext())
    monkeypatch.setattr(QwenModel, "resolve_device_map", staticmethod(lambda pref: {"": "cuda"}))

    QwenModel().load_runtime(dataset, ctx)

    err = capsys.readouterr().err
    assert "Installing optional dependency 'bitsandbytes'" in err
    assert "Installed optional dependency 'bitsandbytes'" in err
    assert captured["install_cmd"][:4] == [sys.executable, "-m", "pip", "install"]
    assert captured["install_cmd"][-2:] == ["--no-cache-dir", "bitsandbytes"]
    assert captured["install_cwd"] == ctx.model_cache
    assert "quantization_config" in captured["model_kwargs"]
    assert captured["model_kwargs"]["quantization_config"].bnb_4bit_use_double_quant is True


def test_qwen_load_runtime_with_auto_infers_offload_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(provider="qwen", task="vlm", device="auto", precision="fp16"),
    )
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=QwenOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        model_cache=tmp_path / "cache" / "qwen",
    )
    captured: dict[str, object] = {}

    class _Tokenizer:
        padding_side = "right"

    class _Processor:
        def __init__(self) -> None:
            self.tokenizer = _Tokenizer()

    class _Model:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1))

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

    monkeypatch.setattr(qwen_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(qwen_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(qwen_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(qwen_module.torch.cuda, "is_bf16_supported", lambda: False, raising=False)
    gib = 1024 ** 3
    monkeypatch.setattr(qwen_module.torch.cuda, "mem_get_info", lambda device=None: (8 * gib, 8 * gib))
    monkeypatch.setattr(qwen_module.utils, "available_cpu_memory_bytes", lambda: 10 * gib)
    monkeypatch.setattr(qwen_module.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(model_type="qwen3_vl"))
    monkeypatch.setattr(qwen_module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: _Processor())
    monkeypatch.setattr(qwen_module.utils, "maybe_autocast", lambda device, precision: nullcontext())

    def _fake_model_from_pretrained(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _Model()

    monkeypatch.setattr(qwen_module.AutoModelForImageTextToText, "from_pretrained", _fake_model_from_pretrained)

    runtime = QwenModel().load_runtime(dataset, ctx)

    assert captured["kwargs"]["device_map"] == "auto"
    assert captured["kwargs"]["max_memory"] == {0: 7 * gib, "cpu": 8 * gib}
    assert captured["kwargs"]["offload_folder"] == str(tmp_path / "cache" / "qwen" / "stage" / "offload")
    assert captured["kwargs"]["torch_dtype"] == torch.float16
    assert runtime.precision.effective == "fp16"


def test_qwen_load_runtime_rejects_auto_nf4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(provider="qwen", task="vlm", device="auto", precision="nf4", max_gpu_memory="14GiB"),
    )
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=QwenOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        model_cache=tmp_path / "cache" / "qwen",
    )

    monkeypatch.setattr(qwen_module.utils, "select_device", lambda pref: torch.device("cuda"))
    with pytest.raises(RuntimeError, match="`--device auto --precision nf4` is not supported yet"):
        QwenModel().load_runtime(dataset, ctx)


def test_qwen_load_runtime_nf4_install_failure_surfaces_manual_retry_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="qwen", task="vlm", device="cuda", precision="nf4"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=QwenOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
        model_cache=tmp_path / "cache" / "qwen",
    )

    monkeypatch.setattr(qwen_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(qwen_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(qwen_module.utils, "optional_dependency_available", lambda name: False if name == "bitsandbytes" else True)
    monkeypatch.setattr(
        "cvsuite.common.fm.providers.base.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="ERROR: install failed"),
    )

    with pytest.raises(RuntimeError, match="Automatic installation of optional dependency 'bitsandbytes' failed") as exc_info:
        QwenModel().load_runtime(dataset, ctx)

    assert f"{sys.executable} -m pip install --no-cache-dir bitsandbytes" in str(exc_info.value)
