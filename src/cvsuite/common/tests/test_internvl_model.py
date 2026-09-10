from __future__ import annotations

import io
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

try:
    from cvsuite.common.fm.providers.vlm import internvl as internvl_module
    from cvsuite.common.fm.providers.vlm.internvl import InternVLConfig, InternVLModel, InternVLOptions, InternVLRuntime
except ModuleNotFoundError as exc:
    if exc.name in {"transformers", "torchvision"}:
        pytest.skip("transformers and torchvision are required for InternVL wrapper unit tests", allow_module_level=True)
    raise

from cvsuite.common.core import FMRequest, ImageRecord, Record, VQA
from cvsuite.common.core import VisionDataset


def test_internvl_load_runtime_disables_flash_attn_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="internvl", task="vlm", device="cuda", precision="bf16"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=InternVLOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
    )
    ref_dir = ctx.weights_dir / "models--OpenGVLab--InternVL2_5-1B" / "refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "main").write_text("test-revision", encoding="utf-8")
    captured: dict[str, object] = {}

    class _Model:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1))
            self.to_calls: list[dict[str, object]] = []

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

        def to(self, *, device, dtype):
            self.to_calls.append({"device": device, "dtype": dtype})
            self._param = torch.nn.Parameter(self._param.to(device=device, dtype=dtype))
            return self

    monkeypatch.setattr(internvl_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(internvl_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(internvl_module.torch.cuda, "is_bf16_supported", lambda: True, raising=False)
    monkeypatch.setattr(internvl_module, "_flash_attn_available", lambda: False)
    monkeypatch.setattr(
        internvl_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(_commit_hash="test-revision"),
    )

    def _fake_tokenizer_from_pretrained(*args, **kwargs):
        captured["tokenizer_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(internvl_module.AutoTokenizer, "from_pretrained", _fake_tokenizer_from_pretrained)
    monkeypatch.setattr(
        internvl_module,
        "_patch_remote_internvl_encoder",
        lambda model_id, revision: captured.setdefault("patch_args", (model_id, revision)),
    )

    def _fake_model_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        model = _Model()
        captured["model"] = model
        return model

    monkeypatch.setattr(internvl_module.AutoModel, "from_pretrained", _fake_model_from_pretrained)

    runtime = InternVLModel().load_runtime(dataset, ctx)

    assert captured["dtype"] == torch.bfloat16
    assert captured["low_cpu_mem_usage"] is False
    assert "device_map" not in captured
    assert captured["use_flash_attn"] is False
    assert captured["revision"] == "test-revision"
    assert captured["patch_args"] == ("OpenGVLab/InternVL2_5-1B", "test-revision")
    assert captured["tokenizer_kwargs"]["revision"] == "test-revision"
    assert captured["model"].to_calls == [{"device": torch.device("cuda"), "dtype": torch.bfloat16}]
    assert runtime.vision_input_device == torch.device("cuda")
    assert runtime.vision_input_dtype == torch.bfloat16


def test_internvl_load_runtime_nf4_cuda_skips_post_load_cast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = VisionDataset(records=[], fm_request=FMRequest(provider="internvl", task="vlm", device="cuda", precision="nf4"))
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=InternVLOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
    )
    ref_dir = ctx.weights_dir / "models--OpenGVLab--InternVL2_5-1B" / "refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "main").write_text("test-revision", encoding="utf-8")
    captured: dict[str, object] = {}

    class _Model:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1))

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

        def to(self, *args, **kwargs):
            raise AssertionError("quantized InternVL load should not call model.to() after from_pretrained")

    monkeypatch.setattr(internvl_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(internvl_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(internvl_module, "_flash_attn_available", lambda: False)
    monkeypatch.setattr(
        internvl_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(_commit_hash="test-revision"),
    )
    monkeypatch.setattr(
        internvl_module,
        "_patch_remote_internvl_encoder",
        lambda model_id, revision: captured.setdefault("patch_args", (model_id, revision)),
    )
    monkeypatch.setattr(internvl_module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    def _fake_model_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        model = _Model()
        model.hf_device_map = {"": "cuda"}
        return model

    monkeypatch.setattr(internvl_module.AutoModel, "from_pretrained", _fake_model_from_pretrained)
    monkeypatch.setattr(
        InternVLModel,
        "resolve_runtime_precision",
        lambda self, dataset, ctx, device: SimpleNamespace(
            compute_dtype=torch.float16,
            quantization_mode="nf4",
            quantization_config=object(),
            warning=None,
        ),
    )

    runtime = InternVLModel().load_runtime(dataset, ctx)

    assert captured["patch_args"] == ("OpenGVLab/InternVL2_5-1B", "test-revision")
    assert "quantization_config" in captured
    assert captured["device_map"] == {"": "cuda"}
    assert "dtype" not in captured
    assert captured["low_cpu_mem_usage"] is False
    assert runtime.dtype == torch.float16


def test_internvl_load_runtime_with_gpu_cap_forwards_auto_offload_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(
            model="internvl",
            task="vlm",
            device="auto",
            precision="fp16",
            max_gpu_memory="14GiB",
        ),
    )
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=InternVLOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
    )
    ref_dir = ctx.weights_dir / "models--OpenGVLab--InternVL2_5-1B" / "refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "main").write_text("test-revision", encoding="utf-8")
    captured: dict[str, object] = {}

    class _Model:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1))
            self.to_calls: list[dict[str, object]] = []

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

        def to(self, *, device, dtype):
            self.to_calls.append({"device": device, "dtype": dtype})
            self._param = torch.nn.Parameter(self._param.to(device=device, dtype=dtype))
            return self

    monkeypatch.setattr(internvl_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(internvl_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(internvl_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(internvl_module.torch.cuda, "is_bf16_supported", lambda: False, raising=False)
    monkeypatch.setattr(internvl_module.utils, "available_cpu_memory_bytes", lambda: 9999)
    monkeypatch.setattr(internvl_module, "_flash_attn_available", lambda: False)
    monkeypatch.setattr(
        internvl_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(_commit_hash="test-revision"),
    )
    monkeypatch.setattr(
        internvl_module,
        "_patch_remote_internvl_encoder",
        lambda model_id, revision: captured.setdefault("patch_args", (model_id, revision)),
    )
    monkeypatch.setattr(internvl_module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    def _fake_model_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        model = _Model()
        captured["model"] = model
        return model

    monkeypatch.setattr(internvl_module.AutoModel, "from_pretrained", _fake_model_from_pretrained)

    runtime = InternVLModel().load_runtime(dataset, ctx)

    assert captured["dtype"] == torch.float16
    assert captured["low_cpu_mem_usage"] is True
    assert captured["device_map"] == "auto"
    assert captured["max_memory"] == {0: "14GiB", "cpu": 9999}
    assert captured["offload_folder"] == str(tmp_path / "weights" / "stage" / "offload")
    assert captured["model"].to_calls == []
    assert runtime.vision_input_device == runtime.model_device


def test_internvl_load_runtime_with_auto_infers_offload_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(provider="internvl", task="vlm", device="auto", precision="fp16"),
    )
    ctx = SimpleNamespace(
        request=dataset.fm_request,
        config_path=None,
        options=InternVLOptions(),
        weights_dir=tmp_path / "weights",
        hub_dir=tmp_path / "hub",
    )
    ref_dir = ctx.weights_dir / "models--OpenGVLab--InternVL2_5-1B" / "refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "main").write_text("test-revision", encoding="utf-8")
    captured: dict[str, object] = {}

    class _Model:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1))
            self.to_calls: list[dict[str, object]] = []

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

        def to(self, *, device, dtype):
            self.to_calls.append({"device": device, "dtype": dtype})
            self._param = torch.nn.Parameter(self._param.to(device=device, dtype=dtype))
            return self

    monkeypatch.setattr(internvl_module.utils, "select_device", lambda pref: torch.device("cuda"))
    monkeypatch.setattr(internvl_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(internvl_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(internvl_module.torch.cuda, "is_bf16_supported", lambda: False, raising=False)
    gib = 1024 ** 3
    monkeypatch.setattr(internvl_module.torch.cuda, "mem_get_info", lambda device=None: (8 * gib, 8 * gib))
    monkeypatch.setattr(internvl_module.utils, "available_cpu_memory_bytes", lambda: 10 * gib)
    monkeypatch.setattr(internvl_module, "_flash_attn_available", lambda: False)
    monkeypatch.setattr(
        internvl_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(_commit_hash="test-revision"),
    )
    monkeypatch.setattr(
        internvl_module,
        "_patch_remote_internvl_encoder",
        lambda model_id, revision: captured.setdefault("patch_args", (model_id, revision)),
    )
    monkeypatch.setattr(internvl_module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    def _fake_model_from_pretrained(*args, **kwargs):
        captured.update(kwargs)
        model = _Model()
        captured["model"] = model
        return model

    monkeypatch.setattr(internvl_module.AutoModel, "from_pretrained", _fake_model_from_pretrained)

    runtime = InternVLModel().load_runtime(dataset, ctx)

    assert captured["dtype"] == torch.float16
    assert captured["low_cpu_mem_usage"] is True
    assert captured["device_map"] == "auto"
    assert captured["max_memory"] == {0: 7 * gib, "cpu": 8 * gib}
    assert captured["offload_folder"] == str(tmp_path / "weights" / "stage" / "offload")
    assert captured["model"].to_calls == []
    assert runtime.vision_input_device == runtime.model_device


def test_force_cpu_linspace_when_default_is_meta() -> None:
    with internvl_module._force_cpu_linspace_when_default_is_meta():
        with torch.device("meta"):
            values = torch.linspace(0, 0.3, 4)

    assert values.device.type == "cpu"
    assert values.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_build_generation_config_prefers_tokenizer_pad_token_id() -> None:
    tokenizer = SimpleNamespace(pad_token_id=42, eos_token_id=151645)

    generation_config = internvl_module._build_generation_config(tokenizer, 64)

    assert generation_config == {"max_new_tokens": 64, "do_sample": False, "pad_token_id": 42}


def test_build_generation_config_falls_back_to_eos_token_id() -> None:
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=151645)

    generation_config = internvl_module._build_generation_config(tokenizer, 64)

    assert generation_config == {"max_new_tokens": 64, "do_sample": False, "pad_token_id": 151645}


def test_filtered_line_stream_suppresses_only_exact_lines() -> None:
    target = io.StringIO()
    stream = internvl_module._FilteredLineStream(target, ("FlashAttention2 is not installed.",))

    stream.write("hello\n")
    stream.write("FlashAttention2 is not installed.\n")
    stream.write("world")
    stream.flush()

    assert target.getvalue() == "hello\nworld"


def test_patch_remote_internvl_encoder_handles_meta_init(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeEncoderLayer(torch.nn.Module):
        def __init__(self, _config, drop_path_rate: float) -> None:
            super().__init__()
            self.drop_path_rate = drop_path_rate

    class _FakeEncoder(torch.nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
            self.layers = torch.nn.ModuleList([_FakeEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
            self.gradient_checkpointing = True

    fake_module = types.SimpleNamespace(
        InternVisionEncoder=_FakeEncoder,
        InternVisionEncoderLayer=_FakeEncoderLayer,
    )
    monkeypatch.setattr(internvl_module, "HF_MODULES_CACHE", str(tmp_path / "hf-modules"))
    monkeypatch.setattr(internvl_module, "init_hf_modules", lambda: None)
    monkeypatch.setattr(internvl_module.importlib, "import_module", lambda name: fake_module)
    (tmp_path / "hf-modules").mkdir(parents=True, exist_ok=True)

    cfg = SimpleNamespace(drop_path_rate=0.3, num_hidden_layers=4)

    with pytest.raises(RuntimeError, match="meta tensors"):
        with torch.device("meta"):
            fake_module.InternVisionEncoder(cfg)

    internvl_module._patch_remote_internvl_encoder("OpenGVLab/InternVL2_5-4B", "test-revision")

    with torch.device("meta"):
        encoder = fake_module.InternVisionEncoder(cfg)

    assert [layer.drop_path_rate for layer in encoder.layers] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert encoder.gradient_checkpointing is True
    assert getattr(fake_module.InternVisionEncoder, "_cvsuite_drop_path_patched", False) is True


def test_patch_remote_internvl_attention_uses_sdpa(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeEncoderLayer(torch.nn.Module):
        def __init__(self, _config, _drop_path_rate: float) -> None:
            super().__init__()

    class _FakeEncoder(torch.nn.Module):
        def __init__(self, _config) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList()
            self.gradient_checkpointing = True

    class _FakeAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 2
            self.scale = 0.5
            self.qkv = torch.nn.Linear(8, 24, bias=False)
            self.attn_drop = torch.nn.Dropout(0.0)
            self.proj = torch.nn.Linear(8, 8, bias=False)
            self.proj_drop = torch.nn.Dropout(0.0)
            self.qk_normalization = False

        def _naive_attn(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = ((q * self.scale) @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    fake_module = types.SimpleNamespace(
        InternVisionEncoder=_FakeEncoder,
        InternVisionEncoderLayer=_FakeEncoderLayer,
        InternAttention=_FakeAttention,
    )
    monkeypatch.setattr(internvl_module, "HF_MODULES_CACHE", str(tmp_path / "hf-modules"))
    monkeypatch.setattr(internvl_module, "init_hf_modules", lambda: None)
    monkeypatch.setattr(internvl_module.importlib, "import_module", lambda name: fake_module)
    (tmp_path / "hf-modules").mkdir(parents=True, exist_ok=True)

    attention = _FakeAttention().eval()
    x = torch.randn(2, 4, 8)
    expected = _FakeAttention._naive_attn(attention, x)

    internvl_module._patch_remote_internvl_encoder("OpenGVLab/InternVL2_5-4B", "test-revision")

    actual = attention._naive_attn(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert getattr(fake_module.InternAttention, "_cvsuite_sdpa_patched", False) is True


def test_patch_remote_internvl_chat_routes_text_to_language_device(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeEncoderLayer(torch.nn.Module):
        def __init__(self, _config, _drop_path_rate: float) -> None:
            super().__init__()

    class _FakeEncoder(torch.nn.Module):
        def __init__(self, _config) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList()
            self.gradient_checkpointing = True

    class _Template:
        roles = ("user", "assistant")
        sep = "<sep>"

        def __init__(self) -> None:
            self.system_message = ""
            self.messages: list[tuple[str, str | None]] = []

        def append_message(self, role: str, message: str | None) -> None:
            self.messages.append((role, message))

        def get_prompt(self) -> str:
            return "<image>\nDescribe the image."

    class _Tokenizer:
        padding_side = "right"

        def convert_tokens_to_ids(self, token: str) -> int:
            if token == "<IMG_CONTEXT>":
                return 7
            return 11

        def __call__(self, _queries, return_tensors="pt", padding=False):
            return {
                "input_ids": torch.tensor([[1, 7, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

        def batch_decode(self, _values, skip_special_tokens=True):
            return ["answer<sep>tail"]

    class _LanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.generate_calls: list[dict[str, torch.Tensor | dict[str, object] | bool | None]] = []

        def get_input_embeddings(self):
            return self.embedding

        def generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    class _FakeChatModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = _LanguageModel()
            self.template = "chat"
            self.system_message = "system"
            self.num_image_token = 1
            self.img_context_token_id = None
            self.device = torch.device("meta")

        def extract_feature(self, pixel_values):
            return torch.ones((pixel_values.shape[0], 4), dtype=self.language_model.embedding.weight.dtype)

    vit_module = types.SimpleNamespace(
        InternVisionEncoder=_FakeEncoder,
        InternVisionEncoderLayer=_FakeEncoderLayer,
    )
    chat_module = types.SimpleNamespace(
        InternVLChatModel=_FakeChatModel,
        get_conv_template=lambda _name: _Template(),
    )

    monkeypatch.setattr(internvl_module, "HF_MODULES_CACHE", str(tmp_path / "hf-modules"))
    monkeypatch.setattr(internvl_module, "init_hf_modules", lambda: None)
    monkeypatch.setattr(
        internvl_module.importlib,
        "import_module",
        lambda name: chat_module if name.endswith("modeling_internvl_chat") else vit_module,
    )
    (tmp_path / "hf-modules").mkdir(parents=True, exist_ok=True)

    internvl_module._patch_remote_internvl_encoder("OpenGVLab/InternVL2_5-4B", "test-revision")

    model = chat_module.InternVLChatModel()
    tokenizer = _Tokenizer()
    pixel_values = torch.zeros(1, 1)

    responses = model.batch_chat(
        tokenizer,
        pixel_values,
        questions=["Describe the image."],
        generation_config={},
        num_patches_list=[1],
    )
    answer = model.chat(tokenizer, pixel_values, "<image>\nDescribe the image.", generation_config={})

    assert responses == ["answer"]
    assert answer == "answer"
    assert len(model.language_model.generate_calls) == 2
    assert [call["input_ids"].device for call in model.language_model.generate_calls] == [
        torch.device("cpu"),
        torch.device("cpu"),
    ]
    assert [call["attention_mask"].device for call in model.language_model.generate_calls] == [
        torch.device("cpu"),
        torch.device("cpu"),
    ]
    assert [call["inputs_embeds"].device for call in model.language_model.generate_calls] == [
        torch.device("cpu"),
        torch.device("cpu"),
    ]
    assert all(call["use_cache"] is True for call in model.language_model.generate_calls)
    assert getattr(chat_module.InternVLChatModel, "_cvsuite_text_device_patched", False) is True


def test_patch_loaded_internvl_chat_model_routes_text_to_language_device(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "tests.fake_internvl_chat_module"

    class _Template:
        roles = ("user", "assistant")
        sep = "<sep>"

        def __init__(self) -> None:
            self.system_message = ""
            self.messages: list[tuple[str, str | None]] = []

        def append_message(self, role: str, message: str | None) -> None:
            self.messages.append((role, message))

        def get_prompt(self) -> str:
            return "<image>\nDescribe the image."

    class _Tokenizer:
        padding_side = "right"

        def convert_tokens_to_ids(self, token: str) -> int:
            if token == "<IMG_CONTEXT>":
                return 7
            return 11

        def __call__(self, _queries, return_tensors="pt", padding=False):
            return {
                "input_ids": torch.tensor([[1, 7, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

        def batch_decode(self, _values, skip_special_tokens=True):
            return ["answer<sep>tail"]

    class _LanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.generate_calls: list[dict[str, torch.Tensor | dict[str, object] | bool | None]] = []

        def get_input_embeddings(self):
            return self.embedding

        def generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    class _FakeChatModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = _LanguageModel()
            self.template = "chat"
            self.system_message = "system"
            self.num_image_token = 1
            self.img_context_token_id = None
            self.device = torch.device("meta")

        def extract_feature(self, pixel_values):
            return torch.ones((pixel_values.shape[0], 4), dtype=self.language_model.embedding.weight.dtype)

    _FakeChatModel.__module__ = module_name
    chat_module = types.SimpleNamespace(
        InternVLChatModel=_FakeChatModel,
        get_conv_template=lambda _name: _Template(),
    )
    monkeypatch.setitem(sys.modules, module_name, chat_module)

    model = _FakeChatModel()
    tokenizer = _Tokenizer()
    pixel_values = torch.zeros(1, 1)

    internvl_module._patch_loaded_internvl_chat_model(model)

    responses = model.batch_chat(
        tokenizer,
        pixel_values,
        questions=["Describe the image."],
        generation_config={},
        num_patches_list=[1],
    )
    answer = model.chat(tokenizer, pixel_values, "<image>\nDescribe the image.", generation_config={})

    assert responses == ["answer"]
    assert answer == "answer"
    assert len(model.language_model.generate_calls) == 2
    assert [call["input_ids"].device for call in model.language_model.generate_calls] == [
        torch.device("cpu"),
        torch.device("cpu"),
    ]
    assert [call["attention_mask"].device for call in model.language_model.generate_calls] == [
        torch.device("cpu"),
        torch.device("cpu"),
    ]
    assert [call["inputs_embeds"].device for call in model.language_model.generate_calls] == [
        torch.device("cpu"),
        torch.device("cpu"),
    ]
    assert getattr(model.__class__, "_cvsuite_text_device_patched", False) is True


def test_patch_loaded_internvl_chat_suppresses_meta_generate_device_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "tests.fake_internvl_chat_warning_module"

    class _Template:
        roles = ("user", "assistant")
        sep = "<sep>"

        def __init__(self) -> None:
            self.system_message = ""
            self.messages: list[tuple[str, str | None]] = []

        def append_message(self, role: str, message: str | None) -> None:
            self.messages.append((role, message))

        def get_prompt(self) -> str:
            return "<image>\nDescribe the image."

    class _Tokenizer:
        padding_side = "right"

        def convert_tokens_to_ids(self, token: str) -> int:
            if token == "<IMG_CONTEXT>":
                return 7
            return 11

        def __call__(self, _queries, return_tensors="pt", padding=False):
            return {
                "input_ids": torch.tensor([[1, 7, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

        def batch_decode(self, _values, skip_special_tokens=True):
            return ["answer<sep>tail"]

    class _LanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)

        @property
        def device(self):
            return torch.device("meta")

        def get_input_embeddings(self):
            return self.embedding

        def generate(self, **_kwargs):
            warnings.warn_explicit(
                "You are calling .generate() with the `input_ids` being on a device type different than your model's device. "
                "`input_ids` is on cpu, whereas the model is on meta. You may experience unexpected behaviors or slower generation.",
                UserWarning,
                filename="generation/utils.py",
                lineno=2534,
                module="transformers.generation.utils",
            )
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    class _FakeChatModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = _LanguageModel()
            self.template = "chat"
            self.system_message = "system"
            self.num_image_token = 1
            self.img_context_token_id = None
            self.device = torch.device("meta")

        def extract_feature(self, pixel_values):
            return torch.ones((pixel_values.shape[0], 4), dtype=self.language_model.embedding.weight.dtype)

    _FakeChatModel.__module__ = module_name
    chat_module = types.SimpleNamespace(
        InternVLChatModel=_FakeChatModel,
        get_conv_template=lambda _name: _Template(),
    )
    monkeypatch.setitem(sys.modules, module_name, chat_module)

    model = _FakeChatModel()
    tokenizer = _Tokenizer()
    pixel_values = torch.zeros(1, 1)

    internvl_module._patch_loaded_internvl_chat_model(model)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        answer = model.chat(tokenizer, pixel_values, "<image>\nDescribe the image.", generation_config={})

    assert answer == "answer"
    assert not any("You are calling .generate()" in str(w.message) for w in caught)


def test_internvl_process_batch_uses_batch_chat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        fm_request=FMRequest(provider="internvl", task="vlm", device="cpu"),
    )
    model = InternVLModel()
    ctx = SimpleNamespace(prompt="", caller_cwd=tmp_path)
    jobs = model.build_jobs(dataset, ctx)
    captured: dict[str, object] = {}

    class _BatchModel:
        def batch_chat(self, tokenizer, pixel_values, *, num_patches_list=None, questions=None, generation_config=None):
            captured["tokenizer"] = tokenizer
            captured["pixel_values"] = pixel_values
            captured["num_patches_list"] = num_patches_list
            captured["questions"] = questions
            captured["generation_config"] = generation_config
            return ["a cat", "a dog"]

    monkeypatch.setattr(InternVLModel, "load_rgb_image", staticmethod(lambda path: path.name))

    def _fake_load_pixel_values(image, input_size, max_num):
        if image == "a.jpg":
            return torch.full((2, 3, 4, 4), 1.0)
        return torch.full((1, 3, 4, 4), 2.0)

    monkeypatch.setattr(internvl_module, "_load_pixel_values", _fake_load_pixel_values)

    runtime = InternVLRuntime(
        cfg=InternVLConfig(max_new_tokens=32, input_size=448, max_num=12),
        revision="test-revision",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        model=_BatchModel(),
        tokenizer=object(),
        model_device=torch.device("cpu"),
        vision_input_device=torch.device("cpu"),
        vision_input_dtype=torch.float16,
    )

    result = model.process_batch(dataset, jobs, runtime, ctx)

    assert result.warnings == []
    assert result.modified_record_indices == [0, 1]
    assert captured["tokenizer"] is runtime.tokenizer
    assert captured["num_patches_list"] == [2, 1]
    assert captured["questions"] == ["<image>\nWhat is shown?", "<image>\nWhat is shown?"]
    assert captured["generation_config"] == {"max_new_tokens": 32, "do_sample": False}
    assert torch.equal(
        captured["pixel_values"],
        torch.cat([torch.full((2, 3, 4, 4), 1.0), torch.full((1, 3, 4, 4), 2.0)], dim=0),
    )
    assert captured["pixel_values"].dtype == torch.float16
    assert dataset.records[0].vqas[0].answer == "a cat"
    assert dataset.records[0].vqas[0].model == "internvl"
    assert dataset.records[1].vqas[0].answer == "a dog"
    assert dataset.records[1].vqas[0].model == "internvl"


def test_internvl_process_batch_falls_back_to_serial_chat_on_oom(
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
        fm_request=FMRequest(provider="internvl", task="vlm", device="cpu"),
    )
    model = InternVLModel()
    ctx = SimpleNamespace(prompt="", caller_cwd=tmp_path, request=dataset.fm_request, batch_size=2)
    jobs = model.build_jobs(dataset, ctx)
    captured: dict[str, object] = {"chat_calls": []}

    class _BatchModel:
        def batch_chat(self, tokenizer, pixel_values, *, num_patches_list=None, questions=None, generation_config=None):
            raise torch.OutOfMemoryError("simulated oom")

        def chat(self, tokenizer, pixel_values, question, generation_config):
            captured["chat_calls"].append({"pixel_values": pixel_values.clone(), "question": question})
            return "fallback answer"

    monkeypatch.setattr(InternVLModel, "load_rgb_image", staticmethod(lambda path: path.name))

    def _fake_load_pixel_values(image, input_size, max_num):
        if image == "a.jpg":
            return torch.full((2, 3, 4, 4), 1.0)
        return torch.full((1, 3, 4, 4), 2.0)

    monkeypatch.setattr(internvl_module, "_load_pixel_values", _fake_load_pixel_values)

    runtime = InternVLRuntime(
        cfg=InternVLConfig(max_new_tokens=32, input_size=448, max_num=12),
        revision="test-revision",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        model=_BatchModel(),
        tokenizer=object(),
        model_device=torch.device("cpu"),
        vision_input_device=torch.device("cpu"),
        vision_input_dtype=torch.float16,
    )

    result = model.process_batch(dataset, jobs, runtime, ctx)

    assert result.modified_record_indices == [0, 1]
    assert result.warnings == [
        "[internvl] batch_chat ran out of CUDA memory; retrying serial chat for 2 job(s) / 3 tile(s)."
    ]
    assert len(captured["chat_calls"]) == 2
    assert all(call["pixel_values"].dtype == torch.float16 for call in captured["chat_calls"])
    assert dataset.records[0].vqas[0].answer == "fallback answer"
    assert dataset.records[1].vqas[0].answer == "fallback answer"
