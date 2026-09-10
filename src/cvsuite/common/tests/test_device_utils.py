from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cvsuite.common.fm.core import utils  # noqa: E402
from cvsuite.common.fm.providers.base import BaseFMModel  # noqa: E402
from cvsuite.common.fm.providers.bases.vlm import resolve_device_map, resolve_hf_load_placement  # noqa: E402


def test_select_device_explicit_cuda_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(RuntimeError, match="CUDA was explicitly requested"):
        utils.select_device("cuda")


def test_select_device_explicit_cuda_returns_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert utils.select_device("cuda").type == "cuda"
    assert utils.select_device("gpu").type == "cuda"


def test_resolve_device_map_explicit_cuda_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", None, raising=False)

    with pytest.raises(RuntimeError, match="CUDA was explicitly requested"):
        resolve_device_map("cuda")


def test_resolve_device_map_explicit_cuda_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device_map("cuda") == {"": "cuda"}
    assert resolve_device_map("gpu") == {"": "cuda"}


def test_resolve_hf_load_placement_without_cap_infers_gpu_cpu_and_offload_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    gib = 1024 ** 3
    mem_info = {
        0: (8 * gib, 8 * gib),
        1: (6 * gib, 8 * gib),
    }
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: mem_info[int(device if device is not None else 0)])
    monkeypatch.setattr(utils, "available_cpu_memory_bytes", lambda: 10 * gib)

    placement = resolve_hf_load_placement(
        SimpleNamespace(device="auto", max_gpu_memory=None),
        device=torch.device("cuda"),
        stage_dir=tmp_path / "stage",
        model_cache=tmp_path / "cache",
        default_device_map="auto",
    )

    assert placement.device_map == "auto"
    assert placement.max_memory == {
        0: 7 * gib,
        1: 5 * gib,
        "cpu": 8 * gib,
    }
    assert placement.offload_folder == str(tmp_path / "stage" / "offload")
    assert (tmp_path / "stage" / "offload").is_dir()
    assert placement.offload_cap_enabled is True
    assert placement.managed_device_map_active is True
    assert placement.memory_budget_source == "inferred"


def test_resolve_hf_load_placement_with_cap_uses_gpu_cpu_and_offload_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(utils, "available_cpu_memory_bytes", lambda: 123456)

    placement = resolve_hf_load_placement(
        SimpleNamespace(device="auto", max_gpu_memory="14GiB"),
        device=torch.device("cuda"),
        stage_dir=tmp_path / "stage",
        model_cache=tmp_path / "cache",
        default_device_map={"": "cuda"},
    )

    assert placement.device_map == "auto"
    assert placement.max_memory == {0: "14GiB", 1: "14GiB", "cpu": 123456}
    assert placement.offload_folder == str(tmp_path / "stage" / "offload")
    assert (tmp_path / "stage" / "offload").is_dir()
    assert placement.offload_cap_enabled is True
    assert placement.managed_device_map_active is True
    assert placement.memory_budget_source == "explicit"


def test_summarize_hf_auto_device_report_groups_parameter_storage(tmp_path: Path) -> None:
    model = torch.nn.Module()
    model.vision = torch.nn.Linear(4, 4, bias=False)
    model.language = torch.nn.Linear(8, 8, bias=False)
    model.hf_device_map = {"vision": 0, "language": "cpu"}
    placement = SimpleNamespace(
        managed_device_map_active=True,
        max_memory={0: 12 * 1024**3, "cpu": 24 * 1024**3},
        offload_folder=str(tmp_path / "offload"),
    )

    lines = BaseFMModel.summarize_hf_auto_device_report(model, placement)

    text = "\n".join(lines)
    assert "managed auto device placement" in text
    assert "budget: cuda:0=12.0 GiB, cpu=24.0 GiB" in text
    assert "estimated parameter storage: total ~320 B" in text
    assert "cpu ~256 B (80%)" in text
    assert "cuda:0 ~64 B (20%)" in text
    assert "language->cpu ~256 B" in text
    assert "vision->cuda:0 ~64 B" in text


def test_emit_hf_auto_device_report_prints_only_for_managed_auto(capsys: pytest.CaptureFixture[str]) -> None:
    wrapper = BaseFMModel()
    wrapper.model_name = "dummy"
    model = torch.nn.Linear(2, 2, bias=False)
    wrapper.emit_hf_auto_device_report(
        model,
        SimpleNamespace(managed_device_map_active=False, max_memory=None, offload_folder=None),
    )
    assert capsys.readouterr().err == ""

    wrapper.emit_hf_auto_device_report(
        model,
        SimpleNamespace(managed_device_map_active=True, max_memory=None, offload_folder=None),
    )
    assert "[dummy] managed auto device placement:" in capsys.readouterr().err


def test_resolve_hf_load_placement_rejects_cap_without_auto_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(ValueError, match="requires `--device auto`"):
        resolve_hf_load_placement(
            SimpleNamespace(device="gpu", max_gpu_memory="14GiB"),
            device=torch.device("cuda"),
            stage_dir=tmp_path / "stage",
            model_cache=tmp_path / "cache",
            default_device_map={"": "cuda"},
        )


def test_resolve_hf_load_placement_explicit_cuda_preserves_single_device_behavior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    placement = resolve_hf_load_placement(
        SimpleNamespace(device="cuda", max_gpu_memory=None),
        device=torch.device("cuda"),
        stage_dir=tmp_path / "stage",
        model_cache=tmp_path / "cache",
        default_device_map={"": "cuda"},
    )

    assert placement.device_map == {"": "cuda"}
    assert placement.max_memory is None
    assert placement.offload_cap_enabled is False
    assert placement.managed_device_map_active is False
    assert placement.memory_budget_source == "none"


def test_ensure_hf_caches_loads_token_from_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace" / "nested"
    workspace.mkdir(parents=True)
    (tmp_path / "workspace" / ".env").write_text('HF_TOKEN="hf-dotenv-token"\n', encoding="utf-8")

    monkeypatch.setenv("FM_CALLER_CWD", str(workspace))
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(utils, "_HF_TOKEN_VALIDITY_CACHE", {})
    monkeypatch.setattr(utils, "_hf_api_whoami", lambda token: {"name": "tester", "token": token})

    weights_dir = tmp_path / "weights"
    utils.ensure_hf_caches(weights_dir)

    assert os.environ["HUGGINGFACE_HUB_TOKEN"] == "hf-dotenv-token"
    assert os.environ["HF_TOKEN"] == "hf-dotenv-token"
    assert os.environ["HF_HOME"] == str(weights_dir / "hf_home")


def test_ensure_hf_caches_clears_deprecated_transformers_cache_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "legacy-transformers"))
    monkeypatch.setenv("PYTORCH_TRANSFORMERS_CACHE", str(tmp_path / "legacy-pytorch-transformers"))
    monkeypatch.setenv("PYTORCH_PRETRAINED_BERT_CACHE", str(tmp_path / "legacy-bert"))

    utils.ensure_hf_caches(tmp_path / "weights")

    assert os.environ["HF_HOME"] == str(tmp_path / "weights" / "hf_home")
    assert "TRANSFORMERS_CACHE" not in os.environ
    assert "PYTORCH_TRANSFORMERS_CACHE" not in os.environ
    assert "PYTORCH_PRETRAINED_BERT_CACHE" not in os.environ


def test_ensure_hf_caches_warns_for_invalid_dotenv_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (workspace / ".env").write_text("HUGGINGFACE_HUB_TOKEN=hf-invalid\n", encoding="utf-8")

    class _Unauthorized(Exception):
        def __init__(self) -> None:
            super().__init__("401 Client Error: Unauthorized")
            self.response = SimpleNamespace(status_code=401)

    monkeypatch.setenv("FM_CALLER_CWD", str(workspace))
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(utils, "_HF_TOKEN_VALIDITY_CACHE", {})
    monkeypatch.setattr(utils, "_hf_api_whoami", lambda token: (_ for _ in ()).throw(_Unauthorized()))

    with pytest.warns(RuntimeWarning, match="appear[s]? invalid|appears invalid"):
        utils.ensure_hf_caches(tmp_path / "weights")

    assert "HUGGINGFACE_HUB_TOKEN" not in os.environ
    assert "HF_TOKEN" not in os.environ


def test_describe_hf_access_error_reports_gated_repo_token_issue() -> None:
    class _Forbidden(Exception):
        def __init__(self) -> None:
            super().__init__(
                "403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings."
            )
            self.response = SimpleNamespace(status_code=403)

    outer = OSError("We couldn't connect to 'https://huggingface.co' to load the files.")
    outer.__cause__ = _Forbidden()

    detail = utils.describe_hf_access_error(outer, "google/paligemma2-3b-mix-448")

    assert detail is not None
    assert "google/paligemma2-3b-mix-448" in detail
    assert "public gated repositories" in detail


def test_describe_hf_access_error_reports_gated_repo_auth_issue() -> None:
    class _Unauthorized(Exception):
        def __init__(self) -> None:
            super().__init__(
                "Cannot access gated repo for url https://huggingface.co/facebook/sam3/resolve/main/config.json."
            )
            self.response = SimpleNamespace(status_code=401)

    outer = OSError("We couldn't connect to 'https://huggingface.co' to load the files.")
    outer.__cause__ = _Unauthorized()

    detail = utils.describe_hf_access_error(outer, "facebook/sam3")

    assert detail is not None
    assert "facebook/sam3" in detail
    assert "not authenticated" in detail
    assert "HF_TOKEN/HUGGINGFACE_HUB_TOKEN" in detail


def test_describe_hf_access_error_ignores_unrelated_errors() -> None:
    detail = utils.describe_hf_access_error(OSError("disk full"), "google/paligemma2-3b-mix-448")
    assert detail is None


def test_resolve_precision_unknown_warns_and_falls_back_to_fp32() -> None:
    resolved = utils.resolve_precision(torch.device("cpu"), "mystery", allow_nf4=False)

    assert resolved.requested == "mystery"
    assert resolved.effective == "fp32"
    assert resolved.compute_dtype == torch.float32
    assert resolved.warning is not None
    assert "falling back to fp32" in resolved.warning


def test_resolve_precision_fp16_on_cpu_warns_and_falls_back_to_fp32() -> None:
    resolved = utils.resolve_precision(torch.device("cpu"), "fp16", allow_nf4=False)

    assert resolved.requested == "fp16"
    assert resolved.effective == "fp32"
    assert resolved.compute_dtype == torch.float32
    assert resolved.warning is not None
    assert "requires CUDA" in resolved.warning


def test_resolve_precision_nf4_builds_4bit_quant_config(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("transformers")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False, raising=False)

    resolved = utils.resolve_precision(torch.device("cuda"), "nf4", allow_nf4=True)

    assert resolved.effective == "nf4"
    assert resolved.compute_dtype == torch.float16
    assert resolved.quantization_mode == "nf4"
    assert resolved.quantization_config is not None
    assert resolved.quantization_config.load_in_4bit is True
    assert resolved.quantization_config.bnb_4bit_quant_type == "nf4"


@pytest.mark.parametrize(
    ("device", "cuda_available"),
    [
        (torch.device("cpu"), True),
        (torch.device("cuda"), False),
    ],
)
def test_resolve_precision_nf4_requires_cuda(device: torch.device, cuda_available: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        utils.resolve_precision(device, "nf4", allow_nf4=True)


def test_materialize_hf_model_source_promotes_staged_repo_into_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_snapshot_download(*, repo_id, cache_dir, revision=None, token=None, allow_patterns=None):
        repo_cache = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
        snapshot_dir = repo_cache / "snapshots" / "rev-1"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
        (repo_cache / "refs").mkdir(parents=True, exist_ok=True)
        (repo_cache / "refs" / "main").write_text("rev-1", encoding="utf-8")
        (repo_cache / "blobs").mkdir(parents=True, exist_ok=True)
        return str(snapshot_dir)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=_fake_snapshot_download))

    weights_dir = tmp_path / "weights"
    stage_dir = tmp_path / "stage"
    source = utils.materialize_hf_model_source(
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        hub_dir=weights_dir,
        stage_dir=stage_dir,
        caller_cwd=tmp_path,
    )

    repo_cache = weights_dir / "models--Qwen--Qwen3-VL-30B-A3B-Instruct"
    assert source.local_files_only is True
    assert source.cache_dir == str(weights_dir)
    assert source.revision == "rev-1"
    assert source.repo_cache_dir == repo_cache
    assert source.snapshot_path == repo_cache / "snapshots" / "rev-1"
    assert (repo_cache / "snapshots" / "rev-1" / "config.json").exists()
    assert not any(path.name.startswith("hf-") for path in stage_dir.iterdir())
    assert not (stage_dir / "models--Qwen--Qwen3-VL-30B-A3B-Instruct").exists()


def test_materialize_hf_model_source_reuses_existing_weights_without_redownload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_cache = tmp_path / "weights" / "models--openbmb--MiniCPM-V-4"
    snapshot_dir = repo_cache / "snapshots" / "rev-existing"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
    (repo_cache / "refs").mkdir(parents=True, exist_ok=True)
    (repo_cache / "refs" / "main").write_text("rev-existing", encoding="utf-8")

    def _unexpected_snapshot_download(**kwargs):
        raise AssertionError("snapshot_download should not run when durable weights already exist")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=_unexpected_snapshot_download))

    source = utils.materialize_hf_model_source(
        "openbmb/MiniCPM-V-4",
        hub_dir=tmp_path / "weights",
        stage_dir=tmp_path / "stage",
        caller_cwd=tmp_path,
    )

    assert source.local_files_only is True
    assert source.revision == "rev-existing"
    assert source.snapshot_path == snapshot_dir


def test_materialize_hf_model_source_replaces_old_revision_when_new_one_is_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_cache = tmp_path / "weights" / "models--OpenGVLab--InternVL2_5-1B"
    old_snapshot = repo_cache / "snapshots" / "old-rev"
    old_snapshot.mkdir(parents=True, exist_ok=True)
    (old_snapshot / "config.json").write_text("old", encoding="utf-8")
    (repo_cache / "refs").mkdir(parents=True, exist_ok=True)
    (repo_cache / "refs" / "main").write_text("old-rev", encoding="utf-8")

    def _fake_snapshot_download(*, repo_id, cache_dir, revision=None, token=None, allow_patterns=None):
        repo_cache = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
        snapshot_dir = repo_cache / "snapshots" / str(revision)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "config.json").write_text("new", encoding="utf-8")
        (repo_cache / "refs").mkdir(parents=True, exist_ok=True)
        (repo_cache / "refs" / "main").write_text(str(revision), encoding="utf-8")
        return str(snapshot_dir)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=_fake_snapshot_download))

    source = utils.materialize_hf_model_source(
        "OpenGVLab/InternVL2_5-1B",
        hub_dir=tmp_path / "weights",
        stage_dir=tmp_path / "stage",
        caller_cwd=tmp_path,
        revision="new-rev",
    )

    assert source.revision == "new-rev"
    assert not old_snapshot.exists()
    assert (repo_cache / "snapshots" / "new-rev" / "config.json").exists()
    assert not any(path.name.startswith("hf-") for path in (tmp_path / "stage").iterdir())
    assert not ((tmp_path / "stage") / "models--OpenGVLab--InternVL2_5-1B").exists()


def test_maybe_autocast_for_module_prefers_loaded_module_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[torch.device, str | utils.ResolvedPrecision]] = []

    def _fake_maybe_autocast(device: torch.device, precision: str | utils.ResolvedPrecision):
        calls.append((device, precision))
        return "sentinel"

    monkeypatch.setattr(utils, "maybe_autocast", _fake_maybe_autocast)
    module = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.float32)
    requested = utils.ResolvedPrecision(
        requested="bf16",
        normalized="bf16",
        effective="bf16",
        compute_dtype=torch.bfloat16,
    )

    result = utils.maybe_autocast_for_module(torch.device("cuda"), requested, module)

    assert result == "sentinel"
    assert calls == [(torch.device("cuda"), "fp32")]


def test_maybe_autocast_for_module_falls_back_when_module_has_no_float_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[torch.device, str | utils.ResolvedPrecision]] = []

    def _fake_maybe_autocast(device: torch.device, precision: str | utils.ResolvedPrecision):
        calls.append((device, precision))
        return "sentinel"

    monkeypatch.setattr(utils, "maybe_autocast", _fake_maybe_autocast)

    class _NoTensors:
        pass

    requested = utils.ResolvedPrecision(
        requested="bf16",
        normalized="bf16",
        effective="bf16",
        compute_dtype=torch.bfloat16,
    )

    result = utils.maybe_autocast_for_module(torch.device("cuda"), requested, _NoTensors())

    assert result == "sentinel"
    assert calls == [(torch.device("cuda"), requested)]
