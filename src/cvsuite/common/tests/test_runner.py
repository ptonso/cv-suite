from __future__ import annotations

import importlib.util
import re
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core.sam3_patch import (
    SAM3_TIMM_IMPORT_PATCH_TARGETS,
    patch_sam3_timm_imports,
)
from cvsuite.common.fm.core.runner import (
    CUDA_SHARED_VENV_MODEL_NAMES,
    DEFAULT_CUDA_TORCH_INDEX_URL,
    DEFAULT_CUDA_TORCHAUDIO_VERSION,
    DEFAULT_CUDA_TORCH_VERSION,
    DEFAULT_CUDA_TORCHVISION_VERSION,
    FMRunner,
    HF_MANAGED_AUTO_MODEL_NAMES,
    RunnerConfig,
    get_fm_runs_root,
)
from cvsuite.common.fm.providers.registry import resolve_model_module
from cvsuite.common.fm.core import runner as runner_module


def test_fm_runner_sets_pythonpath_to_src(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def _fake_run(cmd, check, cwd, env=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)

    runner = FMRunner(RunnerConfig(model_name="clip"))
    runner._invoke_model(tmp_path / "python", tmp_path / "in.json", tmp_path / "out.json")

    env = captured["env"]
    assert env is not None
    pythonpath = env["PYTHONPATH"].split(os.pathsep)
    assert str(runner.src_root) in pythonpath
    assert env["FM_RUNS_ROOT"] == str(get_fm_runs_root())
    assert env["FM_HUB_DIR"] == str(runner.hub_dir)
    assert env["FM_STAGE_DIR"] == str(runner.stage_dir)
    assert captured["cmd"] == [
        str(tmp_path / "python"),
        "-m",
        "cvsuite.common.fm.providers.classify.clip",
        "--input",
        str(tmp_path / "in.json"),
        "--output",
        str(tmp_path / "out.json"),
    ]


def test_fm_runner_adds_no_resume_flag(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def _fake_run(cmd, check, cwd, env=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)

    runner = FMRunner(RunnerConfig(model_name="clip"))
    runner._invoke_model(tmp_path / "python", tmp_path / "in.json", tmp_path / "out.json", no_resume=True)

    assert captured["cmd"][-1] == "--no-resume"


def test_fm_runner_auto_sets_allocator_hint_for_supported_hf_auto(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def _fake_run(cmd, check, cwd, env=None):
        captured["env"] = env

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)

    runner = FMRunner(RunnerConfig(model_name="qwen", device="auto"))
    runner._invoke_model(tmp_path / "python", tmp_path / "in.json", tmp_path / "out.json")

    assert captured["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert captured["env"]["VT_AUTO_ALLOCATOR_HINT_APPLIED"] == "1"


def test_fm_runner_preserves_existing_allocator_hint_when_gpu_cap_is_requested(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def _fake_run(cmd, check, cwd, env=None):
        captured["env"] = env

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

    runner = FMRunner(RunnerConfig(model_name="qwen", max_gpu_memory="14GiB"))
    runner._invoke_model(tmp_path / "python", tmp_path / "in.json", tmp_path / "out.json")

    assert captured["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync"
    assert captured["env"]["VT_AUTO_ALLOCATOR_HINT_APPLIED"] == "0"


def test_fm_runner_does_not_set_allocator_hint_for_unsupported_auto_model(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def _fake_run(cmd, check, cwd, env=None):
        captured["env"] = env

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)

    runner = FMRunner(RunnerConfig(model_name="clip", device="auto"))
    runner._invoke_model(tmp_path / "python", tmp_path / "in.json", tmp_path / "out.json")

    assert "VT_AUTO_ALLOCATOR_HINT_APPLIED" not in captured["env"]
    assert "PYTORCH_CUDA_ALLOC_CONF" not in captured["env"]


def test_fm_runner_cleans_stage_dir_after_run(monkeypatch, tmp_path: Path) -> None:
    runner = FMRunner(RunnerConfig(model_name="clip"))
    runner.cache_root = tmp_path / "cache"
    runner.model_cache = runner.cache_root / "clip"
    runner.venv_dir = runner.model_cache / "venv"
    runner.weights_dir = runner.model_cache / "weights"
    runner.stage_dir = runner.model_cache / "stage"

    dataset = VisionDataset(records=[])

    monkeypatch.setattr(runner, "_ensure_venv", lambda: tmp_path / "python")

    def _fake_invoke_model(_python_bin: Path, _in_json: Path, out_json: Path, *, no_resume: bool = False) -> None:
        stray = runner.stage_dir / "offload" / "stale.safetensors"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_text("temporary", encoding="utf-8")
        dataset.to_json(out_json)

    monkeypatch.setattr(runner, "_invoke_model", _fake_invoke_model)

    result = runner.run(dataset)

    assert isinstance(result, VisionDataset)
    assert runner.stage_dir.is_dir()
    assert list(runner.stage_dir.iterdir()) == []


def test_fm_runner_hides_gpu_for_explicit_cpu_calls(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def _fake_run(cmd, check, cwd, env=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)

    runner = FMRunner(RunnerConfig(model_name="paligemma", device="cpu"))
    runner._invoke_model(tmp_path / "python", tmp_path / "in.json", tmp_path / "out.json")

    env = captured["env"]
    assert env is not None
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_fm_runner_healthcheck_imports_target_module(monkeypatch, tmp_path: Path) -> None:
    runner = FMRunner(RunnerConfig(model_name="blip"))
    runner.fm_root = tmp_path / "fm"
    runner.venv_dir = tmp_path / "cache" / "blip" / "venv"

    python_bin = runner.venv_dir / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")

    captured = {}

    def _fake_run(cmd, check, cwd, env=None, capture_output=False, text=False):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)

    assert runner._venv_is_ready(python_bin) is True
    assert captured["cmd"] == [str(python_bin), "-c", runner._venv_healthcheck_code()]
    assert "torchgen.model" in runner._venv_healthcheck_code()
    assert "cvsuite.common.fm.providers.vlm.blip" in runner._venv_healthcheck_code()
    assert captured["env"]["PYTHONPATH"].split(os.pathsep)[:1] == [str(runner.src_root)]


def test_fm_runner_filters_site_packages_from_inherited_pythonpath(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def _fake_run(cmd, check, cwd, env=None):
        captured["env"] = env

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            [
                "/tmp/custom-src",
                "/opt/example/venv/lib/python3.12/site-packages",
                "/usr/lib/python3/dist-packages",
            ]
        ),
    )

    runner = FMRunner(RunnerConfig(model_name="clip"))
    runner._invoke_model(tmp_path / "python", tmp_path / "in.json", tmp_path / "out.json")

    pythonpath = captured["env"]["PYTHONPATH"].split(os.pathsep)
    assert "/tmp/custom-src" in pythonpath
    assert not any("site-packages" in entry for entry in pythonpath)
    assert not any("dist-packages" in entry for entry in pythonpath)


def test_fm_runner_healthcheck_includes_internvl_remote_code_dependencies() -> None:
    runner = FMRunner(RunnerConfig(model_name="internvl"))

    probe = runner._venv_healthcheck_code()

    assert probe is not None
    assert "import einops;" in probe
    assert "import timm;" in probe
    assert "cvsuite.common.fm.providers.vlm.internvl" in probe


def test_fm_runner_healthcheck_includes_gen_diffusers_dependencies() -> None:
    runner = FMRunner(RunnerConfig(model_name="stable_diffusion"))

    probe = runner._venv_healthcheck_code()

    assert probe is not None
    assert "import accelerate;" in probe
    assert "import diffusers;" in probe
    assert "import transformers;" in probe
    assert "cvsuite.common.fm.providers.create.stable_diffusion" in probe


def test_fm_runner_healthcheck_checks_dim_repo_checkout() -> None:
    runner = FMRunner(RunnerConfig(model_name="dim_edit"))

    probe = runner._venv_healthcheck_code()

    assert probe is not None
    assert "import diffusers;" in probe
    assert "import matplotlib;" in probe
    assert "import mmcv;" in probe
    assert f"{runner.pkgs_dir}') / 'DIM'" in probe
    assert "cvsuite.common.fm.providers.edit.dim_edit" in probe


def test_fm_runner_resolves_model_module_via_registry() -> None:
    runner = FMRunner(RunnerConfig(model_name="clip"))

    assert runner._model_module_name() == resolve_model_module("clip")

def test_fm_runner_healthcheck_checks_sam3_for_deprecated_timm_imports() -> None:
    runner = FMRunner(RunnerConfig(model_name="sam3"))

    probe = runner._venv_healthcheck_code()

    assert probe is not None
    assert "deprecated timm.models.layers imports" in probe
    assert "importlib.util.find_spec('sam3')" in probe
    for rel_path in SAM3_TIMM_IMPORT_PATCH_TARGETS:
        assert rel_path in probe


def test_patch_sam3_timm_imports_rewrites_targets_and_is_idempotent(tmp_path: Path) -> None:
    sam3_root = tmp_path / "sam3"
    (sam3_root / "model").mkdir(parents=True)
    (sam3_root / "model" / "memory.py").write_text(
        "try:\n"
        "    from timm.layers import DropPath\n"
        "except ModuleNotFoundError:\n"
        "    # compatibility for older timm versions\n"
        "    from timm.models.layers import DropPath\n",
        encoding="utf-8",
    )
    (sam3_root / "model" / "video_tracking_multiplex.py").write_text(
        "from timm.models.layers import trunc_normal_\n",
        encoding="utf-8",
    )
    (sam3_root / "model" / "vitdet.py").write_text(
        "try:\n"
        "    from timm.layers import DropPath, trunc_normal_\n"
        "except ModuleNotFoundError:\n"
        "    # compatibility for older timm versions\n"
        "    from timm.models.layers import DropPath, trunc_normal_\n",
        encoding="utf-8",
    )
    (sam3_root / "model" / "sam3_tracker_base.py").write_text(
        "try:\n"
        "    from timm.layers import trunc_normal_\n"
        "except ModuleNotFoundError:\n"
        "    # compatibility for older timm versions\n"
        "    from timm.models.layers import trunc_normal_\n",
        encoding="utf-8",
    )

    modified = patch_sam3_timm_imports(sam3_root)

    assert [str(path) for path in modified] == list(SAM3_TIMM_IMPORT_PATCH_TARGETS)
    for rel_path in SAM3_TIMM_IMPORT_PATCH_TARGETS:
        text = (sam3_root / rel_path).read_text(encoding="utf-8")
        assert "timm.models.layers" not in text
        assert "from timm.layers import" in text

    assert patch_sam3_timm_imports(sam3_root) == []


def test_patch_sam3_timm_imports_accepts_repo_root_with_nested_package(tmp_path: Path) -> None:
    repo_root = tmp_path / "sam3-repo"
    package_root = repo_root / "sam3"
    (package_root / "model").mkdir(parents=True)
    (package_root / "model" / "memory.py").write_text(
        "try:\n"
        "    from timm.layers import DropPath\n"
        "except ModuleNotFoundError:\n"
        "    # compatibility for older timm versions\n"
        "    from timm.models.layers import DropPath\n",
        encoding="utf-8",
    )
    (package_root / "model" / "video_tracking_multiplex.py").write_text(
        "from timm.models.layers import trunc_normal_\n",
        encoding="utf-8",
    )
    (package_root / "model" / "vitdet.py").write_text(
        "try:\n"
        "    from timm.layers import DropPath, trunc_normal_\n"
        "except ModuleNotFoundError:\n"
        "    # compatibility for older timm versions\n"
        "    from timm.models.layers import DropPath, trunc_normal_\n",
        encoding="utf-8",
    )
    (package_root / "model" / "sam3_tracker_base.py").write_text(
        "try:\n"
        "    from timm.layers import trunc_normal_\n"
        "except ModuleNotFoundError:\n"
        "    # compatibility for older timm versions\n"
        "    from timm.models.layers import trunc_normal_\n",
        encoding="utf-8",
    )

    modified = patch_sam3_timm_imports(repo_root)

    assert [str(path) for path in modified] == list(SAM3_TIMM_IMPORT_PATCH_TARGETS)
    for rel_path in SAM3_TIMM_IMPORT_PATCH_TARGETS:
        text = (package_root / rel_path).read_text(encoding="utf-8")
        assert "timm.models.layers" not in text


def test_fm_runner_rebuilds_sam3_when_existing_venv_is_unhealthy(monkeypatch, tmp_path: Path) -> None:
    runner = FMRunner(RunnerConfig(model_name="sam3"))
    runner.fm_root = tmp_path / "fm"
    runner.setup_dir = runner.fm_root / "models" / "setup_venv"
    runner.venv_dir = tmp_path / "cache" / "sam3" / "venv"
    runner.weights_dir = tmp_path / "cache" / "sam3" / "weights"

    python_bin = runner.venv_dir / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")
    script = runner.setup_dir / "sam3.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\n")

    probe_calls = 0
    setup_calls = 0
    removed = []

    monkeypatch.setattr("cvsuite.common.fm.core.runner.shutil.rmtree", lambda path: removed.append(path))

    def _fake_run(cmd, check, cwd, env=None, capture_output=False, text=False):
        nonlocal probe_calls, setup_calls
        if cmd[:2] == [str(python_bin), "-c"]:
            probe_calls += 1
            return SimpleNamespace(
                returncode=1 if probe_calls == 1 else 0,
                stderr="ModuleNotFoundError: No module named 'sam3'" if probe_calls == 1 else "",
                stdout="",
            )
        if cmd == ["bash", str(script), str(runner.venv_dir)]:
            setup_calls += 1
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)

    assert runner._ensure_venv() == python_bin
    assert probe_calls == 2
    assert setup_calls == 1
    assert removed == [runner.venv_dir]


def test_fm_runner_rebuilds_sam3_when_timm_import_patch_is_missing(monkeypatch, tmp_path: Path) -> None:
    runner = FMRunner(RunnerConfig(model_name="sam3"))
    runner.fm_root = tmp_path / "fm"
    runner.setup_dir = runner.fm_root / "models" / "setup_venv"
    runner.venv_dir = tmp_path / "cache" / "sam3" / "venv"
    runner.weights_dir = tmp_path / "cache" / "sam3" / "weights"

    python_bin = runner.venv_dir / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")
    script = runner.setup_dir / "sam3.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\n")

    probe_calls = 0
    setup_calls = 0
    removed = []

    monkeypatch.setattr("cvsuite.common.fm.core.runner.shutil.rmtree", lambda path: removed.append(path))

    def _fake_run(cmd, check, cwd, env=None, capture_output=False, text=False):
        nonlocal probe_calls, setup_calls
        if cmd[:2] == [str(python_bin), "-c"]:
            probe_calls += 1
            return SimpleNamespace(
                returncode=1 if probe_calls == 1 else 0,
                stderr="AssertionError: ('deprecated timm.models.layers imports', ['model/vitdet.py'])"
                if probe_calls == 1
                else "",
                stdout="",
            )
        if cmd == ["bash", str(script), str(runner.venv_dir)]:
            setup_calls += 1
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)

    assert runner._ensure_venv() == python_bin
    assert probe_calls == 2
    assert setup_calls == 1
    assert removed == [runner.venv_dir]


def test_fm_runner_reuses_healthy_sam3_venv(monkeypatch, tmp_path: Path) -> None:
    runner = FMRunner(RunnerConfig(model_name="sam3"))
    runner.fm_root = tmp_path / "fm"
    runner.venv_dir = tmp_path / "cache" / "sam3" / "venv"

    python_bin = runner.venv_dir / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")

    calls = []

    def _fake_run(cmd, check, cwd, env=None, capture_output=False, text=False):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)

    assert runner._ensure_venv() == python_bin
    assert calls == [[str(python_bin), "-c", runner._venv_healthcheck_code()]]


def test_fm_runner_bootstrap_error_includes_last_healthcheck_summary(monkeypatch, tmp_path: Path) -> None:
    runner = FMRunner(RunnerConfig(model_name="qwen"))
    runner.fm_root = tmp_path / "fm"
    runner.setup_dir = runner.fm_root / "models" / "setup_venv"
    runner.venv_dir = tmp_path / "cache" / "qwen" / "venv"
    runner.weights_dir = tmp_path / "cache" / "qwen" / "weights"

    python_bin = runner.venv_dir / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")
    script = runner.setup_dir / "qwen.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\n")

    probe_calls = 0

    def _fake_run(cmd, check, cwd, env=None, capture_output=False, text=False):
        nonlocal probe_calls
        if cmd[:2] == [str(python_bin), "-c"]:
            probe_calls += 1
            return SimpleNamespace(
                returncode=1,
                stderr="ModuleNotFoundError: No module named 'torchgen.model'\nframe #39: _start + 0x25",
                stdout="",
            )
        if cmd == ["bash", str(script), str(runner.venv_dir)]:
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)
    monkeypatch.setattr("cvsuite.common.fm.core.runner.shutil.rmtree", lambda path: None)

    with pytest.raises(RuntimeError, match="torchgen.model"):
        runner._ensure_venv()

    assert probe_calls == 2


def test_fm_runner_setup_env_pins_python_bin_to_current_interpreter() -> None:
    import sys

    runner = FMRunner(RunnerConfig(model_name="clip"))
    assert runner._setup_env()["PYTHON_BIN"] == sys.executable


def test_setup_venv_scripts_use_shared_python_bin_idiom() -> None:
    import re

    setup_dir = Path(runner_module.__file__).resolve().parent.parent / "providers" / "setup_venv"
    scripts = sorted(setup_dir.glob("*.sh"))
    assert scripts, "no setup_venv scripts found"
    bare = re.compile(r"^\s*python3?\s+-m\s+venv\b", re.MULTILINE)
    for script in scripts:
        text = script.read_text()
        assert 'PYTHON_BIN="${PYTHON_BIN:-python3}"' in text, script.name
        assert '"${PYTHON_BIN}" -m venv --clear "${VENV_DIR}"' in text, script.name
        assert not bare.search(text), f"{script.name} still creates its venv with a bare interpreter"


def test_fm_runner_rebuilds_explicit_cuda_venv_when_existing_torch_is_cpu_only(monkeypatch, tmp_path: Path) -> None:
    runner = FMRunner(RunnerConfig(model_name="paligemma", device="cuda"))
    runner.fm_root = tmp_path / "fm"
    runner.setup_dir = runner.fm_root / "models" / "setup_venv"
    runner.venv_dir = tmp_path / "cache" / "paligemma" / "venv"
    runner.weights_dir = tmp_path / "cache" / "paligemma" / "weights"

    python_bin = runner.venv_dir / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")
    script = runner.setup_dir / "paligemma.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\n")

    probe_calls = 0
    setup_calls = 0
    removed = []

    monkeypatch.setattr("cvsuite.common.fm.core.runner.shutil.rmtree", lambda path: removed.append(path))

    def _fake_run(cmd, check, cwd, env=None, capture_output=False, text=False):
        nonlocal probe_calls, setup_calls
        if cmd[:2] == [str(python_bin), "-c"]:
            probe_calls += 1
            return SimpleNamespace(
                returncode=1 if probe_calls == 1 else 0,
                stderr="AssertionError: ('2.7.1', None)" if probe_calls == 1 else "",
                stdout="",
            )
        if cmd == ["bash", str(script), str(runner.venv_dir)]:
            setup_calls += 1
            assert env is not None
            assert env["VT_TORCH_INDEX_URL"] == DEFAULT_CUDA_TORCH_INDEX_URL
            assert env["VT_TORCH_VERSION"] == DEFAULT_CUDA_TORCH_VERSION
            assert env["VT_TORCHVISION_VERSION"] == DEFAULT_CUDA_TORCHVISION_VERSION
            assert env["VT_TORCHAUDIO_VERSION"] == DEFAULT_CUDA_TORCHAUDIO_VERSION
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)

    assert runner._ensure_venv() == python_bin
    assert probe_calls == 2
    assert setup_calls == 1
    assert removed == [runner.venv_dir]


def test_fm_runner_rebuilds_minicpm_venv_when_transformers_is_v5(monkeypatch, tmp_path: Path) -> None:
    runner = FMRunner(RunnerConfig(model_name="minicpm_v"))
    runner.fm_root = tmp_path / "fm"
    runner.setup_dir = runner.fm_root / "models" / "setup_venv"
    runner.venv_dir = tmp_path / "cache" / "minicpm_v" / "venv"
    runner.weights_dir = tmp_path / "cache" / "minicpm_v" / "weights"

    python_bin = runner.venv_dir / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")
    script = runner.setup_dir / "minicpm_v.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\n")

    probe_calls = 0
    setup_calls = 0
    removed = []

    monkeypatch.setattr("cvsuite.common.fm.core.runner.shutil.rmtree", lambda path: removed.append(path))

    def _fake_run(cmd, check, cwd, env=None, capture_output=False, text=False):
        nonlocal probe_calls, setup_calls
        if cmd[:2] == [str(python_bin), "-c"]:
            probe_calls += 1
            return SimpleNamespace(
                returncode=1 if probe_calls == 1 else 0,
                stderr="AssertionError: 5.3.0" if probe_calls == 1 else "",
                stdout="",
            )
        if cmd == ["bash", str(script), str(runner.venv_dir)]:
            setup_calls += 1
            assert env is not None
            assert env["VT_TORCH_INDEX_URL"] == DEFAULT_CUDA_TORCH_INDEX_URL
            assert env["VT_TORCH_VERSION"] == DEFAULT_CUDA_TORCH_VERSION
            assert env["VT_TORCHVISION_VERSION"] == DEFAULT_CUDA_TORCHVISION_VERSION
            assert env["VT_TORCHAUDIO_VERSION"] == DEFAULT_CUDA_TORCHAUDIO_VERSION
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)

    assert runner._ensure_venv() == python_bin
    assert probe_calls == 2
    assert setup_calls == 1
    assert removed == [runner.venv_dir]


@pytest.mark.parametrize("model_id", ["blip", "cogvlm", "internvl", "llava", "minicpm_v", "paligemma", "qwen"])
@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_fm_runner_rebuilds_shared_vlm_venv_as_cuda_capable_for_non_cuda_requests(
    monkeypatch,
    tmp_path: Path,
    model_id: str,
    device: str,
) -> None:
    runner = FMRunner(RunnerConfig(model_name=model_id, device=device))
    runner.fm_root = tmp_path / "fm"
    runner.setup_dir = runner.fm_root / "models" / "setup_venv"
    runner.venv_dir = tmp_path / "cache" / model_id / "venv"
    runner.weights_dir = tmp_path / "cache" / model_id / "weights"

    python_bin = runner.venv_dir / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")
    script = runner.setup_dir / f"{model_id}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\n")

    probe_calls = 0
    setup_calls = 0
    removed = []

    monkeypatch.setattr("cvsuite.common.fm.core.runner.shutil.rmtree", lambda path: removed.append(path))

    def _fake_run(cmd, check, cwd, env=None, capture_output=False, text=False):
        nonlocal probe_calls, setup_calls
        if cmd[:2] == [str(python_bin), "-c"]:
            probe_calls += 1
            return SimpleNamespace(
                returncode=1 if probe_calls == 1 else 0,
                stderr="AssertionError: ('2.10.0+cpu', None)" if probe_calls == 1 else "",
                stdout="",
            )
        if cmd == ["bash", str(script), str(runner.venv_dir)]:
            setup_calls += 1
            assert env is not None
            assert env["VT_TORCH_INDEX_URL"] == DEFAULT_CUDA_TORCH_INDEX_URL
            assert env["VT_TORCH_VERSION"] == DEFAULT_CUDA_TORCH_VERSION
            assert env["VT_TORCHVISION_VERSION"] == DEFAULT_CUDA_TORCHVISION_VERSION
            assert env["VT_TORCHAUDIO_VERSION"] == DEFAULT_CUDA_TORCHAUDIO_VERSION
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)

    assert runner._ensure_venv() == python_bin
    assert probe_calls == 2
    assert setup_calls == 1
    assert removed == [runner.venv_dir]


@pytest.mark.parametrize(
    "model_id",
    ["stable_diffusion", "flux", "qwen_image", "sana", "qwen_image_edit", "flux2_klein", "step1x_edit", "dim_edit", "ovis_u1_3b"],
)
@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_fm_runner_rebuilds_shared_gen_venv_as_cuda_capable_for_non_cuda_requests(
    monkeypatch,
    tmp_path: Path,
    model_id: str,
    device: str,
) -> None:
    runner = FMRunner(RunnerConfig(model_name=model_id, device=device))
    runner.fm_root = tmp_path / "fm"
    runner.setup_dir = runner.fm_root / "models" / "setup_venv"
    runner.venv_dir = tmp_path / "cache" / model_id / "venv"
    runner.weights_dir = tmp_path / "cache" / model_id / "weights"

    python_bin = runner.venv_dir / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")
    script = runner.setup_dir / f"{model_id}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\n")

    if model_id == "dim_edit":
        (runner.venv_dir.parent / "pkgs" / "DIM").mkdir(parents=True, exist_ok=True)

    probe_calls = 0
    setup_calls = 0
    removed = []

    monkeypatch.setattr("cvsuite.common.fm.core.runner.shutil.rmtree", lambda path: removed.append(path))

    def _fake_run(cmd, check, cwd, env=None, capture_output=False, text=False):
        nonlocal probe_calls, setup_calls
        if cmd[:2] == [str(python_bin), "-c"]:
            probe_calls += 1
            return SimpleNamespace(
                returncode=1 if probe_calls == 1 else 0,
                stderr="AssertionError: ('2.10.0+cpu', None)" if probe_calls == 1 else "",
                stdout="",
            )
        if cmd == ["bash", str(script), str(runner.venv_dir)]:
            setup_calls += 1
            assert env is not None
            assert env["VT_TORCH_INDEX_URL"] == DEFAULT_CUDA_TORCH_INDEX_URL
            assert env["VT_TORCH_VERSION"] == DEFAULT_CUDA_TORCH_VERSION
            assert env["VT_TORCHVISION_VERSION"] == DEFAULT_CUDA_TORCHVISION_VERSION
            assert env["VT_TORCHAUDIO_VERSION"] == DEFAULT_CUDA_TORCHAUDIO_VERSION
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    monkeypatch.setattr("cvsuite.common.fm.core.runner.subprocess.run", _fake_run)

    assert runner._ensure_venv() == python_bin
    assert probe_calls == 2
    assert setup_calls == 1
    assert removed == [runner.venv_dir]


def test_fm_runner_exposes_setup_scripts_for_new_gen_models() -> None:
    for model_id in (
        "stable_diffusion",
        "flux",
        "qwen_image",
        "sana",
        "qwen_image_edit",
        "flux2_klein",
        "step1x_edit",
        "dim_edit",
        "ovis_u1_3b",
    ):
        runner = FMRunner(RunnerConfig(model_name=model_id))
        assert (runner.setup_dir / f"{model_id}.sh").exists()


def test_fm_runner_exposes_llmdet_setup_script() -> None:
    runner = FMRunner(RunnerConfig(model_name="llmdet"))
    assert (runner.setup_dir / "llmdet.sh").exists()


def test_fm_runner_exposes_locate_anything_setup_script() -> None:
    runner = FMRunner(RunnerConfig(model_name="locate_anything"))
    assert (runner.setup_dir / "locate_anything.sh").exists()


def test_fm_runner_healthcheck_imports_llmdet_provider() -> None:
    runner = FMRunner(RunnerConfig(model_name="llmdet"))
    probe = runner._venv_healthcheck_code()

    assert probe is not None
    assert "torchgen.model" in probe
    assert "cvsuite.common.fm.providers.ground.llmdet" in probe


def test_fm_runner_healthcheck_imports_locate_anything_provider() -> None:
    runner = FMRunner(RunnerConfig(model_name="locate_anything"))
    probe = runner._venv_healthcheck_code()

    assert probe is not None
    assert "torchgen.model" in probe
    assert "import cv2; import decord; import lmdb; import peft; import transformers;" in probe
    assert "cvsuite.common.fm.providers.ground.locate_anything" in probe


def test_fm_runner_tracks_new_gen_runner_capabilities() -> None:
    assert {
        "stable_diffusion",
        "flux",
        "qwen_image",
        "sana",
        "qwen_image_edit",
        "flux2_klein",
        "step1x_edit",
        "dim_edit",
        "ovis_u1_3b",
        "llmdet",
        "locate_anything",
    }.issubset(CUDA_SHARED_VENV_MODEL_NAMES)
    assert {
        "stable_diffusion",
        "flux",
        "qwen_image",
        "sana",
        "qwen_image_edit",
        "flux2_klein",
        "locate_anything",
    }.issubset(HF_MANAGED_AUTO_MODEL_NAMES)
    assert "step1x_edit" not in HF_MANAGED_AUTO_MODEL_NAMES
    assert "dim_edit" not in HF_MANAGED_AUTO_MODEL_NAMES
    assert "ovis_u1_3b" in HF_MANAGED_AUTO_MODEL_NAMES


def test_setup_scripts_place_repo_checkouts_beside_the_venv() -> None:
    """Every upstream checkout must land in `<provider>/pkgs`, which is what the runtime reads.

    `_ensure_venv` rmtree's the venv before rerunning setup, so a checkout inside it is
    re-cloned on every rebuild, and `load_repo_checkout` / `pkgs_dir` would not find it.
    """

    setup_dir = Path(runner_module.__file__).resolve().parent.parent / "providers" / "setup_venv"
    offenders = []
    for script in sorted(setup_dir.glob("*.sh")):
        for line in script.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "PKG_ROOT=" not in stripped:
                continue
            if "VENV_DIR" in stripped:
                offenders.append(f"{script.name}: {stripped}")
    assert offenders == []


def test_setup_scripts_pkgs_root_matches_runner_pkgs_dir() -> None:
    setup_dir = Path(runner_module.__file__).resolve().parent.parent / "providers" / "setup_venv"
    for script in sorted(setup_dir.glob("*.sh")):
        text = script.read_text(encoding="utf-8")
        if "PKG_ROOT=" not in text:
            continue
        runner = FMRunner(RunnerConfig(model_name=script.stem))
        assert runner.pkgs_dir == runner.venv_dir.parent / "pkgs"
        assert 'PKG_ROOT="${CACHE_ROOT}/pkgs"' in text


def test_setup_scripts_use_uppercase_shell_variables() -> None:
    """Setup scripts name their own variables in UPPER_SNAKE, matching VENV_DIR everywhere."""

    setup_dir = Path(runner_module.__file__).resolve().parent.parent / "providers" / "setup_venv"
    offenders = []
    for script in sorted(setup_dir.glob("*.sh")):
        for lineno, line in enumerate(script.read_text(encoding="utf-8").splitlines(), start=1):
            match = re.match(r"[ \t]*([a-z][A-Za-z0-9_]*)=", line)
            if match:
                offenders.append(f"{script.name}:{lineno} {match.group(1)}")
    assert offenders == []
