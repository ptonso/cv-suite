from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from cvsuite.common.core import FMRequest
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core.paths import hub_dir as resolve_hub_dir
from cvsuite.common.fm.core.paths import providers_dir as resolve_providers_dir
from cvsuite.common.fm.core.paths import cvsuite_home
from cvsuite.common.fm.core.sam3_patch import SAM3_DEPRECATED_TIMM_IMPORT, SAM3_TIMM_IMPORT_PATCH_TARGETS
from cvsuite.common.fm.providers.registry import LOCAL_FAMILY_NAMES, ProviderSpec, resolve_provider_spec


def get_fm_root() -> Path:
    """Directory of the installed fm package -- code only, never written to."""

    return Path(__file__).resolve().parent.parent


def get_fm_cache_root() -> Path:
    return cvsuite_home()


def get_src_root() -> Path:
    """Import root for the subprocess PYTHONPATH, valid for both source and wheel installs."""

    import cvsuite

    return Path(cvsuite.__file__).resolve().parent.parent


def get_fm_runs_root() -> Path:
    return Path(tempfile.gettempdir()) / "cvsuite-fm-runs"


DEFAULT_CUDA_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
DEFAULT_CUDA_TORCH_VERSION = "2.7.1"
DEFAULT_CUDA_TORCHVISION_VERSION = "0.22.1"
DEFAULT_CUDA_TORCHAUDIO_VERSION = "2.7.1"
CUDA_SHARED_VENV_MODEL_NAMES = frozenset(
    {
        "blip",
        "cogvlm",
        "dim_edit",
        "flux",
        "flux2_klein",
        "instruct_pix2pix",
        "internvl",
        "locate_anything",
        "llmdet",
        "llava",
        "minicpm_v",
        "ovis_u1_3b",
        "paligemma",
        "qwen",
        "qwen_image",
        "qwen_image_edit",
        "rex_omni",
        "sana",
        "stable_diffusion",
        "step1x_edit",
    }
)
HF_MANAGED_AUTO_MODEL_NAMES = frozenset(
    {
        "blip",
        "cogvlm",
        "flux",
        "flux2_klein",
        "instruct_pix2pix",
        "internvl",
        "locate_anything",
        "llava",
        "minicpm_v",
        "ovis_u1_3b",
        "paligemma",
        "qwen",
        "qwen_image",
        "qwen_image_edit",
        "rex_omni",
        "sam3",
        "sana",
        "stable_diffusion",
    }
)
HEALTHCHECK_IMPORTS_BY_MODEL_NAME = {
    "dim_edit": (
        "diffusers",
        "matplotlib",
        "mmcv",
        "transformers",
    ),
    "flux": (
        "accelerate",
        "diffusers",
        "transformers",
    ),
    "flux2_klein": (
        "accelerate",
        "diffusers",
        "transformers",
    ),
    "instruct_pix2pix": (
        "accelerate",
        "diffusers",
        "transformers",
    ),
    "internvl": ("einops", "timm"),
    "locate_anything": (
        "cv2",
        "decord",
        "lmdb",
        "peft",
        "transformers",
    ),
    "ovis_u1_3b": (
        "accelerate",
        "transformers",
    ),
    "qwen_image": (
        "accelerate",
        "diffusers",
        "transformers",
    ),
    "qwen_image_edit": (
        "accelerate",
        "diffusers",
        "transformers",
    ),
    "sana": (
        "accelerate",
        "diffusers",
        "transformers",
    ),
    "stable_diffusion": (
        "accelerate",
        "diffusers",
        "transformers",
    ),
    "step1x_edit": (
        "accelerate",
        "diffusers",
        "transformers",
    ),
}
HEALTHCHECK_REPO_CHECKOUTS_BY_MODEL_NAME = {
    "dim_edit": ("DIM",),
}
HEALTHCHECK_ERROR_MARKERS = (
    "ModuleNotFoundError",
    "ImportError",
    "AssertionError",
    "RuntimeError",
    "ValueError",
    "Fatal Python error",
    "Segmentation fault",
    "Aborted",
)


def _resolve_provider_spec_any_family(provider_name: str, *, preferred_family: str = "") -> ProviderSpec:
    families: list[str] = []
    if preferred_family:
        families.append(preferred_family)
    families.extend(family for family in LOCAL_FAMILY_NAMES if family not in families)

    matches: list[ProviderSpec] = []
    for family in families:
        try:
            matches.append(resolve_provider_spec(provider_name, family=family))
        except ValueError:
            continue

    if not matches:
        raise ValueError(f"Unknown FM provider {provider_name!r}.")
    if preferred_family:
        return matches[0]
    if len(matches) > 1:
        families_csv = ", ".join(spec.family for spec in matches)
        raise ValueError(
            f"Provider {provider_name!r} is ambiguous without a provider family. Matching families: {families_csv}."
        )
    return matches[0]


@dataclass
class RunnerConfig:
    provider_name: str = ""
    provider_family: str = ""
    provider_class: str = ""
    provider_module: str = ""
    model_name: str = ""
    task: str = ""
    device: str = "auto"
    batch_size: int = 1
    prompt: str = ""
    precision: str = "fp32"
    max_gpu_memory: str | None = None
    config_path: Optional[Path] = None
    weights_dir: Optional[Path] = None
    model_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_name and self.model_name:
            self.provider_name = self.model_name
        if not self.model_name and self.provider_name:
            self.model_name = self.provider_name
        if not self.provider_name:
            raise ValueError("RunnerConfig requires a provider name.")

        spec = (
            resolve_provider_spec(self.provider_name, family=self.provider_family)
            if self.provider_family
            else _resolve_provider_spec_any_family(self.provider_name)
        )
        if not self.provider_family:
            self.provider_family = spec.family
        if not self.provider_class:
            self.provider_class = spec.provider_class
        if not self.provider_module:
            self.provider_module = spec.module

    @classmethod
    def from_dataset(cls, dataset: VisionDataset) -> "RunnerConfig":
        request = dataset.fm_request
        if request is None:
            raise ValueError("VisionDataset.fm_request must be populated before calling FMRunner.")
        provider_name = str(request.provider or "").strip()
        if not provider_name:
            raise ValueError("FM request is missing a provider name.")
        provider_family = infer_provider_family(dataset, request)
        spec = resolve_provider_spec(provider_name, family=provider_family)
        return cls(
            provider_name=provider_name,
            provider_family=provider_family,
            provider_class=spec.provider_class,
            provider_module=spec.module,
            model_name=provider_name,
            task=request.task,
            device=request.device,
            batch_size=request.batch_size,
            prompt=request.prompt,
            precision=request.precision,
            max_gpu_memory=request.max_gpu_memory,
            config_path=request.config_path,
            weights_dir=request.weights_dir,
            model_args=dict(request.model_args),
        )


def infer_provider_family(dataset: VisionDataset, request: FMRequest) -> str:
    family = str((request.meta or {}).get("provider_family") or "").strip()
    if family:
        return family
    task = str(request.task or "").strip().lower()
    if task == "gen":
        mode = str((dataset.meta or {}).get("gen_mode") or "create").strip().lower()
        if mode in {"create", "edit"}:
            return mode
    if task == "classify":
        return "classify"
    return task


class FMRunner:
    """Orchestrates FM execution in an isolated venv, round-tripping VisionDataset via JSON."""

    def __init__(self, cfg: RunnerConfig) -> None:
        self.cfg = cfg
        self.caller_cwd = Path.cwd()
        self.fm_root = get_fm_root()
        self.src_root = get_src_root()
        self.providers_dir = self.fm_root / "providers"
        self.setup_dir = self.providers_dir / "setup_venv"
        self.cache_root = get_fm_cache_root()
        self.hub_dir = resolve_hub_dir()
        self.model_cache = resolve_providers_dir() / cfg.provider_name
        self.venv_dir = self.model_cache / "venv"
        self.weights_dir = cfg.weights_dir or (self.model_cache / "weights")
        self.pkgs_dir = self.model_cache / "pkgs"
        self.stage_dir = self.model_cache / "stage"
        self._last_venv_healthcheck_summary: str | None = None

    @classmethod
    def from_dataset(cls, dataset: VisionDataset) -> "FMRunner":
        return cls(RunnerConfig.from_dataset(dataset))

    def _log(self, message: str) -> None:
        print(f"[fm:{self.cfg.provider_name}] {message}")

    def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
        self._log("preparing cache directories")
        self._prepare_cache()
        self._reset_stage_dir()
        python_bin = self._ensure_runtime_python()
        with tempfile.TemporaryDirectory(prefix=f"fm-{self.cfg.provider_name}-") as tmpdir:
            tmp_dir = Path(tmpdir)
            in_json = tmp_dir / "in.json"
            out_json = tmp_dir / "out.json"
            self._log(f"serializing dataset to {in_json}")
            dataset.to_json(in_json)
            self._log("invoking model entrypoint")
            try:
                self._invoke_model(python_bin, in_json, out_json, no_resume=no_resume)
            finally:
                self._reset_stage_dir()
            if not out_json.exists():
                raise RuntimeError(f"Provider {self.cfg.provider_name} did not produce {out_json}")
            self._log(f"loading model output from {out_json}")
            return VisionDataset.from_json(out_json)

    def _prepare_cache(self) -> None:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.hub_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    def _reset_stage_dir(self) -> None:
        if self.stage_dir.is_symlink() or self.stage_dir.is_file():
            self.stage_dir.unlink()
        elif self.stage_dir.exists():
            shutil.rmtree(self.stage_dir, ignore_errors=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    def _venv_python(self) -> Path:
        return self.venv_dir / "bin" / "python"

    def _provider_module_name(self) -> str:
        return self.cfg.provider_module

    def _model_module_name(self) -> str:
        return self._provider_module_name()

    def _prefers_cuda_capable_venv(self) -> bool:
        device_pref = str(self.cfg.device or "auto").strip().lower()
        if device_pref in {"cuda", "gpu"}:
            return True
        return self.cfg.provider_name in CUDA_SHARED_VENV_MODEL_NAMES

    def _venv_healthcheck_code(self) -> Optional[str]:
        module_import = f"import importlib; importlib.import_module({self._provider_module_name()!r})"
        extra_imports = "".join(f"import {module}; " for module in HEALTHCHECK_IMPORTS_BY_MODEL_NAME.get(self.cfg.provider_name, ()))
        repo_assertions = "".join(
            (
                "import pathlib; "
                f"assert (pathlib.Path({str(self.pkgs_dir)!r}) / {repo_name!r}).exists(), "
                f"(pathlib.Path({str(self.pkgs_dir)!r}) / {repo_name!r}); "
            )
            for repo_name in HEALTHCHECK_REPO_CHECKOUTS_BY_MODEL_NAME.get(self.cfg.provider_name, ())
        )
        if self.cfg.provider_name == "sam3":
            sam3_timm_guard = (
                "import importlib.util, pathlib; "
                "sam3_spec = importlib.util.find_spec('sam3'); "
                "assert sam3_spec is not None and sam3_spec.origin is not None, sam3_spec; "
                "sam3_pkg_root = pathlib.Path(sam3_spec.origin).resolve().parent; "
                f"sam3_targets = {SAM3_TIMM_IMPORT_PATCH_TARGETS!r}; "
                f"sam3_needle = {SAM3_DEPRECATED_TIMM_IMPORT!r}; "
                "sam3_stale = [rel for rel in sam3_targets if sam3_needle in (sam3_pkg_root / rel).read_text(encoding='utf-8')]; "
                "assert not sam3_stale, ('deprecated timm.models.layers imports', sam3_stale); "
            )
            return (
                "import cv2, decord, einops, importlib, numpy, torch; from PIL import Image; "
                f"{sam3_timm_guard}"
                "from sam3.model_builder import build_sam3_image_model; "
                "from sam3.model.sam3_image_processor import Sam3Processor; "
                "assert torch.version.cuda is not None, torch.__version__; "
                "arch_flags = getattr(torch._C, '_cuda_getArchFlags', lambda: '')(); "
                "arch_list = {flag for flag in arch_flags.split() if flag.startswith('sm_')}; "
                "arch_list = arch_list or set(torch.cuda.get_arch_list()); "
                "has_cuda = torch.cuda.is_available(); "
                "cap = torch.cuda.get_device_capability(0) if has_cuda else None; "
                "sm = f'sm_{cap[0]}{cap[1]}' if cap is not None else None; "
                "assert (sm is None or sm in arch_list), (sm, sorted(arch_list), torch.__version__, torch.version.cuda); "
                f"{repo_assertions}{module_import}"
            )
        if self.cfg.provider_name == "minicpm_v":
            return (
                "import torch, transformers; "
                "major = int(transformers.__version__.split('.', 1)[0]); "
                "assert major < 5, transformers.__version__; "
                "assert torch.version.cuda is not None, (torch.__version__, torch.version.cuda); "
                f"{repo_assertions}{module_import}"
            )
        if self.cfg.provider_name == "internvl":
            return (
                "import torch, transformers; "
                "major = int(transformers.__version__.split('.', 1)[0]); "
                "assert major < 5, transformers.__version__; "
                "assert torch.version.cuda is not None, (torch.__version__, torch.version.cuda); "
                f"{extra_imports}{repo_assertions}{module_import}"
            )
        if self._prefers_cuda_capable_venv():
            return (
                "import torch, torchgen.model; "
                "assert torch.version.cuda is not None, (torch.__version__, torch.version.cuda); "
                f"{extra_imports}{repo_assertions}{module_import}"
            )
        return f"{extra_imports}{repo_assertions}{module_import}"

    def _summarize_healthcheck_failure(self, result: subprocess.CompletedProcess[str]) -> str:
        lines = [line.strip() for line in (result.stderr or result.stdout or "").splitlines() if line.strip()]
        if not lines:
            return f"exit={result.returncode}"
        for line in reversed(lines):
            if any(marker in line for marker in HEALTHCHECK_ERROR_MARKERS):
                return line
        return lines[-1]

    def _venv_is_ready(self, python_bin: Path) -> bool:
        if not python_bin.exists():
            self._last_venv_healthcheck_summary = f"missing python executable: {python_bin}"
            return False

        probe = self._venv_healthcheck_code()
        if probe is None:
            self._last_venv_healthcheck_summary = None
            return True

        result = subprocess.run(
            [str(python_bin), "-c", probe],
            check=False,
            cwd=self.fm_root,
            capture_output=True,
            text=True,
            env=self._runtime_env(),
        )
        if result.returncode == 0:
            self._last_venv_healthcheck_summary = None
            return True

        summary = self._summarize_healthcheck_failure(result)
        self._last_venv_healthcheck_summary = summary
        self._log(f"existing venv failed health check; rebuilding ({summary})")
        return False

    def _setup_env(self) -> dict[str, str]:
        env = os.environ.copy()
        # Provider setup scripts create their venv with `${PYTHON_BIN:-python3}`.
        # Pin it to the interpreter running cvsuite so a missing `python` shim
        # (common on Debian/Ubuntu) can't break the bootstrap.
        env.setdefault("PYTHON_BIN", sys.executable)
        if self._prefers_cuda_capable_venv():
            env.setdefault("VT_TORCH_INDEX_URL", DEFAULT_CUDA_TORCH_INDEX_URL)
            env.setdefault("VT_TORCH_VERSION", DEFAULT_CUDA_TORCH_VERSION)
            env.setdefault("VT_TORCHVISION_VERSION", DEFAULT_CUDA_TORCHVISION_VERSION)
            env.setdefault("VT_TORCHAUDIO_VERSION", DEFAULT_CUDA_TORCHAUDIO_VERSION)
        return env

    @staticmethod
    def _is_site_package_path(entry: str | Path) -> bool:
        normalized = str(entry).replace("\\", "/")
        return "/site-packages" in normalized or "/dist-packages" in normalized

    @classmethod
    def _filtered_existing_pythonpath(cls, existing_path: str) -> list[str]:
        entries: list[str] = []
        for raw_entry in str(existing_path or "").split(os.pathsep):
            entry = raw_entry.strip()
            if not entry:
                continue
            if cls._is_site_package_path(entry):
                continue
            entries.append(entry)
        return entries

    def _runtime_env(self, *, hide_gpu: bool = False) -> dict[str, str]:
        env = os.environ.copy()
        existing_path = env.get("PYTHONPATH", "")
        python_paths = [str(self.src_root)]
        python_paths.extend(self._filtered_existing_pythonpath(existing_path))
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env["FM_CACHE_DIR"] = str(self.cache_root)
        env["FM_HUB_DIR"] = str(self.hub_dir)
        env["FM_MODEL_CACHE"] = str(self.model_cache)
        env["FM_RUNS_ROOT"] = str(get_fm_runs_root())
        env["FM_WEIGHTS_DIR"] = str(self.weights_dir)
        env["FM_STAGE_DIR"] = str(self.stage_dir)
        env["FM_CALLER_CWD"] = str(self.caller_cwd)
        env.setdefault("HF_MODULES_CACHE", str(self.stage_dir / "hf_modules"))
        if self._should_enable_allocator_hint():
            allocator_env_set = bool(env.get("PYTORCH_CUDA_ALLOC_CONF") or env.get("PYTORCH_ALLOC_CONF"))
            env["VT_AUTO_ALLOCATOR_HINT_APPLIED"] = "0" if allocator_env_set else "1"
            if not allocator_env_set:
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if self._should_disable_cuda_memory_caching():
            env.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
        if hide_gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def _should_enable_allocator_hint(self) -> bool:
        device_pref = str(self.cfg.device or "auto").strip().lower()
        if device_pref != "auto":
            return False
        if str(self.cfg.precision or "").strip().lower() == "nf4":
            return False
        return self.cfg.provider_name in HF_MANAGED_AUTO_MODEL_NAMES

    def _should_disable_cuda_memory_caching(self) -> bool:
        device_pref = str(self.cfg.device or "auto").strip().lower()
        if device_pref == "cpu":
            return False
        return self.cfg.provider_name == "dim_edit"

    def _ensure_venv(self) -> Path:
        python_bin = self._venv_python()
        if self._venv_is_ready(python_bin):
            self._log(f"venv ready at {python_bin}")
            return python_bin
        self._log("bootstrapping environment and weights cache")
        os.makedirs(self.weights_dir, exist_ok=True)
        script = self.setup_dir / f"{self.cfg.provider_name}.sh"
        if not script.exists():
            raise FileNotFoundError(f"Missing venv setup script: {script}")
        if self.venv_dir.is_symlink() or self.venv_dir.is_file():
            self._log(f"clearing stale venv path {self.venv_dir}")
            self.venv_dir.unlink()
        elif self.venv_dir.exists():
            self._log(f"clearing stale venv directory {self.venv_dir}")
            shutil.rmtree(self.venv_dir)
        setup_env = self._setup_env()
        torch_index_url = setup_env.get("VT_TORCH_INDEX_URL")
        if torch_index_url and self._prefers_cuda_capable_venv():
            self._log(f"bootstrapping CUDA-capable torch from {torch_index_url}")
        self._log(f"running setup script {script}")
        cmd = ["bash", str(script), str(self.venv_dir)]
        subprocess.run(cmd, check=True, cwd=self.fm_root, env=setup_env)
        if not self._venv_is_ready(python_bin):
            detail = f" ({self._last_venv_healthcheck_summary})" if self._last_venv_healthcheck_summary else ""
            raise RuntimeError(
                f"Venv bootstrap for {self.cfg.provider_name} did not produce a healthy venv at {python_bin}{detail}"
            )
        self._log(f"venv created at {python_bin}")
        return python_bin

    def _ensure_runtime_python(self) -> Path:
        if self.cfg.provider_class == "api":
            return Path(sys.executable)
        return self._ensure_venv()

    def _invoke_model(self, python_bin: Path, in_json: Path, out_json: Path, *, no_resume: bool = False) -> None:
        model_module = self._provider_module_name()
        try:
            spec = importlib.util.find_spec(model_module)
        except (ImportError, AttributeError, ValueError) as exc:
            raise FileNotFoundError(f"Missing provider entrypoint module: {model_module}") from exc
        if spec is None:
            raise FileNotFoundError(f"Missing provider entrypoint module: {model_module}")

        cmd = [
            str(python_bin),
            "-m",
            model_module,
            "--input",
            str(in_json),
            "--output",
            str(out_json),
        ]
        if no_resume:
            cmd.append("--no-resume")

        env = self._runtime_env(hide_gpu=str(self.cfg.device).lower() == "cpu")

        self._log(f"spawning provider process: {model_module}")
        subprocess.run(cmd, check=True, cwd=self.fm_root, env=env)
