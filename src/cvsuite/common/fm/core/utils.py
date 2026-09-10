from __future__ import annotations

import inspect
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple, TypeVar

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight test envs
    class _MissingTorch:
        __version__ = "missing"
        version = SimpleNamespace(cuda=None)
        float32 = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"

        class dtype:
            pass

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def device_count() -> int:
                return 0

            @staticmethod
            def get_arch_list() -> list[str]:
                return []

        class device:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError(
                f"torch is required for FM runtime operation; attempted to access torch.{name} without torch installed."
            )

    torch = _MissingTorch()

from cvsuite.common.core import FMRequest, Record
from cvsuite.common.core import VisionDataset

from .paths import (
    drop_stale_share,
    hf_repo_cache_dir,
    iter_env_var_candidates,
    link_shared_repo,
    resolve_cached_hf_snapshot,
    shared_repo_target,
)

from .prompt_utils import (
    CLASS_TEMPLATE_TOKEN,
    DEFAULT_CLASS_TEMPLATE_PROMPT,
    build_prompt_map,
    parse_prompt_legacy,
    render_class_template,
)

T = TypeVar("T")

HF_TOKEN_KEYS = ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN")
HF_CANONICAL_TOKEN_KEYS = ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN")
_HF_TOKEN_VALIDITY_CACHE: Dict[str, bool] = {}
_PRECISION_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "nf4": "nf4",
}


@dataclass(frozen=True)
class ResolvedPrecision:
    requested: str
    normalized: str
    effective: str
    compute_dtype: torch.dtype
    quantization_mode: str | None = None
    quantization_config: Any | None = None
    warning: str | None = None

    @property
    def is_quantized(self) -> bool:
        return self.quantization_mode is not None

    @property
    def autocast_precision(self) -> str:
        if self.compute_dtype == torch.bfloat16:
            return "bf16"
        if self.compute_dtype == torch.float16:
            return "fp16"
        return "fp32"


@dataclass(frozen=True)
class HFModelSource:
    load_arg: str
    cache_dir: str | None
    revision: str | None
    local_files_only: bool
    snapshot_path: Path | None = None
    repo_cache_dir: Path | None = None


def _iter_hf_token_candidates() -> Iterable[Tuple[str, str]]:
    seen: set[str] = set()
    for _key, token, source in iter_env_var_candidates(HF_TOKEN_KEYS):
        seen.add(token)
        yield token, source
    token_path = Path.home() / ".huggingface" / "token"
    try:
        token = token_path.read_text(encoding="utf-8").strip() if token_path.exists() else ""
    except Exception:
        token = ""
    if token and token not in seen:
        yield token, str(token_path)


def _hf_api_whoami(token: str) -> Any:
    from huggingface_hub import HfApi

    return HfApi().whoami(token=token)


def _is_invalid_hf_token_error(exc: Exception) -> bool:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 401:
        return True
    message = str(exc).lower()
    return any(
        needle in message
        for needle in ("invalid user token", "token is invalid", "unauthorized", "401 client error")
    )


def _validate_hf_token(token: str, source: str) -> bool:
    cached = _HF_TOKEN_VALIDITY_CACHE.get(token)
    if cached is not None:
        return cached
    try:
        _hf_api_whoami(token)
    except Exception as exc:
        if _is_invalid_hf_token_error(exc):
            warnings.warn(
                (
                    "Hugging Face credentials from "
                    f"{source} appear invalid; falling back to anonymous downloads. "
                    "Update the token in .env or your environment if you need gated/private model access."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            _HF_TOKEN_VALIDITY_CACHE[token] = False
            return False
        return True
    _HF_TOKEN_VALIDITY_CACHE[token] = True
    return True


def _iter_exception_chain(exc: BaseException | None) -> Iterable[BaseException]:
    seen: set[int] = set()
    pending: List[BaseException] = []
    if exc is not None:
        pending.append(exc)
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        yield current
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)


def describe_hf_access_error(exc: BaseException, model_id: str) -> str | None:
    gated_repo = False
    unauthorized = False
    forbidden = False
    mentions_public_gated = False
    for current in _iter_exception_chain(exc):
        status_code = getattr(getattr(current, "response", None), "status_code", None)
        message = str(current).strip()
        lowered = message.lower()
        if status_code == 401 or "401 unauthorized" in lowered or "401 client error" in lowered:
            unauthorized = True
        if status_code == 403 or "403 forbidden" in lowered or "403 client error" in lowered:
            forbidden = True
        if "public gated repositories" in lowered:
            gated_repo = True
            mentions_public_gated = True
        elif "gated repo" in lowered or "gated repos" in lowered:
            gated_repo = True

    if gated_repo and unauthorized:
        return (
            f"Access to Hugging Face model {model_id!r} was denied. "
            "This repository is gated and the current process is not authenticated with an approved account. "
            "Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN (or log in with huggingface_hub) for an account that has access, "
            "or use a non-gated checkpoint."
        )
    if gated_repo:
        permissions_detail = (
            "Check token permissions, including public gated repositories access"
            if mentions_public_gated
            else "Check token permissions and confirm that the account can access this repository"
        )
        return (
            f"Access to Hugging Face model {model_id!r} was denied. "
            "This repository is gated and the current token/account does not have access. "
            f"{permissions_detail}, or use a non-gated checkpoint."
        )
    if forbidden:
        return (
            f"Access to Hugging Face model {model_id!r} was denied with HTTP 403. "
            "Check the token permissions and confirm that the account can access this repository."
        )
    return None


def resolve_existing_local_model_path(model_id: str, caller_cwd: Path | None = None) -> Path | None:
    raw = str(model_id or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    paths: list[Path] = []
    if candidate.is_absolute():
        paths.append(candidate)
    else:
        if caller_cwd is not None:
            paths.append((caller_cwd / candidate).resolve())
        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate not in paths:
            paths.append(cwd_candidate)
    for path in paths:
        if path.exists():
            return path
    return None


def build_hf_source_kwargs(source: HFModelSource, **kwargs: Any) -> dict[str, Any]:
    merged = dict(kwargs)
    if source.cache_dir is not None:
        merged.setdefault("cache_dir", source.cache_dir)
    if source.revision is not None:
        merged.setdefault("revision", source.revision)
    if source.local_files_only:
        merged.setdefault("local_files_only", True)
    return merged


def materialize_hf_model_source(
    model_id: str,
    *,
    hub_dir: Path | None,
    stage_dir: Path | None,
    caller_cwd: Path | None = None,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> HFModelSource:
    local_path = resolve_existing_local_model_path(model_id, caller_cwd)
    if local_path is not None:
        return HFModelSource(
            load_arg=str(local_path),
            cache_dir=None,
            revision=None,
            local_files_only=True,
            snapshot_path=local_path,
            repo_cache_dir=None,
        )

    cache_dir = str(hub_dir) if hub_dir is not None else None
    if hub_dir is None or "/" not in model_id:
        return HFModelSource(
            load_arg=model_id,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=False,
            snapshot_path=None,
            repo_cache_dir=None,
        )

    hub_dir.mkdir(parents=True, exist_ok=True)
    repo_cache = hf_repo_cache_dir(hub_dir, model_id)
    drop_stale_share(repo_cache, revision)
    snapshot_path, resolved_revision = resolve_cached_hf_snapshot(repo_cache, revision)
    if snapshot_path is None and not repo_cache.exists():
        shared = shared_repo_target(model_id, revision, require_patterns=allow_patterns)
        if shared is not None:
            repo_cache = link_shared_repo(hub_dir, model_id, shared)
            snapshot_path, resolved_revision = resolve_cached_hf_snapshot(repo_cache, revision)
    if stage_dir is None:
        return HFModelSource(
            load_arg=model_id,
            cache_dir=str(hub_dir),
            revision=resolved_revision or revision,
            local_files_only=snapshot_path is not None,
            snapshot_path=snapshot_path,
            repo_cache_dir=repo_cache,
        )

    stage_dir.mkdir(parents=True, exist_ok=True)
    if snapshot_path is None:
        stage_work_dir = Path(tempfile.mkdtemp(prefix="hf-", dir=str(stage_dir)))
        try:
            ensure_hf_caches(stage_work_dir)
            from huggingface_hub import snapshot_download

            token = str(os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or "").strip() or None
            try:
                downloaded_snapshot = Path(
                    snapshot_download(
                        repo_id=model_id,
                        cache_dir=str(stage_work_dir),
                        revision=revision,
                        token=token,
                        allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
                    )
                )
            except Exception as exc:
                access_error = describe_hf_access_error(exc, model_id)
                if access_error is not None:
                    raise RuntimeError(access_error) from exc
                raise
            staged_repo_cache = hf_repo_cache_dir(stage_work_dir, model_id)
            if not staged_repo_cache.exists():
                raise RuntimeError(
                    f"Hugging Face download for {model_id!r} did not populate the staged repo cache at {staged_repo_cache}."
                )
            if repo_cache.is_symlink():
                repo_cache.unlink()
            elif repo_cache.exists():
                shutil.rmtree(repo_cache)
            try:
                staged_repo_cache.replace(repo_cache)
            except OSError:
                shutil.move(str(staged_repo_cache), str(repo_cache))
            snapshot_path, resolved_revision = resolve_cached_hf_snapshot(repo_cache, revision)
            if snapshot_path is None and downloaded_snapshot.exists():
                snapshot_path = downloaded_snapshot
                if downloaded_snapshot.parent.name == "snapshots":
                    resolved_revision = downloaded_snapshot.name
        finally:
            shutil.rmtree(stage_work_dir, ignore_errors=True)
    ensure_hf_caches(stage_dir, hub=hub_dir)

    return HFModelSource(
        load_arg=model_id,
        cache_dir=str(hub_dir),
        revision=resolved_revision or revision,
        local_files_only=True,
        snapshot_path=snapshot_path,
        repo_cache_dir=repo_cache,
    )


def materialize_hf_file(
    repo_id: str,
    filename: str,
    *,
    hub_dir: Path | None,
    stage_dir: Path | None,
    caller_cwd: Path | None = None,
    revision: str | None = None,
) -> Path:
    source = materialize_hf_model_source(
        repo_id,
        hub_dir=hub_dir,
        stage_dir=stage_dir,
        caller_cwd=caller_cwd,
        revision=revision,
        allow_patterns=[filename],
    )
    if source.snapshot_path is None:
        raise RuntimeError(f"Could not resolve a cached snapshot for Hugging Face repo {repo_id!r}.")
    file_path = source.snapshot_path / filename
    if not file_path.exists():
        raise RuntimeError(f"Hugging Face repo {repo_id!r} is missing expected file {filename!r} in {source.snapshot_path}.")
    return file_path


def normalize_device_pref(pref: str = "auto") -> str:
    choice = str(pref or "auto").strip().lower()
    if choice == "gpu":
        return "cuda"
    return choice


def normalize_precision_name(precision: str = "fp32") -> str:
    choice = str(precision or "fp32").strip().lower()
    return _PRECISION_ALIASES.get(choice, choice or "fp32")


def optional_dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def refresh_transformers_bitsandbytes_availability() -> bool:
    """Refresh Transformers' cached bitsandbytes availability after an in-process install."""

    importlib.invalidate_caches()
    import_utils = sys.modules.get("transformers.utils.import_utils")
    if import_utils is None:
        return False

    refreshed = False
    available = optional_dependency_available("bitsandbytes")
    if hasattr(import_utils, "_bitsandbytes_available"):
        setattr(import_utils, "_bitsandbytes_available", available)
        refreshed = True

    checker = getattr(import_utils, "is_bitsandbytes_available", None)
    cache_clear = getattr(checker, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
        refreshed = True

    return refreshed


def patch_quantized_parameter_constructor_compat(param_cls: type[Any]) -> bool:
    compat_state = getattr(param_cls, "_cvsuite_unknown_kw_compat_patched", None)
    if compat_state is not None:
        return bool(compat_state)

    try:
        signature = inspect.signature(param_cls.__new__)
    except (TypeError, ValueError):
        setattr(param_cls, "_cvsuite_unknown_kw_compat_patched", False)
        return False

    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        setattr(param_cls, "_cvsuite_unknown_kw_compat_patched", False)
        return False

    accepted_kwargs = {parameter.name for parameter in parameters if parameter.name != "cls"}
    original_new = param_cls.__new__

    def _patched_new(cls, *args, **kwargs):
        forwarded_kwargs = kwargs
        extra_kwargs: dict[str, Any] = {}
        if kwargs:
            extra_kwargs = {key: value for key, value in kwargs.items() if key not in accepted_kwargs}
            if extra_kwargs:
                forwarded_kwargs = {key: value for key, value in kwargs.items() if key in accepted_kwargs}
        value = original_new(cls, *args, **forwarded_kwargs)
        for key, attr_value in extra_kwargs.items():
            try:
                setattr(value, key, attr_value)
            except Exception:
                continue
        return value

    param_cls.__new__ = _patched_new
    setattr(param_cls, "_cvsuite_unknown_kw_compat_patched", True)
    return True


def ensure_bitsandbytes_quant_parameter_compat() -> list[str]:
    import bitsandbytes.nn.modules as bnb_modules

    patched: list[str] = []
    for class_name in ("Params4bit", "Int8Params"):
        param_cls = getattr(bnb_modules, class_name, None)
        if param_cls is None:
            continue
        if patch_quantized_parameter_constructor_compat(param_cls):
            patched.append(class_name)
    return patched


def canonical_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return "fp32"


def available_cpu_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages


def build_precision_meta(precision: ResolvedPrecision) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "requested_precision": precision.requested,
        "effective_precision": precision.effective,
        "compute_dtype": canonical_dtype_name(precision.compute_dtype),
    }
    if precision.quantization_mode is not None:
        meta["quantization"] = precision.quantization_mode
    if precision.quantization_config is not None and hasattr(precision.quantization_config, "bnb_4bit_use_double_quant"):
        meta["double_quant"] = bool(precision.quantization_config.bnb_4bit_use_double_quant)
    return meta


def require_cuda_available(pref: str = "cuda") -> None:
    choice = normalize_device_pref(pref)
    if choice != "cuda":
        return
    if torch.cuda.is_available():
        return
    torch_cuda = getattr(torch.version, "cuda", None)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_text = "<unset>" if visible_devices is None else visible_devices
    raise RuntimeError(
        "CUDA was explicitly requested, but torch.cuda.is_available() is False. "
        f"requested={pref!r} torch.version.cuda={torch_cuda!r} CUDA_VISIBLE_DEVICES={visible_text!r}. "
        "Use --device auto to allow CPU fallback, or install/use a CUDA-enabled torch environment."
    )


def select_device(pref: str = "auto", force_cpu: bool = False) -> torch.device:
    choice = normalize_device_pref(pref)
    if force_cpu or choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        require_cuda_available(pref)
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision(device: torch.device, precision: str, *, allow_nf4: bool = False) -> ResolvedPrecision:
    requested = str(precision or "fp32").strip() or "fp32"
    normalized = normalize_precision_name(requested)

    if normalized == "nf4":
        if not allow_nf4:
            return ResolvedPrecision(
                requested=requested,
                normalized=normalized,
                effective="fp32",
                compute_dtype=torch.float32,
                warning=f"Precision {requested!r} is not supported for this model; falling back to fp32.",
            )
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Precision 'nf4' requires CUDA and is only supported for VLM wrappers running on GPU.")
        if not optional_dependency_available("transformers"):
            raise RuntimeError("Precision 'nf4' requires transformers support in the model environment.")
        from transformers import BitsAndBytesConfig

        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
        return ResolvedPrecision(
            requested=requested,
            normalized=normalized,
            effective="nf4",
            compute_dtype=compute_dtype,
            quantization_mode="nf4",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            ),
        )

    if normalized == "bf16":
        if device.type == "cuda" and torch.cuda.is_available() and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
            return ResolvedPrecision(
                requested=requested,
                normalized=normalized,
                effective="bf16",
                compute_dtype=torch.bfloat16,
            )
        return ResolvedPrecision(
            requested=requested,
            normalized=normalized,
            effective="fp32",
            compute_dtype=torch.float32,
            warning=f"Precision {requested!r} could not be honored on this device; falling back to fp32.",
        )

    if normalized == "fp16":
        if device.type == "cuda":
            return ResolvedPrecision(
                requested=requested,
                normalized=normalized,
                effective="fp16",
                compute_dtype=torch.float16,
            )
        return ResolvedPrecision(
            requested=requested,
            normalized=normalized,
            effective="fp32",
            compute_dtype=torch.float32,
            warning=f"Precision {requested!r} requires CUDA; falling back to fp32.",
        )

    if normalized == "fp32":
        return ResolvedPrecision(
            requested=requested,
            normalized=normalized,
            effective="fp32",
            compute_dtype=torch.float32,
        )

    return ResolvedPrecision(
        requested=requested,
        normalized=normalized,
        effective="fp32",
        compute_dtype=torch.float32,
        warning=f"Precision {requested!r} is not recognized; falling back to fp32.",
    )


def select_dtype(device: torch.device, precision: str) -> torch.dtype:
    return resolve_precision(device, precision, allow_nf4=False).compute_dtype


def maybe_autocast(device: torch.device, precision: str | ResolvedPrecision):
    if isinstance(precision, ResolvedPrecision):
        p = precision.autocast_precision
    else:
        p = normalize_precision_name(precision)
    if device.type != "cuda":
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    if p == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if p == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def module_floating_dtype(module: Any) -> torch.dtype | None:
    for getter_name in ("parameters", "buffers"):
        getter = getattr(module, getter_name, None)
        if not callable(getter):
            continue
        try:
            tensors = getter()
        except TypeError:
            tensors = getter(recurse=True)
        for value in tensors:
            if isinstance(value, torch.Tensor) and value.is_floating_point() and value.device.type != "meta":
                return value.dtype
    return None


def maybe_autocast_for_module(device: torch.device, precision: str | ResolvedPrecision, module: Any):
    dtype = module_floating_dtype(module)
    if dtype is None:
        return maybe_autocast(device, precision)
    return maybe_autocast(device, canonical_dtype_name(dtype))


def resolve_path(rec_path: Path, root: Path | None, caller_cwd: Path, allow_root_basename: bool = True) -> Path:
    rec_path = rec_path if isinstance(rec_path, Path) else Path(rec_path)
    if rec_path.is_absolute():
        return rec_path

    candidates: List[Path] = []
    if root is not None:
        base_root = root if isinstance(root, Path) else Path(root)
        if not base_root.is_absolute():
            base_root = caller_cwd / base_root
        if allow_root_basename and rec_path.parts and rec_path.parts[0] == base_root.name:
            candidates.append(base_root.parent / rec_path)
        candidates.append(base_root / rec_path)

    candidates.append(caller_cwd / rec_path)
    candidates.append(rec_path)

    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def batched(seq: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def bbox_to_norm(xyxy: np.ndarray, w: float, h: float) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = xyxy.tolist()
    cx = ((x0 + x1) / 2.0) / w
    cy = ((y0 + y1) / 2.0) / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    return cx, cy, bw, bh


def mask_to_polygons(mask: np.ndarray) -> List[List[Tuple[float, float]]]:
    import cv2

    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[Tuple[float, float]]] = []
    if not contours:
        return polys
    H, W = mask.shape[:2]
    min_area = max(1.0, 0.0005 * H * W)
    eps = 0.002 * max(H, W)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = [(float(x), float(y)) for [[x, y]] in approx]
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def polys_to_norm(polys: List[List[Tuple[float, float]]], w: float, h: float) -> List[List[Tuple[float, float]]]:
    if w <= 0 or h <= 0:
        return []
    out: List[List[Tuple[float, float]]] = []
    for poly in polys:
        out.append([(px / w, py / h) for px, py in poly])
    return out


def to_numpy(val) -> np.ndarray | None:
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return np.array(val)


def append_once(target: List[str], value: str) -> None:
    if value not in target:
        target.append(value)


def get_fm_request(dataset: VisionDataset) -> FMRequest | None:
    return dataset.fm_request


def request_prompt(dataset: VisionDataset, fallback: str = "") -> str:
    request = get_fm_request(dataset)
    if request is None or not request.prompt:
        return fallback
    return str(request.prompt)


def request_model_arg(dataset: VisionDataset, key: str, fallback: Any = None) -> Any:
    request = get_fm_request(dataset)
    if request is None:
        return fallback
    if key not in request.model_args:
        return fallback
    return request.model_args[key]


def ground_prompts_for_record(record: Record) -> List[str]:
    raw = record.attributes.get("ground_prompts", [])
    if not isinstance(raw, list):
        raise ValueError("Record.attributes['ground_prompts'] must be a list of strings.")
    prompts: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("Record.attributes['ground_prompts'] must contain strings only.")
        if item:
            prompts.append(item)
    return prompts


def dataset_ground_prompts(dataset: VisionDataset) -> List[str]:
    prompts: List[str] = []
    seen: set[str] = set()
    for record in dataset.records:
        for prompt in ground_prompts_for_record(record):
            if prompt in seen:
                continue
            seen.add(prompt)
            prompts.append(prompt)
    return prompts


def ensure_hf_caches(cache_root: Path | None, *, hub: Path | None = None) -> None:
    """Redirect Hugging Face and torch caches under `cache_root`.

    Containment is the point: when `cache_root` is a staging directory, everything a
    download pulls in transitively lands there and dies with it, so the durable cache only
    ever receives a finished snapshot. Pass `hub` to send repo downloads to the shared hub
    instead -- correct for a durable `cache_root`, wrong during staging.
    """

    if not cache_root:
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    hf_home = cache_root / "hf_home"
    hf_hub_cache = hub if hub is not None else cache_root / "hub"
    datasets_cache = cache_root / "datasets"
    xdg_cache = cache_root / "xdg"
    torch_home = cache_root / "torch"
    for path in (hf_home, hf_hub_cache, datasets_cache, xdg_cache, torch_home):
        path.mkdir(parents=True, exist_ok=True)

    # Force all Hugging Face and torch caches to model-local paths.
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub_cache)
    for key in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE"):
        os.environ.pop(key, None)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    os.environ["TORCH_HOME"] = str(torch_home)
    had_token_candidate = False
    for token, source in _iter_hf_token_candidates():
        had_token_candidate = True
        if not _validate_hf_token(token, source):
            continue
        for key in HF_CANONICAL_TOKEN_KEYS:
            os.environ[key] = token
        break
    else:
        if had_token_candidate:
            for key in HF_CANONICAL_TOKEN_KEYS:
                os.environ.pop(key, None)


def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = box_a.tolist()
    bx0, by0, bx1, by1 = box_b.tolist()
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter_area
    return float(inter_area / denom) if denom > 0 else 0.0


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> List[int]:
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        order = rest[ious <= iou_thresh]
    return keep


def normalize_prompt_text(prompt: str) -> str:
    q = prompt.lower().strip()
    if not q.endswith("."):
        q += "."
    return q


def dedup_detections(entries: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
    if not entries:
        return []
    boxes = np.stack([e["box"] for e in entries])
    scores = np.array([e["score"] for e in entries], dtype=np.float32)
    keep = nms_indices(boxes, scores, iou_thresh=iou_thresh)
    return [entries[i] for i in keep]
