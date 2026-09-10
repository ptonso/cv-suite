from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Iterable, List, Optional, Sequence, TypeVar

import yaml

from cvsuite.common.core.enums import Task
from cvsuite.common.core import FMRequest, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.core.utils import _decode_dataclass, _encode

from cvsuite.common.fm.core import utils as fm_utils
from cvsuite.common.fm.core.paths import hub_dir as resolve_hub_dir
from cvsuite.common.fm.core.paths import providers_dir as resolve_providers_dir
from cvsuite.common.fm.core.paths import cvsuite_home

if TYPE_CHECKING:
    from cvsuite.common.fm.providers.bases.vlm import HFLoadPlacement

try:  # pragma: no cover - tqdm is optional in the model envs
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback keeps runtime usable without tqdm
    tqdm = None

TOptions = TypeVar("TOptions")
TJob = TypeVar("TJob")
TRuntime = TypeVar("TRuntime")


@dataclass(frozen=True)
class ProcessArgs:
    input: Path
    output: Path
    no_resume: bool = False


@dataclass
class BatchResult:
    modified_record_indices: List[int] = field(default_factory=list)
    completed_job_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _env_path(key: str, default: Path) -> Path:
    """Read an FM_* root, falling back to the resolver FMRunner itself uses.

    The fallback matters when a provider module is run directly rather than spawned by
    FMRunner: it must land on the same roots, not scatter a second cache elsewhere.
    """

    raw = str(os.environ.get(key) or "").strip()
    return Path(raw) if raw else default


@dataclass
class RuntimeContext(Generic[TOptions]):
    model_name: str
    request: FMRequest
    prompt: str
    options: TOptions
    input_path: Path
    output_path: Path
    caller_cwd: Path
    cache_dir: Path
    hub_dir: Path
    model_cache: Path
    runs_root: Path
    weights_dir: Optional[Path]
    stage_dir: Path
    config_path: Optional[Path]
    work_dir: Path
    manifest_path: Path
    records_path: Path
    completed_jobs_path: Path
    state_path: Path
    batch_size: int
    resume_enabled: bool


class BaseFMModel(Generic[TOptions, TJob, TRuntime]):
    model_name: str = ""
    description: str = "Foundation model wrapper."
    options_cls: type[Any] | None = None
    supports_nf4_precision: bool = False
    supports_managed_auto_device: bool = False

    def __init__(self) -> None:
        if not self.model_name:
            module_name = self.__class__.__module__
            if module_name == "__main__":
                main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
                module_name = getattr(main_spec, "name", module_name) or module_name
            self.model_name = module_name.rsplit(".", 1)[-1]
        self._last_hf_load_placement: HFLoadPlacement | None = None

    def build_cli_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument("--input", type=Path, required=True, help="Path to VisionDataset JSON.")
        parser.add_argument("--output", type=Path, required=True, help="Where to write updated VisionDataset JSON.")
        parser.add_argument("--no-resume", action="store_true", help="Ignore any existing JSONL checkpoint for this run.")
        return parser

    def main(self, argv: Sequence[str] | None = None) -> int:
        args = self.build_cli_parser().parse_args(argv)
        process_args = ProcessArgs(input=args.input, output=args.output, no_resume=args.no_resume)
        self.run_from_paths(process_args)
        return 0

    def run_from_paths(self, process_args: ProcessArgs) -> VisionDataset:
        dataset = VisionDataset.from_json(process_args.input)
        request = dataset.fm_request
        if request is None:
            raise ValueError("VisionDataset.fm_request must be populated before running a model.")
        if request.provider and request.provider != self.model_name:
            raise ValueError(f"Dataset requested provider {request.provider!r}, but {self.model_name!r} was invoked.")
        self.validate_and_sanitize_model_args(dataset, request)

        caller_cwd = Path(os.environ.get("FM_CALLER_CWD", Path.cwd()))
        prompt = self.resolve_prompt(request.prompt, caller_cwd)
        options = self.parse_options(request.model_args)
        cache_dir = _env_path("FM_CACHE_DIR", cvsuite_home())
        hub_dir = _env_path("FM_HUB_DIR", resolve_hub_dir())
        model_cache = _env_path("FM_MODEL_CACHE", resolve_providers_dir() / self.model_name)
        runs_root = Path(os.environ.get("FM_RUNS_ROOT", Path(tempfile.gettempdir()) / "cvsuite-fm-runs"))
        weights_env = os.environ.get("FM_WEIGHTS_DIR")
        weights_dir = Path(weights_env) if weights_env else request.weights_dir
        stage_dir = Path(os.environ.get("FM_STAGE_DIR", model_cache / "stage"))
        config_path = self.resolve_config_path(request.config_path, caller_cwd)
        run_id = self.compute_run_id(dataset, prompt=prompt, config_path=config_path, caller_cwd=caller_cwd)
        work_dir = runs_root / self.model_name / run_id
        ctx = RuntimeContext(
            model_name=self.model_name,
            request=request,
            prompt=prompt,
            options=options,
            input_path=process_args.input,
            output_path=process_args.output,
            caller_cwd=caller_cwd,
            cache_dir=cache_dir,
            hub_dir=hub_dir,
            model_cache=model_cache,
            runs_root=runs_root,
            weights_dir=weights_dir,
            stage_dir=stage_dir,
            config_path=config_path,
            work_dir=work_dir,
            manifest_path=work_dir / "manifest.json",
            records_path=work_dir / "records.jsonl",
            completed_jobs_path=work_dir / "completed_jobs.jsonl",
            state_path=work_dir / "state.json",
            batch_size=max(1, int(request.batch_size or 1)),
            resume_enabled=not process_args.no_resume,
        )
        self.validate_auto_device_request(request)
        self.ensure_work_dir(ctx)

        completed_job_ids = set()
        if ctx.resume_enabled:
            completed_job_ids = self.restore_checkpoint(dataset, ctx)

        jobs = list(self.build_jobs(dataset, ctx))
        completed_job_ids = self.validate_completed_job_ids(dataset, jobs, ctx, completed_job_ids)
        pending_jobs = [job for job in jobs if self.job_id(job, dataset, ctx) not in completed_job_ids]
        runtime = self.load_runtime(dataset, ctx)
        iterator = self.iter_batches(pending_jobs, dataset, ctx)

        for batch in iterator:
            result = self.process_batch(dataset, batch, runtime, ctx)
            warnings = self.normalize_warnings(result.warnings)
            self.add_warnings(dataset, warnings)
            for warning in warnings:
                print(warning, file=sys.stderr)
            job_ids = result.completed_job_ids or [self.job_id(job, dataset, ctx) for job in batch]
            self.save_checkpoint(
                dataset,
                ctx,
                modified_record_indices=result.modified_record_indices,
                completed_job_ids=job_ids,
            )

        self.finalize_dataset(dataset, runtime, ctx)
        self.update_fm_meta(dataset, runtime, ctx)
        dataset.to_json(process_args.output)
        return dataset

    def ensure_work_dir(self, ctx: RuntimeContext[TOptions]) -> None:
        if not ctx.resume_enabled and ctx.work_dir.exists():
            shutil.rmtree(ctx.work_dir, ignore_errors=True)
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        if not ctx.manifest_path.exists():
            self.write_json(
                ctx.manifest_path,
                {
                    "provider": ctx.model_name,
                    "input": str(ctx.input_path),
                    "output": str(ctx.output_path),
                    "resume_enabled": ctx.resume_enabled,
                },
            )

    def resolve_prompt(self, raw_prompt: str, caller_cwd: Path) -> str:
        prompt_str = str(raw_prompt or "").strip()
        if not prompt_str.endswith(".yaml"):
            return raw_prompt
        prompt_path = Path(prompt_str)
        if not prompt_path.is_absolute():
            prompt_path = caller_cwd / prompt_path
        if not prompt_path.exists():
            alt = Path.cwd() / Path(prompt_str)
            if alt.exists():
                prompt_path = alt
            else:
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        try:
            data = yaml.safe_load(prompt_path.read_text()) or []
        except yaml.YAMLError as exc:  # pragma: no cover - parse errors bubble to caller
            raise ValueError(f"Failed to parse prompt YAML {prompt_path}: {exc}") from exc
        if isinstance(data, (list, dict)):
            return json.dumps(data)
        raise ValueError(f"Prompt YAML must contain a list or dict; got {type(data).__name__}.")

    def resolve_config_path(self, config_path: Path | None, caller_cwd: Path) -> Path | None:
        if config_path is None:
            return None
        if config_path.is_absolute():
            return config_path
        return caller_cwd / config_path

    def model_arg_options_schema_name(self) -> str | None:
        cls = self.options_cls
        if cls is None:
            return None
        if cls is dict:
            return "builtins.dict"
        return f"{cls.__module__}.{cls.__qualname__}"

    def supported_model_arg_keys(self) -> tuple[str, ...] | None:
        cls = self.options_cls
        if cls is None or cls is dict:
            return None
        if not is_dataclass(cls):
            raise TypeError(f"{self.__class__.__name__}.options_cls must be a dataclass type, dict, or None.")
        return tuple(sorted(field.name for field in fields(cls)))

    def sanitize_model_args(
        self,
        raw_model_args: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
        model_args = dict(raw_model_args or {})
        supported = self.supported_model_arg_keys()
        if supported is None:
            return model_args, (), ()
        supported_set = set(supported)
        unknown = tuple(sorted(key for key in model_args if key not in supported_set))
        if not unknown:
            return model_args, (), supported
        sanitized = {
            key: value
            for key, value in model_args.items()
            if key in supported_set
        }
        return sanitized, unknown, supported

    def validate_and_sanitize_model_args(self, dataset: VisionDataset, request: FMRequest) -> None:
        model_args, unknown, supported = self.sanitize_model_args(request.model_args)
        if unknown:
            supported_text = ", ".join(supported) if supported else "none"
            schema_name = self.model_arg_options_schema_name() or "unknown"
            raise SystemExit(
                f"[{self.model_name}] Unsupported --model-arg key(s): {', '.join(unknown)}. "
                f"Supported keys: {supported_text}. Options class: {schema_name}."
            )
        request.model_args = model_args

    def parse_options(self, raw_model_args: dict[str, Any]) -> TOptions:
        model_args, _unknown, _supported = self.sanitize_model_args(raw_model_args)
        cls = self.options_cls
        if cls is None or cls is dict:
            return model_args  # type: ignore[return-value]
        return _decode_dataclass(cls, model_args)

    def compute_run_id(
        self,
        dataset: VisionDataset,
        *,
        prompt: str,
        config_path: Path | None,
        caller_cwd: Path,
    ) -> str:
        payload = _encode(dataset)
        if isinstance(payload, dict):
            request_payload = dict(payload.get("fm_request") or {})
            request_payload.pop("no_resume", None)
            request_payload["prompt"] = prompt
            if config_path is not None:
                request_payload["config_path"] = str(config_path)
                if config_path.exists():
                    request_payload["config_sha256"] = self.hash_file(config_path)
            payload["fm_request"] = request_payload
        class_file = None
        try:
            candidate = inspect.getfile(self.__class__)
            if candidate and not candidate.startswith("<"):
                class_file = Path(candidate)
        except (OSError, TypeError):
            class_file = None
        fingerprint = {
            "dataset": payload,
            "input_image_count": len(dataset.records),
            "input_images": self.dataset_image_fingerprint(dataset, caller_cwd),
            "provider": self.model_name,
            "provider_module": str(class_file) if class_file else f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "provider_module_sha256": self.hash_file(class_file) if class_file else None,
            "base_module_sha256": self.hash_file(Path(__file__)),
        }
        encoded = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def dataset_image_fingerprint(self, dataset: VisionDataset, caller_cwd: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for idx, record in enumerate(dataset.records):
            resolved = fm_utils.resolve_path(record.image.path, dataset.root, caller_cwd)
            row: dict[str, Any] = {
                "record_idx": idx,
                "image_path": str(record.image.path),
                "resolved_path": str(resolved),
            }
            try:
                stat = resolved.stat()
            except OSError:
                row["exists"] = False
            else:
                row["exists"] = True
                row["size"] = int(stat.st_size)
                row["mtime_ns"] = int(stat.st_mtime_ns)
            items.append(row)
        return items

    def restore_checkpoint(self, dataset: VisionDataset, ctx: RuntimeContext[TOptions]) -> set[str]:
        if not ctx.state_path.exists():
            return set()

        state = json.loads(ctx.state_path.read_text(encoding="utf-8"))
        dataset.classes = list(state.get("classes") or [])
        raw_task = state.get("task")
        if raw_task is None or isinstance(raw_task, Task):
            dataset.task = raw_task
        else:
            dataset.task = Task(raw_task)
        dataset.meta = dict(state.get("meta") or {})

        restored_records: dict[int, Record] = {}
        for row in self.read_jsonl(ctx.records_path):
            raw_idx = row.get("record_idx")
            raw_record = row.get("record")
            if not isinstance(raw_idx, int) or not isinstance(raw_record, dict):
                continue
            restored_records[raw_idx] = _decode_dataclass(Record, raw_record)
        for idx, record in restored_records.items():
            if 0 <= idx < len(dataset.records):
                dataset.records[idx] = record

        completed = {
            str(row["job_id"])
            for row in self.read_jsonl(ctx.completed_jobs_path)
            if row.get("job_id") is not None
        }
        return completed

    def validate_completed_job_ids(
        self,
        dataset: VisionDataset,
        jobs: Sequence[TJob],
        ctx: RuntimeContext[TOptions],
        completed_job_ids: set[str],
    ) -> set[str]:
        return completed_job_ids

    def save_checkpoint(
        self,
        dataset: VisionDataset,
        ctx: RuntimeContext[TOptions],
        *,
        modified_record_indices: Sequence[int],
        completed_job_ids: Sequence[str],
    ) -> None:
        rows = []
        for idx in sorted(set(int(i) for i in modified_record_indices if 0 <= int(i) < len(dataset.records))):
            rows.append({"record_idx": idx, "record": _encode(dataset.records[idx])})
        if rows:
            self.append_jsonl(ctx.records_path, rows)
        completed_rows = [{"job_id": job_id} for job_id in completed_job_ids]
        if completed_rows:
            self.append_jsonl(ctx.completed_jobs_path, completed_rows)
        self.write_json(
            ctx.state_path,
            {
                "classes": _encode(dataset.classes),
                "task": _encode(dataset.task),
                "meta": _encode(dataset.meta),
            },
        )

    def iter_batches(
        self,
        jobs: Sequence[TJob],
        dataset: VisionDataset,
        ctx: RuntimeContext[TOptions],
    ) -> Iterable[Sequence[TJob]]:
        batches = [
            jobs[idx : idx + ctx.batch_size]
            for idx in range(0, len(jobs), ctx.batch_size)
        ]
        if tqdm is None:
            return batches
        return tqdm(batches, total=len(batches), desc=f"[fm:{ctx.model_name}] batches", disable=None)

    def normalize_warnings(self, warnings: Sequence[str]) -> List[str]:
        return [str(msg) for msg in warnings if str(msg)]

    def add_warnings(self, dataset: VisionDataset, warnings: Sequence[str]) -> None:
        if not warnings:
            return
        fm_meta = dataset.meta.setdefault("fm", {})
        bucket = fm_meta.setdefault("warnings", [])
        for warning in warnings:
            if warning not in bucket:
                bucket.append(warning)

    def update_fm_meta(self, dataset: VisionDataset, runtime: TRuntime, ctx: RuntimeContext[TOptions]) -> None:
        existing_fm = dict(dataset.meta.get("fm", {}))
        warnings = list(existing_fm.get("warnings", []))
        existing_fm.update(self.build_fm_meta(dataset, runtime, ctx))
        if warnings:
            existing_fm["warnings"] = warnings
        dataset.meta["fm"] = existing_fm

    def emit_runtime_warning(self, dataset: VisionDataset, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        self.add_warnings(dataset, [text])
        print(text, file=sys.stderr)

    def validate_auto_device_request(self, request: FMRequest, *, require_supported: bool = True) -> None:
        if fm_utils.normalize_device_pref(request.device) != "auto":
            return
        if fm_utils.normalize_precision_name(request.precision) == "nf4":
            raise RuntimeError(
                "`--device auto --precision nf4` is not supported yet. "
                "Managed auto placement currently supports fp32, fp16, and bf16 Hugging Face loads only. "
                "Use `--device cuda --precision nf4` or switch to fp32/fp16/bf16."
            )
        if require_supported and not self.supports_managed_auto_device:
            raise RuntimeError(
                f"`--device auto` is not implemented for model '{self.model_name}'. "
                "Managed auto placement is currently only supported for selected Hugging Face-backed models. "
                "Use `--device cpu` or `--device cuda`."
            )

    @staticmethod
    def _summarize_subprocess_failure(result: subprocess.CompletedProcess[str]) -> str:
        lines = [line.strip() for line in (result.stderr or result.stdout or "").splitlines() if line.strip()]
        return lines[-1] if lines else f"exit={result.returncode}"

    def ensure_optional_dependency(
        self,
        *,
        package_name: str,
        module_name: str | None = None,
        ctx: RuntimeContext[TOptions],
        reason: str,
    ) -> None:
        import_name = module_name or package_name
        if fm_utils.optional_dependency_available(import_name):
            return
        cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", package_name]
        cmd_text = " ".join(cmd)
        print(
            f"[{self.model_name}] Installing optional dependency '{package_name}' in the managed model venv for {reason}...",
            file=sys.stderr,
        )
        result = subprocess.run(
            cmd,
            check=False,
            cwd=ctx.model_cache,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            summary = self._summarize_subprocess_failure(result)
            raise RuntimeError(
                f"Automatic installation of optional dependency '{package_name}' failed ({summary}). "
                f"Retry manually with: {cmd_text}"
            )
        importlib.invalidate_caches()
        if import_name == "bitsandbytes":
            fm_utils.refresh_transformers_bitsandbytes_availability()
        if not fm_utils.optional_dependency_available(import_name):
            raise RuntimeError(
                f"Installed optional dependency '{package_name}', but Python still cannot import {import_name!r}. "
                f"Retry manually with: {cmd_text}"
            )
        print(f"[{self.model_name}] Installed optional dependency '{package_name}'.", file=sys.stderr)

    def resolve_runtime_precision(
        self,
        dataset: VisionDataset,
        ctx: RuntimeContext[TOptions],
        device,
    ) -> fm_utils.ResolvedPrecision:
        requested = fm_utils.normalize_precision_name(ctx.request.precision)
        if requested == "nf4" and self.supports_nf4_precision:
            if getattr(device, "type", None) != "cuda":
                raise RuntimeError("Precision 'nf4' requires CUDA and is only supported for wrappers running on GPU.")
            if not fm_utils.optional_dependency_available("bitsandbytes"):
                self.ensure_optional_dependency(
                    package_name="bitsandbytes",
                    ctx=ctx,
                    reason="NF4 quantization",
                )
            try:
                fm_utils.ensure_bitsandbytes_quant_parameter_compat()
            except Exception as exc:
                raise RuntimeError(
                    "bitsandbytes is available, but its quantized parameter classes could not be prepared for NF4 loading."
                ) from exc
        precision = fm_utils.resolve_precision(device, ctx.request.precision, allow_nf4=self.supports_nf4_precision)
        if precision.warning:
            self.emit_runtime_warning(dataset, f"[{self.model_name}] {precision.warning}")
        return precision

    @staticmethod
    def build_hf_precision_kwargs(
        precision: fm_utils.ResolvedPrecision,
        *,
        dtype_key: str = "torch_dtype",
        allow_quantized_cpu_offload: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if precision.quantization_config is not None:
            if allow_quantized_cpu_offload and hasattr(precision.quantization_config, "llm_int8_enable_fp32_cpu_offload"):
                precision.quantization_config.llm_int8_enable_fp32_cpu_offload = True
            kwargs["quantization_config"] = precision.quantization_config
            return kwargs
        kwargs[dtype_key] = precision.compute_dtype
        return kwargs

    def build_hf_load_kwargs(
        self,
        ctx: RuntimeContext[TOptions],
        device,
        precision: fm_utils.ResolvedPrecision,
        *,
        dtype_key: str = "torch_dtype",
        include_default_device_map: bool = True,
        default_device_map: Any = None,
        default_offload_folder: bool = False,
    ) -> tuple[dict[str, Any], "HFLoadPlacement"]:
        from cvsuite.common.fm.providers.bases.vlm import resolve_hf_load_placement

        self.validate_auto_device_request(ctx.request, require_supported=False)
        model_cache = getattr(ctx, "model_cache", None)
        if model_cache is None:
            model_cache = getattr(ctx, "weights_dir", None) or Path.cwd()
        placement = resolve_hf_load_placement(
            ctx.request,
            device=device,
            stage_dir=getattr(ctx, "stage_dir", None),
            model_cache=model_cache,
            include_default_device_map=include_default_device_map,
            default_device_map=default_device_map,
            default_offload_folder=default_offload_folder,
        )
        kwargs = self.build_hf_precision_kwargs(
            precision,
            dtype_key=dtype_key,
            allow_quantized_cpu_offload=placement.offload_cap_enabled,
        )
        if placement.device_map is not None:
            kwargs["device_map"] = placement.device_map
        if placement.max_memory is not None:
            kwargs["max_memory"] = placement.max_memory
        if placement.offload_folder is not None:
            kwargs["offload_folder"] = placement.offload_folder
        self._last_hf_load_placement = placement
        return kwargs, placement

    @staticmethod
    def _is_known_nf4_offload_failure(exc: Exception) -> bool:
        text = str(exc)
        return any(
            needle in text
            for needle in (
                "Tensor.item() cannot be called on meta tensors",
                "Cannot copy out of meta tensor; no data!",
            )
        )

    def load_hf_pretrained_model(
        self,
        loader: Any,
        precision: fm_utils.ResolvedPrecision,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            model = loader(*args, **kwargs)
        except Exception as exc:
            if precision.quantization_mode == "nf4" and self._is_known_nf4_offload_failure(exc):
                raise RuntimeError(
                    "NF4 quantization fell through to a disk-offloaded/meta-tensor path during Hugging Face loading. "
                    "Plain `--device auto` can work here because regular fp16/bf16 weights may be offloaded to CPU or disk, "
                    "but the managed transformers/accelerate/bitsandbytes stack is not reliably handling this model with 4-bit disk "
                    "offload. Try increasing `--max-gpu-memory`, freeing more system RAM so the fallback stays in CPU RAM instead of "
                    "disk, or use a smaller checkpoint."
                ) from exc
            raise
        self.validate_hf_quantized_runtime(model, precision)
        self.emit_hf_auto_device_report(model)
        return model

    @staticmethod
    def _format_bytes(num_bytes: int | float) -> str:
        value = float(max(0, num_bytes))
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024.0 or unit == "TiB":
                if unit == "B":
                    return f"{int(value)} {unit}"
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f} TiB"

    @staticmethod
    def _format_device_map_target(target: Any) -> str:
        if isinstance(target, int):
            return f"cuda:{target}"
        text = str(target)
        return f"cuda:{text}" if text.isdigit() else text

    @staticmethod
    def _parameter_nbytes(param: Any) -> int:
        try:
            return int(param.numel()) * int(param.element_size())
        except Exception:
            return 0

    @staticmethod
    def _iter_model_parameters(model: Any) -> Iterable[tuple[str, Any]]:
        if hasattr(model, "named_parameters"):
            try:
                yield from model.named_parameters()
                return
            except Exception:
                return
        components = getattr(model, "components", None)
        if isinstance(components, dict):
            for component_name, component in components.items():
                if not hasattr(component, "named_parameters"):
                    continue
                try:
                    for name, param in component.named_parameters():
                        yield f"{component_name}.{name}", param
                except Exception:
                    continue

    @staticmethod
    def _collect_device_map(model: Any) -> dict[str, Any]:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            return dict(device_map)
        merged: dict[str, Any] = {}
        components = getattr(model, "components", None)
        if isinstance(components, dict):
            for component_name, component in components.items():
                component_map = getattr(component, "hf_device_map", None)
                if not isinstance(component_map, dict):
                    continue
                for key, target in component_map.items():
                    prefix = str(component_name)
                    suffix = str(key)
                    merged[prefix if suffix == "" else f"{prefix}.{suffix}"] = target
        return merged

    @classmethod
    def _target_for_parameter(cls, param_name: str, device_map: dict[str, Any], param: Any) -> str:
        best_key: str | None = None
        for key in device_map:
            key_text = str(key)
            if key_text == "" or param_name == key_text or param_name.startswith(f"{key_text}."):
                if best_key is None or len(key_text) > len(best_key):
                    best_key = key_text
        if best_key is not None:
            return cls._format_device_map_target(device_map[best_key])
        device = getattr(param, "device", None)
        return str(device) if device is not None else "unknown"

    @classmethod
    def _component_for_parameter(cls, param_name: str, device_map: dict[str, Any]) -> str:
        best_key = ""
        for key in device_map:
            key_text = str(key)
            if key_text and (param_name == key_text or param_name.startswith(f"{key_text}.")) and len(key_text) > len(best_key):
                best_key = key_text
        source = best_key or param_name
        if not source:
            return "(root)"
        parts = source.split(".")
        if len(parts) >= 3 and parts[1] in {"blocks", "layers", "transformer_blocks"}:
            return ".".join(parts[:3])
        if len(parts) >= 4 and parts[2] in {"blocks", "layers", "transformer_blocks"}:
            return ".".join(parts[:4])
        return parts[0]

    @classmethod
    def summarize_hf_auto_device_report(cls, model: Any, placement: "HFLoadPlacement | None") -> list[str]:
        if not placement or not placement.managed_device_map_active:
            return []
        device_map = cls._collect_device_map(model)
        by_target: dict[str, int] = defaultdict(int)
        by_component_target: dict[tuple[str, str], int] = defaultdict(int)
        total = 0
        for name, param in cls._iter_model_parameters(model):
            nbytes = cls._parameter_nbytes(param)
            if nbytes <= 0:
                continue
            target = cls._target_for_parameter(str(name), device_map, param)
            component = cls._component_for_parameter(str(name), device_map)
            by_target[target] += nbytes
            by_component_target[(component, target)] += nbytes
            total += nbytes

        lines = ["managed auto device placement:"]
        if placement.max_memory:
            budget_items = [
                f"{cls._format_device_map_target(key)}={cls._format_bytes(value) if isinstance(value, int) else value}"
                for key, value in placement.max_memory.items()
            ]
            lines.append(f"  budget: {', '.join(budget_items)}")
        if placement.offload_folder:
            lines.append(f"  offload: {placement.offload_folder}")
        if not total:
            lines.append("  estimated parameter storage: unavailable")
            return lines

        target_parts = []
        for target, nbytes in sorted(by_target.items(), key=lambda item: item[1], reverse=True):
            share = (float(nbytes) / float(total)) * 100.0 if total else 0.0
            target_parts.append(f"{target} ~{cls._format_bytes(nbytes)} ({share:.0f}%)")
        lines.append(f"  estimated parameter storage: total ~{cls._format_bytes(total)}; " + "; ".join(target_parts))

        largest = sorted(by_component_target.items(), key=lambda item: item[1], reverse=True)[:6]
        if largest:
            parts = [
                f"{component}->{target} ~{cls._format_bytes(nbytes)}"
                for (component, target), nbytes in largest
            ]
            lines.append(f"  largest components: {'; '.join(parts)}")
        return lines

    def emit_hf_auto_device_report(self, model: Any, placement: "HFLoadPlacement | None" = None) -> None:
        placement = placement if placement is not None else self._last_hf_load_placement
        lines = self.summarize_hf_auto_device_report(model, placement)
        for line in lines:
            print(f"[{self.model_name}] {line}", file=sys.stderr)

    @staticmethod
    def validate_hf_quantized_runtime(model: Any, precision: fm_utils.ResolvedPrecision) -> None:
        if precision.quantization_mode != "nf4":
            return
        device_map = getattr(model, "hf_device_map", None)
        if not isinstance(device_map, dict):
            return
        if any(str(target) == "disk" for target in device_map.values()):
            raise RuntimeError(
                "NF4 quantization loaded this model with disk offload in the inferred Hugging Face device map. "
                "That disk-offloaded 4-bit path is not reliable in the managed transformers/accelerate/bitsandbytes stack "
                "and can fail at runtime. Increase `--max-gpu-memory`, free more system RAM so the fallback stays in CPU RAM, "
                "or use a smaller checkpoint."
            )

    def build_jobs(self, dataset: VisionDataset, ctx: RuntimeContext[TOptions]) -> Sequence[TJob]:
        raise NotImplementedError

    def job_id(self, job: TJob, dataset: VisionDataset, ctx: RuntimeContext[TOptions]) -> str:
        raise NotImplementedError

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[TOptions]) -> TRuntime:
        raise NotImplementedError

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: Sequence[TJob],
        runtime: TRuntime,
        ctx: RuntimeContext[TOptions],
    ) -> BatchResult:
        raise NotImplementedError

    def finalize_dataset(self, dataset: VisionDataset, runtime: TRuntime, ctx: RuntimeContext[TOptions]) -> None:
        return None

    def build_fm_meta(self, dataset: VisionDataset, runtime: TRuntime, ctx: RuntimeContext[TOptions]) -> dict[str, Any]:
        return {}

    @staticmethod
    def hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def read_jsonl(path: Path) -> List[dict[str, Any]]:
        if not path.exists():
            return []
        rows: List[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False))
                fh.write("\n")

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
