from __future__ import annotations

import importlib
import inspect
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer
from transformers.dynamic_module_utils import HF_MODULES_CACHE, _sanitize_module_name, init_hf_modules

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.core.paths import hf_repo_cache_dir
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseVLMModel
from cvsuite.common.fm.providers.bases.vlm import VLMJob

DEFAULT_MODEL_ID = "openbmb/MiniCPM-V-4"
DEFAULT_HF_REVISION = "0968cf95dd0ed5584b5a50d0ded4ce29421674f7"
PARAMS = "8B"
MODEL_ALIASES = ("minicpm-v",)


def _resolve_model_revision(model_id: str, revision: str | None, hub_dir: Path | None) -> str | None:
    if revision:
        return revision
    if model_id == DEFAULT_MODEL_ID:
        return DEFAULT_HF_REVISION
    if hub_dir is None or "/" not in model_id:
        return None
    ref_path = hf_repo_cache_dir(hub_dir, model_id) / "refs" / "main"
    try:
        cached_revision = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return cached_revision or None


def _resolve_snapshot_dir(model_id: str, revision: str | None, hub_dir: Path | None) -> Path | None:
    if revision is None or hub_dir is None or "/" not in model_id:
        return None
    return hf_repo_cache_dir(hub_dir, model_id) / "snapshots" / revision


def _has_cached_snapshot(model_id: str, revision: str | None, hub_dir: Path | None) -> bool:
    snapshot_dir = _resolve_snapshot_dir(model_id, revision, hub_dir)
    if snapshot_dir is None or not snapshot_dir.is_dir():
        return False
    required = (
        "config.json",
        "configuration_minicpm.py",
        "modeling_minicpmv.py",
        "modeling_navit_siglip.py",
        "resampler.py",
        "model.safetensors.index.json",
    )
    return all((snapshot_dir / name).exists() for name in required)


def _patch_remote_minicpm_init(model_id: str, revision: str | None) -> None:
    if revision is None or "/" not in model_id:
        return
    namespace, repo = model_id.split("/", 1)
    module_name = ".".join(
        (
            "transformers_modules",
            _sanitize_module_name(namespace),
            _sanitize_module_name(repo),
            revision,
            "modeling_minicpmv",
        )
    )
    with suppress(Exception):
        init_hf_modules()
        modules_cache = Path(HF_MODULES_CACHE)
        if not modules_cache.exists():
            return
        module = importlib.import_module(module_name)
        minicpm_cls = getattr(module, "MiniCPMV", None)
        if minicpm_cls is None or getattr(minicpm_cls, "_cvsuite_post_init_patched", False):
            return
        original_init = minicpm_cls.__init__

        def _patched_init(self, config, *args, **kwargs):
            original_init(self, config, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.post_init()

        minicpm_cls.__init__ = _patched_init
        minicpm_cls._cvsuite_post_init_patched = True


@contextmanager
def _suppress_legacy_minicpm_image_processor_register():
    original_register = AutoImageProcessor.register

    def _patched_register(
        config_class,
        image_processor_class=None,
        slow_image_processor_class=None,
        fast_image_processor_class=None,
        exist_ok=False,
    ):
        if (
            config_class == "MiniCPMVImageProcessor"
            and image_processor_class is not None
            and slow_image_processor_class is None
            and fast_image_processor_class is None
        ):
            return None
        return original_register(
            config_class=config_class,
            image_processor_class=image_processor_class,
            slow_image_processor_class=slow_image_processor_class,
            fast_image_processor_class=fast_image_processor_class,
            exist_ok=exist_ok,
        )

    AutoImageProcessor.register = staticmethod(_patched_register)
    try:
        yield
    finally:
        AutoImageProcessor.register = staticmethod(original_register)


@dataclass(frozen=True)
class MiniCPMVOptions:
    model_id: str | None = None
    max_new_tokens: int | None = None
    trust_remote_code: bool | None = None
    enable_thinking: bool | None = None
    revision: str | None = None


@dataclass(frozen=True)
class MiniCPMVConfig:
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 64
    trust_remote_code: bool = True
    enable_thinking: bool = False
    revision: str | None = None


@dataclass
class MiniCPMVRuntime:
    cfg: MiniCPMVConfig
    revision: str | None
    device: torch.device
    dtype: torch.dtype
    model: AutoModel
    processor: AutoProcessor
    tokenizer: AutoTokenizer
    model_device: torch.device
    precision: utils.ResolvedPrecision | None = None


class MiniCPMVModel(BaseVLMModel[MiniCPMVOptions, MiniCPMVRuntime]):
    description = "MiniCPM-V wrapper: runs image-chat prompts over a VisionDataset JSON."
    options_cls = MiniCPMVOptions
    supports_nf4_precision = True
    supports_managed_auto_device = True
    warning_prefix = "minicpm-v"

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[MiniCPMVOptions]) -> MiniCPMVRuntime:
        cfg = self.load_dataclass_config(ctx.config_path, MiniCPMVConfig, ctx.options)
        requested_revision = _resolve_model_revision(cfg.model_id, cfg.revision, ctx.hub_dir)
        source = utils.materialize_hf_model_source(
            cfg.model_id,
            hub_dir=ctx.hub_dir,
            stage_dir=getattr(ctx, "stage_dir", None),
            caller_cwd=getattr(ctx, "caller_cwd", Path.cwd()),
            revision=requested_revision,
        )
        device = utils.select_device(ctx.request.device)
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        dtype = precision.compute_dtype
        default_device_map = self.resolve_device_map(ctx.request.device)
        revision = source.revision or requested_revision
        shared_kwargs = utils.build_hf_source_kwargs(
            source,
            trust_remote_code=cfg.trust_remote_code,
        )

        _patch_remote_minicpm_init(cfg.model_id, revision)

        with _suppress_legacy_minicpm_image_processor_register():
            processor = AutoProcessor.from_pretrained(
                source.load_arg,
                use_fast=False,
                **shared_kwargs,
            )
        load_kwargs, _placement = self.build_hf_load_kwargs(
            ctx,
            device,
            precision,
            dtype_key="dtype",
            default_device_map=default_device_map,
        )
        model = self.load_hf_pretrained_model(
            AutoModel.from_pretrained,
            precision,
            source.load_arg,
            **shared_kwargs,
            **load_kwargs,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            source.load_arg,
            **shared_kwargs,
        )
        model_device = next(model.parameters()).device
        return MiniCPMVRuntime(
            cfg=cfg,
            revision=revision,
            device=device,
            dtype=dtype,
            precision=precision,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            model_device=model_device,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: list[VLMJob],
        runtime: MiniCPMVRuntime,
        ctx: RuntimeContext[MiniCPMVOptions],
    ) -> BatchResult:
        chat_params = inspect.signature(runtime.model.chat).parameters
        ready_jobs, ready_images, warnings = self.load_batch_images(dataset, batch)

        if not ready_jobs:
            return BatchResult(modified_record_indices=[], warnings=warnings)

        common_kwargs = {}
        if "processor" in chat_params:
            common_kwargs["processor"] = runtime.processor
        if "tokenizer" in chat_params:
            common_kwargs["tokenizer"] = runtime.tokenizer
        if "max_new_tokens" in chat_params:
            common_kwargs["max_new_tokens"] = runtime.cfg.max_new_tokens
        if "sampling" in chat_params:
            common_kwargs["sampling"] = False
        if "stream" in chat_params:
            common_kwargs["stream"] = False
        if "enable_thinking" in chat_params:
            common_kwargs["enable_thinking"] = runtime.cfg.enable_thinking

        if len(ready_jobs) > 1:
            batch_kwargs = dict(common_kwargs)
            if "image" in chat_params:
                batch_kwargs["image"] = None
            batch_kwargs["msgs"] = [
                [{"role": "user", "content": [image, job.question]}]
                for job, image in zip(ready_jobs, ready_images)
            ]
            with torch.inference_mode():
                answers = runtime.model.chat(**batch_kwargs)
            if not isinstance(answers, (list, tuple)):
                raise TypeError(
                    f"MiniCPM-V batch chat returned {type(answers).__name__}, expected a list/tuple for {len(ready_jobs)} jobs."
                )
            if len(answers) != len(ready_jobs):
                raise ValueError(
                    f"MiniCPM-V batch chat returned {len(answers)} answers for {len(ready_jobs)} jobs."
                )
            return BatchResult(
                modified_record_indices=self.append_answers(dataset, ready_jobs, answers),
                warnings=warnings,
            )

        job = ready_jobs[0]
        image = ready_images[0]
        kwargs = dict(common_kwargs)
        if "image" in chat_params:
            kwargs["image"] = image
            kwargs["msgs"] = [{"role": "user", "content": job.question}]
        else:
            kwargs["msgs"] = [{"role": "user", "content": [image, job.question]}]

        with torch.inference_mode():
            answer = runtime.model.chat(**kwargs)
        return BatchResult(
            modified_record_indices=self.append_answers(dataset, [job], [answer]),
            warnings=warnings,
        )

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: MiniCPMVRuntime,
        ctx: RuntimeContext[MiniCPMVOptions],
    ) -> dict[str, object]:
        return self.build_vlm_meta(
            runtime=runtime,
            ctx=ctx,
            hf_model_id=runtime.cfg.model_id,
            precision=runtime.precision,
            extra={
                "max_new_tokens": runtime.cfg.max_new_tokens,
                "enable_thinking": runtime.cfg.enable_thinking,
                "revision": runtime.revision,
            },
        )


MODEL = MiniCPMVModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
