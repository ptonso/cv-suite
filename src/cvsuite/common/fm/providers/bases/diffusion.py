from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from PIL import Image

from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.bases.gen import GeneratedImage
from cvsuite.common.fm.providers.bases.vlm import build_hf_placement_meta, resolve_hf_load_placement


def _signature_accepts_kwargs(target: Any) -> bool:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def filter_supported_kwargs(target: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if _signature_accepts_kwargs(target):
        return dict(kwargs)
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return dict(kwargs)
    return {name: value for name, value in kwargs.items() if name in signature.parameters}


def extract_images(result: Any) -> list[Image.Image]:
    images = getattr(result, "images", result)
    if isinstance(images, Image.Image):
        return [images]
    if isinstance(images, list) and all(isinstance(item, Image.Image) for item in images):
        return list(images)
    raise TypeError("Generation backend expected a diffusers-like result with PIL images.")


def _diffusers_device_map(device_map: object) -> str | None:
    if device_map == "auto":
        return "balanced"
    if isinstance(device_map, str):
        return device_map
    return None


def load_repo_checkout(ctx, repo_name: str) -> Path:
    repo_root = ctx.model_cache / "pkgs" / repo_name
    if not repo_root.exists():
        raise RuntimeError(
            f"Expected cached repo checkout at {repo_root}. "
            f"Rebuild the managed venv for model {ctx.model_name!r} to populate upstream sources."
        )
    return repo_root


class BaseGenerationBackend:
    default_model_id: str = ""
    default_width: int = 1024
    default_height: int = 1024
    default_num_inference_steps: int = 28
    default_guidance_scale: float = 4.5

    def __init__(self, *, config: dict[str, Any] | None = None, dataset=None, ctx=None, mode: str = "", model=None) -> None:
        if model is None:
            raise TypeError("Generation backends require the invoking FM wrapper instance.")
        self.model_wrapper = model
        self.config = dict(config or {})
        self.dataset = dataset
        self.ctx = ctx
        self.mode = str(mode or "")
        self.device = utils.select_device(ctx.request.device)
        self.precision = model.resolve_runtime_precision(dataset, ctx, self.device)
        self.model_id = str(self.config.get("model_id") or self.default_model_id).strip()
        self.fm_meta_extra = self.build_common_fm_meta()

    def build_common_fm_meta(self) -> dict[str, object]:
        meta: dict[str, object] = {
            "hf_model": self.model_id,
        }
        meta.update(utils.build_precision_meta(self.precision))
        return meta

    def _numeric(self, key: str, default: int | float | None) -> int | float | None:
        value = self.config.get(key, default)
        if value is None:
            return None
        return value

    def width_for(self, image: Image.Image | None = None) -> int:
        value = self._numeric("width", image.width if image is not None else self.default_width)
        return max(1, int(value or self.default_width))

    def height_for(self, image: Image.Image | None = None) -> int:
        value = self._numeric("height", image.height if image is not None else self.default_height)
        return max(1, int(value or self.default_height))

    def steps_for(self) -> int:
        value = self._numeric("num_inference_steps", self.default_num_inference_steps)
        return max(1, int(value or self.default_num_inference_steps))

    def guidance_for(self) -> float | None:
        value = self._numeric("guidance_scale", self.default_guidance_scale)
        if value is None:
            return None
        return float(value)

    def strength_for(self) -> float | None:
        value = self.config.get("strength")
        if value is None:
            return None
        return float(value)

    def negative_prompt(self) -> str | None:
        value = self.config.get("negative_prompt")
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def generator_for(self, job) -> torch.Generator | None:
        seed_value = self.config.get("seed")
        if seed_value is None:
            return None
        base_seed = int(seed_value)
        offset = int(getattr(job, "prompt_index", 0))
        if self.device.type == "cuda" and torch.cuda.is_available():
            return torch.Generator(device="cuda").manual_seed(base_seed + offset)
        return torch.Generator().manual_seed(base_seed + offset)

    def add_config_call_kwargs(self, kwargs: dict[str, Any], **converters: Callable[[Any], Any]) -> None:
        for key, converter in converters.items():
            value = self.config.get(key)
            if value is not None:
                kwargs[key] = converter(value)


class BaseDiffusersGenerationBackend(BaseGenerationBackend):
    pipeline_class_name: str = "DiffusionPipeline"
    diffusers_config_ignored_attrs: Mapping[str, Sequence[str]] = {}
    diffusers_execution_device_components: Mapping[str, str] = {}
    managed_auto_diffusers_components: Mapping[str, str] = {}
    managed_auto_transformers_components: Mapping[str, str] = {}
    managed_auto_cuda_runtime_reserve_fraction: float = 0.0
    managed_auto_cuda_runtime_reserve_bytes: int = 0

    def __init__(self, *, config: dict[str, Any] | None = None, dataset=None, ctx=None, mode: str = "", model=None) -> None:
        super().__init__(config=config, dataset=dataset, ctx=ctx, mode=mode, model=model)
        self.source = utils.materialize_hf_model_source(
            self.model_id,
            hub_dir=ctx.hub_dir,
            stage_dir=getattr(ctx, "stage_dir", None),
            caller_cwd=getattr(ctx, "caller_cwd", Path.cwd()),
        )
        self.pipeline, self.placement = self.load_pipeline()
        self.fm_meta_extra.update(self.build_diffusers_fm_meta())

    def build_diffusers_fm_meta(self) -> dict[str, object]:
        meta = dict(build_hf_placement_meta(self.ctx, self.placement))
        meta["pipeline_class"] = self.pipeline.__class__.__name__
        return meta

    def pipeline_class(self):
        import diffusers

        try:
            return getattr(diffusers, self.pipeline_class_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"Installed diffusers package does not expose {self.pipeline_class_name!r}; "
                f"rebuild the managed venv for model {self.ctx.model_name!r}."
            ) from exc

    def patch_diffusers_config_compat(self) -> None:
        if not self.diffusers_config_ignored_attrs and not self.diffusers_execution_device_components:
            return

        import diffusers

        for class_name, attrs in self.diffusers_config_ignored_attrs.items():
            cls = getattr(diffusers, class_name, None)
            if cls is None:
                continue
            ignored = set(getattr(cls, "_cvsuite_ignored_config_attrs", ()) or ())
            ignored.update(attrs)
            cls._cvsuite_ignored_config_attrs = frozenset(ignored)
            if hasattr(cls, "_cvsuite_original_extract_init_dict"):
                continue

            original_extract_init_dict = cls.extract_init_dict

            def _extract_init_dict_compat(
                inner_cls,
                config_dict,
                _original_extract_init_dict=original_extract_init_dict,
                **kwargs,
            ):
                ignored_attrs = getattr(inner_cls, "_cvsuite_ignored_config_attrs", frozenset())
                if isinstance(config_dict, dict) and ignored_attrs:
                    config_dict = {key: value for key, value in config_dict.items() if key not in ignored_attrs}
                return _original_extract_init_dict(config_dict, **kwargs)

            cls._cvsuite_original_extract_init_dict = original_extract_init_dict
            cls.extract_init_dict = classmethod(_extract_init_dict_compat)

        for class_name, component_name in self.diffusers_execution_device_components.items():
            cls = getattr(diffusers, class_name, None)
            if cls is None or hasattr(cls, "_cvsuite_original_execution_device"):
                continue

            original_execution_device = getattr(cls, "_execution_device", None)

            def _execution_device_compat(
                pipeline,
                _component_name=component_name,
                _original_execution_device=original_execution_device,
            ):
                component = getattr(pipeline, _component_name, None)
                if isinstance(component, torch.nn.Module):
                    try:
                        return next(component.parameters()).device
                    except StopIteration:
                        pass
                    component_device = getattr(component, "device", None)
                    if component_device is not None:
                        return torch.device(component_device)
                if isinstance(_original_execution_device, property):
                    return _original_execution_device.fget(pipeline)
                return torch.device("cpu")

            cls._cvsuite_original_execution_device = original_execution_device
            cls._execution_device = property(_execution_device_compat)

    def placement_load_kwargs(self) -> tuple[dict[str, Any], object]:
        placement = resolve_hf_load_placement(
            self.ctx.request,
            device=self.device,
            stage_dir=getattr(self.ctx, "stage_dir", None),
            model_cache=getattr(self.ctx, "model_cache", None) or self.ctx.weights_dir or Path.cwd(),
            include_default_device_map=True,
            default_device_map=None,
            default_offload_folder=True,
        )
        kwargs: dict[str, Any] = {}
        device_map = _diffusers_device_map(placement.device_map)
        if device_map is not None:
            kwargs["device_map"] = device_map
        if "device_map" in kwargs and placement.max_memory is not None:
            kwargs["max_memory"] = placement.max_memory
        if "device_map" in kwargs and placement.offload_folder is not None:
            kwargs["offload_folder"] = placement.offload_folder
        return kwargs, placement

    def _move_pipeline(self, pipeline: Any, placement) -> Any:
        if getattr(placement, "managed_device_map_active", False):
            return pipeline
        if self.precision.quantization_mode is None and hasattr(pipeline, "to"):
            pipeline.to(self.device)
        return pipeline

    @staticmethod
    def _cuda_index_from_memory_key(key: Any) -> int | None:
        if isinstance(key, int):
            return key
        text = str(key)
        if text.isdigit():
            return int(text)
        if text.startswith("cuda:") and text[5:].isdigit():
            return int(text[5:])
        return None

    @staticmethod
    def _memory_budget_to_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        try:
            from accelerate.utils import convert_file_size_to_int

            return int(convert_file_size_to_int(str(value)))
        except Exception:
            return None

    def _managed_component_max_memory(self, placement: Any) -> dict[Any, Any] | None:
        if placement.max_memory is None:
            return None
        max_memory = dict(placement.max_memory)
        reserve_fraction = max(0.0, float(self.managed_auto_cuda_runtime_reserve_fraction))
        reserve_floor = max(0, int(self.managed_auto_cuda_runtime_reserve_bytes))
        apply_runtime_reserve = getattr(placement, "memory_budget_source", "") != "explicit"
        for key, value in list(max_memory.items()):
            cuda_index = self._cuda_index_from_memory_key(key)
            if cuda_index is None:
                continue
            budget = self._memory_budget_to_int(value)
            if budget is None:
                continue
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_index)
            except Exception:
                continue
            runtime_reserve = 0
            if apply_runtime_reserve:
                runtime_reserve = max(int(total_bytes * reserve_fraction), reserve_floor)
            available_bytes = max(1, int(free_bytes) - int(runtime_reserve))
            max_memory[key] = max(1, min(int(budget), available_bytes))
        return max_memory

    def _managed_component_load_kwargs(self, placement: Any, component_name: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"device_map": "auto"}
        max_memory = self._managed_component_max_memory(placement)
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        if placement.offload_folder is not None:
            offload_folder = Path(placement.offload_folder) / str(component_name)
            offload_folder.mkdir(parents=True, exist_ok=True)
            kwargs["offload_folder"] = str(offload_folder)
        return kwargs

    def prepare_managed_auto_pipeline(self, pipeline: Any) -> Any:
        return pipeline

    def load_managed_auto_component_pipeline(self, placement: Any):
        self.patch_diffusers_config_compat()
        pipeline_cls = self.pipeline_class()
        t_dtype = self.precision.compute_dtype

        import diffusers
        import transformers

        components: dict[str, Any] = {}
        for component_name, class_name in self.managed_auto_diffusers_components.items():
            component_cls = getattr(diffusers, class_name)
            components[component_name] = component_cls.from_pretrained(
                self.source.load_arg,
                subfolder=component_name,
                **self._managed_component_load_kwargs(placement, component_name),
                **utils.build_hf_source_kwargs(self.source, torch_dtype=t_dtype),
            )
        for component_name, class_name in self.managed_auto_transformers_components.items():
            component_cls = getattr(transformers, class_name)
            components[component_name] = component_cls.from_pretrained(
                self.source.load_arg,
                subfolder=component_name,
                **self._managed_component_load_kwargs(placement, component_name),
                **utils.build_hf_source_kwargs(self.source, torch_dtype=t_dtype),
            )

        pipeline = pipeline_cls.from_pretrained(
            self.source.load_arg,
            **components,
            **utils.build_hf_source_kwargs(self.source, torch_dtype=t_dtype),
        )
        pipeline = self.prepare_managed_auto_pipeline(pipeline)
        self.model_wrapper.emit_hf_auto_device_report(pipeline, placement)
        return pipeline, placement

    def load_standard_pipeline(self):
        self.patch_diffusers_config_compat()
        pipeline_cls = self.pipeline_class()
        load_kwargs, placement = self.placement_load_kwargs()
        if (
            getattr(placement, "managed_device_map_active", False)
            and (self.managed_auto_diffusers_components or self.managed_auto_transformers_components)
        ):
            return self.load_managed_auto_component_pipeline(placement)
        pipeline = pipeline_cls.from_pretrained(
            self.source.load_arg,
            **utils.build_hf_source_kwargs(
                self.source,
                torch_dtype=self.precision.compute_dtype,
                **load_kwargs,
            ),
        )
        self.model_wrapper.emit_hf_auto_device_report(pipeline, placement)
        return self._move_pipeline(pipeline, placement), placement

    def load_quantized_pipeline(self):
        raise RuntimeError(f"Backend {self.__class__.__name__} does not implement NF4 loading.")

    def load_pipeline(self):
        if self.precision.quantization_mode == "nf4":
            return self.load_quantized_pipeline()
        return self.load_standard_pipeline()

    def call_pipeline(self, **kwargs: Any) -> Image.Image:
        call_kwargs = filter_supported_kwargs(self.pipeline.__call__, kwargs)
        result = self.pipeline(**call_kwargs)
        return extract_images(result)[0]


class BaseDiffusersCreateBackend(BaseDiffusersGenerationBackend):
    def build_call_kwargs(self, job) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": job.prompt,
            "width": self.width_for(),
            "height": self.height_for(),
            "num_inference_steps": self.steps_for(),
            "generator": self.generator_for(job),
        }
        guidance_scale = self.guidance_for()
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        negative_prompt = self.negative_prompt()
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        self.add_config_call_kwargs(kwargs, max_sequence_length=int, true_cfg_scale=float)
        return kwargs

    def generate_batch(self, *, jobs: Sequence[Any], **_kwargs: Any):
        return [GeneratedImage(image=self.call_pipeline(**self.build_call_kwargs(job))) for job in jobs]


class BaseDiffusersEditBackend(BaseDiffusersGenerationBackend):
    def build_call_kwargs(self, job, image: Image.Image) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": job.prompt,
            "image": image,
            "width": self.width_for(image),
            "height": self.height_for(image),
            "num_inference_steps": self.steps_for(),
            "generator": self.generator_for(job),
        }
        guidance_scale = self.guidance_for()
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        strength = self.strength_for()
        if strength is not None:
            kwargs["strength"] = strength
        negative_prompt = self.negative_prompt()
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        self.add_config_call_kwargs(
            kwargs,
            image_guidance_scale=float,
            max_sequence_length=int,
            true_cfg_scale=float,
        )
        return kwargs

    def generate_batch(self, *, jobs: Sequence[Any], **_kwargs: Any):
        outputs: list[GeneratedImage] = []
        for job in jobs:
            with Image.open(job.source_image_path) as source:
                image = source.convert("RGB")
            outputs.append(GeneratedImage(image=self.call_pipeline(**self.build_call_kwargs(job, image))))
        return outputs


def build_flux_nf4_pipeline(
    *,
    backend: BaseDiffusersGenerationBackend,
    pipeline_class_name: str = "DiffusionPipeline",
) -> tuple[Any, object]:
    if not utils.optional_dependency_available("bitsandbytes"):
        backend.model_wrapper.ensure_optional_dependency(
            package_name="bitsandbytes",
            ctx=backend.ctx,
            reason="NF4 quantization",
        )
    try:
        utils.ensure_bitsandbytes_quant_parameter_compat()
    except Exception as exc:
        raise RuntimeError("bitsandbytes is available, but quantized parameter compatibility could not be prepared.") from exc

    import diffusers
    from diffusers import AutoModel, BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig, T5EncoderModel

    pipeline_cls = getattr(diffusers, pipeline_class_name)
    t_dtype = backend.precision.compute_dtype
    transformers_qcfg = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=t_dtype,
        bnb_4bit_use_double_quant=True,
    )
    diffusers_qcfg = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=t_dtype,
        bnb_4bit_use_double_quant=True,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        backend.source.load_arg,
        subfolder="text_encoder_2",
        quantization_config=transformers_qcfg,
        **utils.build_hf_source_kwargs(backend.source, torch_dtype=t_dtype),
    )
    transformer = AutoModel.from_pretrained(
        backend.source.load_arg,
        subfolder="transformer",
        quantization_config=diffusers_qcfg,
        **utils.build_hf_source_kwargs(backend.source, torch_dtype=t_dtype),
    )
    pipeline = pipeline_cls.from_pretrained(
        backend.source.load_arg,
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        **utils.build_hf_source_kwargs(backend.source, torch_dtype=t_dtype),
    )
    if hasattr(pipeline, "to"):
        pipeline.to(backend.device)
    return pipeline, None


def build_diffusers_nf4_pipeline(
    *,
    backend: BaseDiffusersGenerationBackend,
    pipeline_class_name: str = "DiffusionPipeline",
    diffusers_components: Mapping[str, str] | None = None,
    transformers_components: Mapping[str, str] | None = None,
) -> tuple[Any, object]:
    if not utils.optional_dependency_available("bitsandbytes"):
        backend.model_wrapper.ensure_optional_dependency(
            package_name="bitsandbytes",
            ctx=backend.ctx,
            reason="NF4 quantization",
        )
    try:
        utils.ensure_bitsandbytes_quant_parameter_compat()
    except Exception as exc:
        raise RuntimeError("bitsandbytes is available, but quantized parameter compatibility could not be prepared.") from exc

    import diffusers
    import transformers
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

    backend.patch_diffusers_config_compat()
    pipeline_cls = getattr(diffusers, pipeline_class_name)
    t_dtype = backend.precision.compute_dtype
    transformers_qcfg = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=t_dtype,
        bnb_4bit_use_double_quant=True,
    )
    diffusers_qcfg = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=t_dtype,
        bnb_4bit_use_double_quant=True,
    )
    components: dict[str, Any] = {}
    for component_name, class_name in (transformers_components or {}).items():
        component_cls = getattr(transformers, class_name)
        components[component_name] = component_cls.from_pretrained(
            backend.source.load_arg,
            subfolder=component_name,
            quantization_config=transformers_qcfg,
            **utils.build_hf_source_kwargs(backend.source, torch_dtype=t_dtype),
        )
    for component_name, class_name in (diffusers_components or {}).items():
        component_cls = getattr(diffusers, class_name)
        components[component_name] = component_cls.from_pretrained(
            backend.source.load_arg,
            subfolder=component_name,
            quantization_config=diffusers_qcfg,
            **utils.build_hf_source_kwargs(backend.source, torch_dtype=t_dtype),
        )
    pipeline = pipeline_cls.from_pretrained(
        backend.source.load_arg,
        **components,
        **utils.build_hf_source_kwargs(backend.source, torch_dtype=t_dtype),
    )
    if hasattr(pipeline, "to"):
        pipeline.to(backend.device)
    return pipeline, None


def parse_json_output_images(stdout: str) -> list[GeneratedImage]:
    rows = json.loads(stdout)
    if not isinstance(rows, list):
        raise RuntimeError("Custom generation backend expected a JSON list of output image paths.")
    outputs: list[GeneratedImage] = []
    for item in rows:
        outputs.append(GeneratedImage(path=Path(str(item))))
    return outputs


def prepend_repo_to_syspath(repo_root: Path) -> None:
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)


def current_python_bin() -> str:
    return sys.executable or os.environ.get("PYTHON", "python")
