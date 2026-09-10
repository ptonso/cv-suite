from __future__ import annotations

import hashlib
import os
import pickle
import sys
import types
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any, Sequence

import torch
import torchvision.transforms as T
from PIL import Image

from cvsuite.common.core import FMRequest
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.bases.diffusion import (
    BaseGenerationBackend,
    load_repo_checkout,
    prepend_repo_to_syspath,
)
from cvsuite.common.fm.providers.bases.edit import (
    BaseEditGenerationModel,
    DEFAULT_EDIT_BACKEND_ALIASES,
    EditGenerationOptions,
)
from cvsuite.common.fm.providers.bases.gen import GeneratedImage

DEFAULT_MODEL_ID = "stdKonjac/DIM-4.6B-Edit"
DEFAULT_DESIGNER_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PARAMS = "4.6B"

# ---------------------------------------------------------------------------
# Patch helpers — replicated from the old edit_custom.py so that we keep
# them co-located with the backend that uses them.
# ---------------------------------------------------------------------------

_DIM_MODELING_REL_PATH = Path("models") / "modeling_dim.py"
_DIM_SANA_PIPELINE_REL_PATH = Path("models") / "sana_pipeline.py"

_DIM_FLASH_ATTN_NEEDLE = """import math
import os
import re
"""
_DIM_FLASH_ATTN_IMPORT_PATCH = """import importlib.util
import math
import os
import re
import sys
"""
_DIM_FLASH_ATTN_LOAD_NEEDLE = """            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.pretrained_model_name_or_path,
                torch_dtype="auto",
                attn_implementation='flash_attention_2',
            )
"""
_DIM_FLASH_ATTN_LOAD_PATCH = """            mllm_load_kwargs = {
                "torch_dtype": "auto",
                "attn_implementation": "flash_attention_2"
                if importlib.util.find_spec("flash_attn") is not None
                else "sdpa",
            }
            if os.environ.get("VT_DIM_MLLM_NF4") == "1":
                from transformers import BitsAndBytesConfig

                mllm_compute_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_available()
                    and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                    else torch.float16
                )
                mllm_load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=mllm_compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                mllm_load_kwargs.setdefault("device_map", os.environ.get("VT_DIM_MLLM_DEVICE_MAP", "auto"))
            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.pretrained_model_name_or_path,
                **mllm_load_kwargs,
            )
"""
_DIM_NF4_LOAD_NEEDLE = """            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.pretrained_model_name_or_path,
                **mllm_load_kwargs,
            )
"""
_DIM_NF4_LOAD_PATCH = """            if os.environ.get("VT_DIM_MLLM_NF4") == "1":
                from transformers import BitsAndBytesConfig

                mllm_compute_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_available()
                    and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                    else torch.float16
                )
                mllm_load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=mllm_compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                mllm_load_kwargs.setdefault("device_map", os.environ.get("VT_DIM_MLLM_DEVICE_MAP", "auto"))
            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.pretrained_model_name_or_path,
                **mllm_load_kwargs,
            )
"""
_DIM_FROM_PRETRAINED_NEEDLE = """    def from_pretrained(self, pretrained_model_name_or_path):
        state_dict = load_file(os.path.join(pretrained_model_name_or_path, "model.safetensors"))
        # remove text encoder (if any)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder.text_encoder.")}

        # load weights, need to check carefully
        info = self.load_state_dict(state_dict, strict=False)
        print(f'Load pretrained weights from {pretrained_model_name_or_path}\\n\\n{info}\\n\\n')

        assert not info.unexpected_keys, 'Only missing keys are allowed, got unexpected keys'
"""
_DIM_FROM_PRETRAINED_OLD_PATCH = """    def from_pretrained(self, pretrained_model_name_or_path):
        state_dict = load_file(os.path.join(pretrained_model_name_or_path, "model.safetensors"))
        # remove text encoder (if any)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder.text_encoder.")}
        if os.environ.get("VT_DIM_MLLM_NF4") == "1":
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("mllm.")}

        # load weights, need to check carefully
        info = self.load_state_dict(state_dict, strict=False)
        if os.environ.get("VT_DIM_MLLM_NF4") == "1":
            missing_keys = [key for key in info.missing_keys if not key.startswith("mllm.")]
            skipped_mllm_keys = [key for key in info.missing_keys if key.startswith("mllm.")]
            print(
                f"Load pretrained weights from {pretrained_model_name_or_path}\\n\\n"
                f"missing_keys={len(missing_keys)} skipped_nf4_mllm_keys={len(skipped_mllm_keys)} "
                f"unexpected_keys={len(info.unexpected_keys)}\\n\\n"
            )
            assert not missing_keys, f"Unexpected non-MLLM missing keys: {missing_keys}"
        else:
            print(f'Load pretrained weights from {pretrained_model_name_or_path}\\n\\n{info}\\n\\n')

        assert not info.unexpected_keys, 'Only missing keys are allowed, got unexpected keys'
"""
_DIM_FROM_PRETRAINED_PATCH = """    def from_pretrained(self, pretrained_model_name_or_path):
        state_dict = load_file(os.path.join(pretrained_model_name_or_path, "model.safetensors"))
        # remove text encoder (if any)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder.text_encoder.")}
        if os.environ.get("VT_DIM_MLLM_NF4_SKIP_RELOAD") == "1":
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("mllm.")}

        # load weights, need to check carefully
        info = self.load_state_dict(state_dict, strict=False)
        if os.environ.get("VT_DIM_MLLM_NF4_SKIP_RELOAD") == "1":
            missing_keys = [key for key in info.missing_keys if not key.startswith("mllm.")]
            premerged_mllm_keys = [key for key in info.missing_keys if key.startswith("mllm.")]
            print(
                f"Load pretrained weights from {pretrained_model_name_or_path}\\n\\n"
                f"missing_keys={len(missing_keys)} premerged_nf4_mllm_keys={len(premerged_mllm_keys)} "
                f"unexpected_keys={len(info.unexpected_keys)}\\n\\n"
            )
            assert not missing_keys, f"Unexpected non-MLLM missing keys: {missing_keys}"
        else:
            print(f'Load pretrained weights from {pretrained_model_name_or_path}\\n\\n{info}\\n\\n')

        assert not info.unexpected_keys, 'Only missing keys are allowed, got unexpected keys'
"""
_DIM_FROM_PRETRAINED_OLD_PRINT = """        print(f'Load pretrained weights from {pretrained_model_name_or_path}\\n\\n{info}\\n\\n')
"""
_DIM_FROM_PRETRAINED_SUMMARY_PRINT = """        if os.environ.get("VT_DIM_MLLM_NF4_SKIP_RELOAD") == "1":
            missing_keys = [key for key in info.missing_keys if not key.startswith("mllm.")]
            premerged_mllm_keys = [key for key in info.missing_keys if key.startswith("mllm.")]
            print(
                f"Load pretrained weights from {pretrained_model_name_or_path}\\n\\n"
                f"missing_keys={len(missing_keys)} premerged_nf4_mllm_keys={len(premerged_mllm_keys)} "
                f"unexpected_keys={len(info.unexpected_keys)}\\n\\n"
            )
            assert not missing_keys, f"Unexpected non-MLLM missing keys: {missing_keys}"
        else:
            print(f'Load pretrained weights from {pretrained_model_name_or_path}\\n\\n{info}\\n\\n')
"""
_DIM_GENERATE_SEED_NEEDLE = """        generator = torch.Generator(device=y.device).manual_seed(233)
"""
_DIM_GENERATE_SEED_PATCH = """        seed_value = int(os.environ.get("VT_DIM_SEED", "233"))
        generator = torch.Generator(device=y.device).manual_seed(seed_value)
"""
_DIM_DPM_CFG_NEEDLE = """                    # cfg_scale=guidance_scale,
                    cfg_scale=7.5,
"""
_DIM_DPM_CFG_PATCH = """                    cfg_scale=float(os.environ.get("VT_DIM_GUIDANCE_SCALE", "7.5")),
"""
_DIM_DPM_STEPS_NEEDLE = """                _latents_denoised = scheduler.sample(
                    z,
                    # steps=num_inference_steps,
                    steps=30,
"""
_DIM_DPM_STEPS_PATCH = """                _latents_denoised = scheduler.sample(
                    z,
                    steps=int(os.environ.get("VT_DIM_NUM_INFERENCE_STEPS", "30")),
"""
_DIM_AUTO_REPORT_MARKER = "VT_DIM_MLLM_AUTO_DEVICE_REPORT"
_DIM_AUTO_REPORT_NEEDLE = """            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.pretrained_model_name_or_path,
                **mllm_load_kwargs,
            )
"""
_DIM_AUTO_REPORT_PATCH = """            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.pretrained_model_name_or_path,
                **mllm_load_kwargs,
            )
            if (
                str(mllm_load_kwargs.get("device_map", "")) == "auto"
                and os.environ.get("VT_DIM_MLLM_AUTO_DEVICE_REPORT", "1") != "0"
            ):
                device_map = getattr(self.mllm, "hf_device_map", {}) or {}
                by_target = {}
                total_bytes = 0
                for name, param in self.mllm.named_parameters():
                    nbytes = int(param.numel()) * int(param.element_size())
                    target = None
                    best_key = ""
                    for key, value in device_map.items():
                        key = str(key)
                        if key == "" or name == key or name.startswith(key + "."):
                            if target is None or len(key) > len(best_key):
                                best_key = key
                                target = value
                    target = str(target if target is not None else getattr(param, "device", "unknown"))
                    by_target[target] = by_target.get(target, 0) + nbytes
                    total_bytes += nbytes
                def _fmt_mllm_bytes(value):
                    value = float(value)
                    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
                        if value < 1024.0 or unit == "TiB":
                            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
                        value /= 1024.0
                parts = [
                    f"{target} ~{_fmt_mllm_bytes(nbytes)} ({(nbytes / total_bytes * 100.0):.0f}%)"
                    for target, nbytes in sorted(by_target.items(), key=lambda item: item[1], reverse=True)
                ] if total_bytes else ["unavailable"]
                print("[dim-edit] MLLM auto device placement: " + "; ".join(parts), file=sys.stderr)
"""
_DIM_CUSTOM_AUTOTUNE_REL_PATHS = (
    Path("models")
    / "diffusion"
    / "model"
    / "nets"
    / "fastlinear"
    / "modules"
    / "triton_lite_mla_kernels"
    / "custom_autotune.py",
    Path("models")
    / "diffusion"
    / "model"
    / "nets"
    / "fastlinear"
    / "modules"
    / "utils"
    / "custom_autotune.py",
)
_DIM_AUTOTUNE_DEVICE_NEEDLE = """                torch.cuda.get_device_name(0).replace(" ", "_"),
"""
_DIM_AUTOTUNE_DEVICE_PATCH = """                (
                    torch.cuda.get_device_name(0).replace(" ", "_")
                    if torch.cuda.is_available()
                    else "cpu"
                ),
"""
_DIM_SANA_DEVICE_NEEDLE = """        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
_DIM_SANA_IMPORT_NEEDLE = """# SPDX-License-Identifier: Apache-2.0
import warnings
"""
_DIM_SANA_IMPORT_PATCH = """# SPDX-License-Identifier: Apache-2.0
import os
import warnings
"""
_DIM_SANA_DEVICE_PATCH = """        sana_init_device = os.environ.get("VT_DIM_SANA_INIT_DEVICE", "").strip()
        self.device = (
            torch.device(sana_init_device)
            if sana_init_device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
"""
_DIM_SANA_TEXT_BLOCK_NEEDLE = """        # 1. build vae and text encoder
        self.vae = self.build_vae(config.vae)
        self.tokenizer, self.text_encoder = self.build_text_encoder(config.text_encoder)

        # 2. build Sana model
        self.model = self.build_sana_model(config).to(self.device)

        # 3. pre-compute null embedding
        with torch.no_grad():
            null_caption_token = self.tokenizer(
                "", max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[
                0
            ]
"""
_DIM_SANA_TEXT_BLOCK_PATCH = """        # 1. build VAE on the chosen init device
        self.vae = self.build_vae(config.vae)

        # DIM edit inference supplies multimodal conditioning from the MLLM
        # and does not use Sana's standalone text encoder path.
        decoder_only = os.environ.get("VT_DIM_SANA_DECODER_ONLY", "0") == "1"
        if decoder_only:
            self.tokenizer = None
            self.text_encoder = None
            self.null_caption_embs = None
        else:
            self.tokenizer, self.text_encoder = self.build_text_encoder(config.text_encoder)

        # 2. build Sana model lazily so callers can choose final placement later
        self.model = self.build_sana_model(config)
        if os.environ.get("VT_DIM_SANA_LAZY_TO_DEVICE", "0") != "1":
            self.model = self.model.to(self.device)

        # 3. pre-compute null embedding when the text encoder is active
        if not decoder_only:
            with torch.no_grad():
                null_caption_token = self.tokenizer(
                    "", max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
                ).to(self.device)
                self.null_caption_embs = self.text_encoder(
                    null_caption_token.input_ids, null_caption_token.attention_mask
                )[0]
"""

warnings.filterwarnings(
    "ignore",
    message="Can't initialize NVML",
    category=UserWarning,
)


def _install_mmcv_compat() -> None:
    """Provide the small subset of mmcv that DIM needs during inference."""
    try:
        import importlib.util

        if importlib.util.find_spec("mmcv") is not None:
            return
    except Exception:
        pass
    if "mmcv" in sys.modules:
        return

    import json
    import torch.nn as nn
    import yaml

    class Config(dict):
        def __getattr__(self, name: str) -> Any:
            try:
                value = self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc
            if isinstance(value, dict) and not isinstance(value, Config):
                value = Config(value)
                self[name] = value
            return value

        def __setattr__(self, name: str, value: Any) -> None:
            self[name] = value

        @classmethod
        def fromfile(cls, filename: str) -> "Config":
            path = Path(filename)
            text = path.read_text(encoding="utf-8")
            if path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(text) or {}
            elif path.suffix == ".json":
                data = json.loads(text)
            else:
                namespace: dict[str, Any] = {}
                exec(compile(text, str(path), "exec"), {}, namespace)
                data = {key: value for key, value in namespace.items() if not key.startswith("_")}
            if not isinstance(data, dict):
                raise TypeError(f"Unsupported config payload in {filename}: expected mapping, got {type(data).__name__}.")
            return cls(data)

        def merge_from_dict(self, other: dict[str, Any]) -> None:
            for key, value in dict(other).items():
                if isinstance(value, dict) and isinstance(self.get(key), dict):
                    merged = Config(self[key])
                    merged.merge_from_dict(value)
                    self[key] = merged
                else:
                    self[key] = value

    def mkdir_or_exist(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def dump(obj: Any, filename: str) -> None:
        with open(filename, "wb") as handle:
            pickle.dump(obj, handle)

    def load(filename: str) -> Any:
        with open(filename, "rb") as handle:
            return pickle.load(handle)

    def build_from_cfg(
        cfg: dict[str, Any],
        registry: "Registry",
        default_args: dict[str, Any] | None = None,
    ) -> Any:
        if cfg is None:
            raise TypeError("cfg must be a mapping, got None.")
        args = dict(default_args or {})
        args.update(dict(cfg))
        obj_type = args.pop("type", None)
        if obj_type is None:
            raise KeyError("cfg must contain a 'type' entry.")
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(f"{obj_type!r} is not registered in {registry.name!r}.")
        elif callable(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f"Unsupported cfg type entry: {type(obj_type).__name__}.")
        return obj_cls(**args)

    class Registry:
        def __init__(self, name: str) -> None:
            self.name = name
            self._module_dict: dict[str, Any] = {}

        def get(self, key: str) -> Any:
            return self._module_dict.get(key)

        def register_module(self, module: Any | None = None, name: str | None = None):
            def _register(target: Any) -> Any:
                module_name = name or getattr(target, "__name__", type(target).__name__)
                self._module_dict[module_name] = target
                return target

            if module is not None:
                return _register(module)
            return _register

        def build(self, cfg: dict[str, Any], default_args: dict[str, Any] | None = None) -> Any:
            return build_from_cfg(cfg, self, default_args=default_args)

    def get_dist_info() -> tuple[int, int]:
        return 0, 1

    class DefaultOptimizerConstructor:
        def __init__(self, optimizer_cfg: dict[str, Any], paramwise_cfg: dict[str, Any] | None = None) -> None:
            self.optimizer_cfg = optimizer_cfg
            self.paramwise_cfg = paramwise_cfg or {}

        def __call__(self, model: Any) -> Any:
            return build_optimizer(model, self.optimizer_cfg)

    def build_optimizer(model: Any, optimizer_cfg: dict[str, Any]) -> Any:
        cfg = dict(optimizer_cfg)
        optim_type = cfg.pop("type", "AdamW")
        optim_cls = getattr(torch.optim, str(optim_type))
        return optim_cls(model.parameters(), **cfg)

    mmcv_mod = types.ModuleType("mmcv")
    mmcv_mod.Config = Config
    mmcv_mod.Registry = Registry
    mmcv_mod.build_from_cfg = build_from_cfg
    mmcv_mod.mkdir_or_exist = mkdir_or_exist
    mmcv_mod.dump = dump
    mmcv_mod.load = load

    runner_mod = types.ModuleType("mmcv.runner")
    runner_mod.get_dist_info = get_dist_info
    runner_mod.OPTIMIZER_BUILDERS = Registry("optimizer builder")
    runner_mod.OPTIMIZERS = Registry("optimizer")
    runner_mod.DefaultOptimizerConstructor = DefaultOptimizerConstructor
    runner_mod.build_optimizer = build_optimizer

    utils_mod = types.ModuleType("mmcv.utils")
    utils_mod._BatchNorm = nn.modules.batchnorm._BatchNorm
    utils_mod._InstanceNorm = nn.modules.instancenorm._InstanceNorm

    logging_mod = types.ModuleType("mmcv.utils.logging")
    logging_mod.logger_initialized = {}
    utils_mod.logging = logging_mod

    mmcv_mod.runner = runner_mod
    mmcv_mod.utils = utils_mod

    sys.modules["mmcv"] = mmcv_mod
    sys.modules["mmcv.runner"] = runner_mod
    sys.modules["mmcv.utils"] = utils_mod
    sys.modules["mmcv.utils.logging"] = logging_mod


def _install_pytz_compat() -> None:
    """Provide a minimal pytz.timezone() for DIM's logging helpers."""
    try:
        import importlib.util

        if importlib.util.find_spec("pytz") is not None:
            return
    except Exception:
        pass
    if "pytz" in sys.modules:
        return

    pytz_mod = types.ModuleType("pytz")
    pytz_mod.timezone = ZoneInfo
    sys.modules["pytz"] = pytz_mod


def _install_ipdb_compat() -> None:
    """DIM imports ipdb in Triton helpers, but inference does not need it."""
    try:
        import importlib.util

        if importlib.util.find_spec("ipdb") is not None:
            return
    except Exception:
        pass
    if "ipdb" in sys.modules:
        return

    ipdb_mod = types.ModuleType("ipdb")

    def _set_trace(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("ipdb.set_trace() is unavailable in the managed dim_edit runtime.")

    ipdb_mod.set_trace = _set_trace
    sys.modules["ipdb"] = ipdb_mod


def _decoder_weight_device(model: Any) -> torch.device:
    """Choose a decoder target device with the least MLLM pressure when possible."""
    fallback = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device_map = getattr(model.mllm, "hf_device_map", {}) or {}
    except Exception:
        device_map = {}
    if not isinstance(device_map, dict) or not device_map:
        return fallback

    by_target: dict[str, int] = {}
    total_cuda_targets = set()
    for target in device_map.values():
        target_text = str(target)
        if target_text.startswith("cuda") or target_text.isdigit():
            total_cuda_targets.add(target_text if target_text.startswith("cuda") else f"cuda:{target_text}")

    if not total_cuda_targets:
        return fallback

    for name, param in model.mllm.named_parameters():
        nbytes = int(param.numel()) * int(param.element_size())
        chosen = None
        best_key = ""
        for key, value in device_map.items():
            key = str(key)
            if key == "" or name == key or name.startswith(key + "."):
                if chosen is None or len(key) > len(best_key):
                    best_key = key
                    chosen = value
        target_text = str(chosen if chosen is not None else getattr(param, "device", "cpu"))
        if target_text.isdigit():
            target_text = f"cuda:{target_text}"
        if not target_text.startswith("cuda"):
            continue
        by_target[target_text] = by_target.get(target_text, 0) + nbytes

    for target in total_cuda_targets:
        by_target.setdefault(target, 0)

    best_target = min(by_target.items(), key=lambda item: item[1])[0]
    return torch.device(best_target)


def _ensure_dim_nf4_merged_mllm_snapshot(
    *,
    checkpoint_path: str,
    designer_path: str,
    cache_root: Path | str,
) -> str:
    """Create a local dense Qwen snapshot with DIM's MLLM finetune applied."""
    import importlib.util  # noqa: PLC0415

    from safetensors.torch import load_file  # noqa: PLC0415
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415

    checkpoint_dir = Path(checkpoint_path)
    designer_dir = Path(designer_path)
    cache_dir = Path(cache_root)

    fingerprint = hashlib.sha256()
    for path in (checkpoint_dir, designer_dir):
        fingerprint.update(str(path.resolve()).encode("utf-8"))
        for name in ("model.safetensors", "model.safetensors.index.json", "config.json"):
            candidate = path / name
            if candidate.exists():
                stat = candidate.stat()
                fingerprint.update(name.encode("utf-8"))
                fingerprint.update(str(stat.st_size).encode("utf-8"))
                fingerprint.update(str(stat.st_mtime_ns).encode("utf-8"))

    merged_dir = cache_dir / "merged_mllm_nf4" / fingerprint.hexdigest()[:16]
    if (merged_dir / "model.safetensors").exists() or (merged_dir / "model.safetensors.index.json").exists():
        return str(merged_dir)

    state_dict = load_file(str(checkpoint_dir / "model.safetensors"))
    mllm_state_dict = {
        key[len("mllm."):]: value
        for key, value in state_dict.items()
        if key.startswith("mllm.")
    }
    if not mllm_state_dict:
        raise RuntimeError(
            f"DIM checkpoint at {checkpoint_path} did not contain any mllm.* weights to merge."
        )

    print(
        f"[dim-edit] Preparing NF4 merged MLLM snapshot with {len(mllm_state_dict)} DIM finetune tensors...",
        file=sys.stderr,
    )
    dense_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(designer_dir),
        torch_dtype="auto",
        attn_implementation="flash_attention_2"
        if importlib.util.find_spec("flash_attn") is not None
        else "sdpa",
    )
    info = dense_model.load_state_dict(mllm_state_dict, strict=False)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(
            "Failed to merge DIM MLLM weights before NF4 quantization: "
            f"missing_keys={len(info.missing_keys)} unexpected_keys={len(info.unexpected_keys)}"
        )

    merged_dir.mkdir(parents=True, exist_ok=True)
    dense_model.save_pretrained(str(merged_dir), safe_serialization=True)
    print(f"[dim-edit] Saved NF4 merged MLLM snapshot to {merged_dir}", file=sys.stderr)
    return str(merged_dir)


def patch_dim_modeling(dim_root: Path | str) -> None:
    """Patch models/modeling_dim.py in the DIM repo checkout.

    Applies in-place patches to the upstream source:
    1. Flash-attention optional fallback to SDPA when flash_attn is absent.
    2. NF4 / bitsandbytes 4-bit quantisation for the MLLM (Qwen2.5-VL).
    3. Auto device-map placement report printed to stderr.
    4. Runtime sampler overrides for seed, guidance scale, and inference steps.
    """
    dim_root = Path(dim_root)
    target = dim_root / _DIM_MODELING_REL_PATH
    if not target.exists():
        raise FileNotFoundError(f"DIM patch target is missing: {target}")

    text = target.read_text(encoding="utf-8")
    patched = text

    # 1. Add importlib.util / sys imports
    if "importlib.util" not in patched:
        if _DIM_FLASH_ATTN_NEEDLE not in patched:
            raise RuntimeError(f"Unable to patch {target}: expected import block not found.")
        patched = patched.replace(_DIM_FLASH_ATTN_NEEDLE, _DIM_FLASH_ATTN_IMPORT_PATCH, 1)
    elif "import sys" not in patched:
        patched = patched.replace("import re\n", "import re\nimport sys\n", 1)

    # 2. Flash-attn conditional + NF4 load kwargs
    if _DIM_FLASH_ATTN_LOAD_NEEDLE in patched:
        patched = patched.replace(_DIM_FLASH_ATTN_LOAD_NEEDLE, _DIM_FLASH_ATTN_LOAD_PATCH, 1)
    elif "mllm_load_kwargs" not in patched:
        raise RuntimeError(f"Unable to patch {target}: expected FlashAttention load block not found.")

    if "VT_DIM_MLLM_NF4" not in patched:
        if _DIM_NF4_LOAD_NEEDLE not in patched:
            raise RuntimeError(f"Unable to patch {target}: expected MLLM load kwargs block not found.")
        patched = patched.replace(_DIM_NF4_LOAD_NEEDLE, _DIM_NF4_LOAD_PATCH, 1)

    # 3. Auto device placement report
    if _DIM_AUTO_REPORT_MARKER not in patched:
        if _DIM_AUTO_REPORT_NEEDLE not in patched:
            raise RuntimeError(f"Unable to patch {target}: expected MLLM load call not found.")
        patched = patched.replace(_DIM_AUTO_REPORT_NEEDLE, _DIM_AUTO_REPORT_PATCH, 1)

    # 4. Skip dense MLLM checkpoint reload only after NF4 used a pre-merged MLLM snapshot
    if "VT_DIM_MLLM_NF4_SKIP_RELOAD" in patched:
        pass
    elif _DIM_FROM_PRETRAINED_OLD_PATCH in patched:
        patched = patched.replace(_DIM_FROM_PRETRAINED_OLD_PATCH, _DIM_FROM_PRETRAINED_PATCH, 1)
    elif "state_dict = {k: v for k, v in state_dict.items() if not k.startswith(\"mllm.\")}" not in patched:
        if _DIM_FROM_PRETRAINED_NEEDLE not in patched:
            raise RuntimeError(f"Unable to patch {target}: expected DIM from_pretrained block not found.")
        patched = patched.replace(_DIM_FROM_PRETRAINED_NEEDLE, _DIM_FROM_PRETRAINED_PATCH, 1)
    elif _DIM_FROM_PRETRAINED_OLD_PRINT in patched and "premerged_nf4_mllm_keys" not in patched:
        patched = patched.replace(_DIM_FROM_PRETRAINED_OLD_PRINT, _DIM_FROM_PRETRAINED_SUMMARY_PRINT, 1)

    # 5. Runtime sampler overrides so the wrapper can control the actual DIM denoiser.
    if "VT_DIM_SEED" not in patched:
        if _DIM_GENERATE_SEED_NEEDLE in patched:
            patched = patched.replace(_DIM_GENERATE_SEED_NEEDLE, _DIM_GENERATE_SEED_PATCH, 1)
    if "VT_DIM_GUIDANCE_SCALE" not in patched:
        if _DIM_DPM_CFG_NEEDLE in patched:
            patched = patched.replace(_DIM_DPM_CFG_NEEDLE, _DIM_DPM_CFG_PATCH, 1)
    if "VT_DIM_NUM_INFERENCE_STEPS" not in patched:
        if _DIM_DPM_STEPS_NEEDLE in patched:
            patched = patched.replace(_DIM_DPM_STEPS_NEEDLE, _DIM_DPM_STEPS_PATCH, 1)

    # Fix upstream dtype key typo ("dtype" → "torch_dtype")
    patched = patched.replace(
        '                "dtype": "auto",\n                "attn_implementation":',
        '                "torch_dtype": "auto",\n                "attn_implementation":',
    )

    if patched != text:
        target.write_text(patched, encoding="utf-8")

    sana_pipeline_path = dim_root / _DIM_SANA_PIPELINE_REL_PATH
    if not sana_pipeline_path.exists():
        raise FileNotFoundError(f"DIM SanaPipeline patch target is missing: {sana_pipeline_path}")
    sana_text = sana_pipeline_path.read_text(encoding="utf-8")
    sana_patched = sana_text
    if "import os\n" not in sana_patched and _DIM_SANA_IMPORT_NEEDLE in sana_patched:
        sana_patched = sana_patched.replace(_DIM_SANA_IMPORT_NEEDLE, _DIM_SANA_IMPORT_PATCH, 1)
    if _DIM_SANA_DEVICE_NEEDLE in sana_patched:
        sana_patched = sana_patched.replace(_DIM_SANA_DEVICE_NEEDLE, _DIM_SANA_DEVICE_PATCH, 1)
    if _DIM_SANA_TEXT_BLOCK_NEEDLE in sana_patched:
        sana_patched = sana_patched.replace(_DIM_SANA_TEXT_BLOCK_NEEDLE, _DIM_SANA_TEXT_BLOCK_PATCH, 1)
    if sana_patched != sana_text:
        sana_pipeline_path.write_text(sana_patched, encoding="utf-8")

    for rel_path in _DIM_CUSTOM_AUTOTUNE_REL_PATHS:
        autotune_path = dim_root / rel_path
        if not autotune_path.exists():
            continue
        autotune_text = autotune_path.read_text(encoding="utf-8")
        if _DIM_AUTOTUNE_DEVICE_NEEDLE not in autotune_text:
            continue
        autotune_path.write_text(
            autotune_text.replace(_DIM_AUTOTUNE_DEVICE_NEEDLE, _DIM_AUTOTUNE_DEVICE_PATCH, 1),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------

def _build_image_transform(resolution: int) -> T.Compose:
    """Return the same transform that TosDatasetBase uses for raw/target images."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize(resolution),
        T.CenterCrop(resolution),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])


def _tensor_to_pil(tensor: torch.Tensor, height: int, width: int) -> Image.Image:
    """Convert a 1×3×H×W normalized tensor (value range −1…1) to a PIL image."""
    # Clamp and denormalise from [-1, 1] → [0, 1]
    img = tensor.squeeze(0).float().clamp(-1, 1)
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1)
    to_pil = T.ToPILImage()
    return to_pil(img)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class DIMEditBackend(BaseGenerationBackend):
    """In-process DIM edit backend.

    Loads the DIM-4.6B-Edit model (Qwen2.5-VL-3B MLLM + MLP projector +
    SANA-1.5 decoder) directly into the current process using the DIM repo
    source code from the managed venv's pkgs/DIM checkout.

    Supports:
    - ``--device auto``  →  ``device_map="auto"`` splits model layers across
      all available GPUs (and falls back to CPU/disk when needed).
    - ``--precision nf4`` → bitsandbytes NF4 4-bit quantisation applied to
      the MLLM (Qwen2.5-VL-3B) component.
    """

    default_model_id = DEFAULT_MODEL_ID
    default_designer_model_id = DEFAULT_DESIGNER_MODEL_ID
    default_num_inference_steps = 30
    default_guidance_scale = 7.5
    default_seed = 233

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        dataset: Any = None,
        ctx: Any = None,
        mode: str = "",
        model: Any = None,
    ) -> None:
        super().__init__(config=config, dataset=dataset, ctx=ctx, mode=mode, model=model)

        self._dim_root = load_repo_checkout(ctx, "DIM")
        patch_dim_modeling(self._dim_root)
        _install_mmcv_compat()
        _install_pytz_compat()
        _install_ipdb_compat()
        prepend_repo_to_syspath(self._dim_root)

        self._resolution = max(1, int(self.config.get("gen_resolution", 1024)))
        self._task_type = str(self.config.get("task_type", "MM-PAD"))
        self._max_condition_length = int(self.config.get("max_condition_length", 8192))
        self._with_latents_condition = bool(self.config.get("with_latents_condition", True))
        self._num_inference_steps = self.steps_for()
        guidance_scale = self.guidance_for()
        self._guidance_scale = self.default_guidance_scale if guidance_scale is None else guidance_scale
        self._seed = self.seed_for()
        self._sana_config = self.config.get(
            "sana_config",
            str(
                self._dim_root
                / "models"
                / "sana1-5_config"
                / "1024ms"
                / "Sana_1600M_1024px_allqknorm_bf16_lr2e5_channel_cond.yaml"
            ),
        )
        self.fm_meta_extra.update(
            {
                "num_inference_steps": self._num_inference_steps,
                "guidance_scale": self._guidance_scale,
                "seed": self._seed,
                "task_type": self._task_type,
                "gen_resolution": self._resolution,
            }
        )
        strength = self.strength_for()
        if strength is not None:
            self.fm_meta_extra["ignored_strength"] = strength
            warnings.warn(
                "`dim_edit` does not expose a continuous `strength` control in this inference path; "
                "the provided value will be ignored. Use `task_type=MM-PAD` for stronger source-image "
                "anchoring or `task_type=MM-NPAD` for looser source adherence.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Resolve HF snapshot paths
        designer_model_id = str(
            self.config.get("designer_model_id") or self.default_designer_model_id
        )
        self._designer_model_id = designer_model_id

        main_source = utils.materialize_hf_model_source(
            self.model_id,
            hub_dir=ctx.hub_dir,
            stage_dir=getattr(ctx, "stage_dir", None),
            caller_cwd=getattr(ctx, "caller_cwd", Path.cwd()),
        )
        designer_source = utils.materialize_hf_model_source(
            designer_model_id,
            hub_dir=ctx.hub_dir,
            stage_dir=getattr(ctx, "stage_dir", None),
            caller_cwd=getattr(ctx, "caller_cwd", Path.cwd()),
        )

        checkpoint_path = str(main_source.snapshot_path or main_source.load_arg)
        designer_path = str(designer_source.snapshot_path or designer_source.load_arg)

        self._model, self._processor = self._load_dim(
            checkpoint_path=checkpoint_path,
            designer_path=designer_path,
        )

    def _load_dim(self, *, checkpoint_path: str, designer_path: str):
        """Instantiate DIM, load weights, and wire up the designer."""
        # Set env vars consumed by the patched modeling_dim.py before import
        use_auto_device = utils.normalize_device_pref(self.ctx.request.device) == "auto"
        use_nf4 = self.precision.quantization_mode == "nf4"
        mllm_path = designer_path

        if use_nf4:
            os.environ["VT_DIM_MLLM_NF4"] = "1"
            mllm_path = _ensure_dim_nf4_merged_mllm_snapshot(
                checkpoint_path=checkpoint_path,
                designer_path=designer_path,
                cache_root=self._dim_root.parent,
            )
            os.environ["VT_DIM_MLLM_NF4_SKIP_RELOAD"] = "1"
        else:
            os.environ.pop("VT_DIM_MLLM_NF4", None)
            os.environ.pop("VT_DIM_MLLM_NF4_SKIP_RELOAD", None)

        if use_auto_device or use_nf4:
            os.environ["VT_DIM_MLLM_DEVICE_MAP"] = "auto"
        else:
            os.environ.pop("VT_DIM_MLLM_DEVICE_MAP", None)
        os.environ["VT_DIM_SANA_DECODER_ONLY"] = "1"
        os.environ["VT_DIM_SANA_LAZY_TO_DEVICE"] = "1"
        os.environ["VT_DIM_SANA_INIT_DEVICE"] = "cpu"

        # Late-import DIM classes (needs pkgs/DIM on sys.path)
        from models import ModelArguments  # type: ignore[import]  # noqa: PLC0415
        from models.modeling_dim import DIM  # type: ignore[import]  # noqa: PLC0415
        from transformers import AutoProcessor  # noqa: PLC0415

        model_args = ModelArguments(
            pretrained_model_name_or_path=mllm_path,
            model_name_or_path=checkpoint_path,
            condition_type="LMToken",
            max_condition_length=self._max_condition_length,
            with_latents_condition=self._with_latents_condition,
            text_only_condition=False,
            sana_config=self._sana_config,
            sana_pretrained="",
        )

        model = DIM(model_args)
        model.decoder.expand_context(max_condition_length=self._max_condition_length)
        if self._with_latents_condition:
            model.decoder.expand_channels()

        model.from_pretrained(checkpoint_path)

        decoder_target = self.device
        if use_auto_device or use_nf4:
            decoder_target = _decoder_weight_device(model)

        projector_dtype = model.decoder.weight_dtype
        if use_nf4:
            projector_dtype = self.precision.compute_dtype or model.decoder.weight_dtype

        model.decoder.model.eval().to(device=decoder_target, dtype=model.decoder.weight_dtype)
        model.decoder.vae.eval().to(device=decoder_target, dtype=model.decoder.vae_dtype)

        if use_nf4:
            # MLLM was loaded quantised; move projector + decoder to GPU
            model.projector.to(device=decoder_target, dtype=projector_dtype)
            model.decoder.to(decoder_target)
        elif use_auto_device:
            # MLLM was loaded with device_map="auto"; move remaining modules
            model.projector.to(device=decoder_target, dtype=projector_dtype)
            model.decoder.to(decoder_target)
        else:
            model.mllm.to(self.device)
            model.projector.to(device=self.device, dtype=projector_dtype)
            model.decoder.to(self.device)
            if self.precision.compute_dtype is not None:
                model.decoder.model.to(self.precision.compute_dtype)

        model.eval()

        # Wire the MLLM as its own designer (no external VLM needed)
        processor = AutoProcessor.from_pretrained(designer_path, padding_side="right")
        model.designer = model.mllm
        model.designer_name = self._designer_model_id
        model.designer_processor = processor

        return model, processor

    # ----------------------------------------------------------------- inference

    @staticmethod
    def _build_processor_inputs(
        processor: Any,
        image_path: str,
        prompt: str,
        resolution: int,
        task_type: str,
        force_resolution: bool = True,
    ) -> dict[str, Any]:
        """Build the tokenised inputs that DIM's generate() expects.

        This replicates the essential logic of TosDatasetBase.getitem() /
        TosDatasetEdit.getitem() without depending on mmcv or datasets.
        """
        from utils.vision_process import process_vision_info  # type: ignore[import]  # noqa: PLC0415

        contain_image = task_type not in ("T2I-NPAD", "T2I-PAD")
        contain_text = task_type not in ("IR-NPAD", "IR-PAD")

        content: list[dict[str, Any]] = []
        if contain_image:
            img_entry: dict[str, Any] = {"type": "image", "image": image_path}
            if force_resolution:
                img_entry["resized_height"] = resolution
                img_entry["resized_width"] = resolution
            content.append(img_entry)
        if contain_text:
            content.append({"type": "text", "text": prompt})
        else:
            content.append({"type": "text", "text": ""})

        conv = [{"role": "user", "content": content}]

        image_inputs, video_inputs = process_vision_info(conv)
        texts = processor.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False, return_tensors="pt"
        )
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        # Build user_labels (mark the user turn tokens; rest stays −100)
        input_ids = inputs.input_ids
        _, L = input_ids.shape

        im_start_id = processor.tokenizer("<|im_start|>").input_ids[0]
        im_end_id = processor.tokenizer("<|im_end|>").input_ids[0]
        user_id = processor.tokenizer("user").input_ids[0]
        newline_id = processor.tokenizer("\n").input_ids[0]

        user_labels = torch.full_like(input_ids, -100, dtype=torch.long)
        user_begin, user_end = [], []
        for idx in range(3, L):
            if (
                input_ids[0, idx - 3] == im_start_id
                and input_ids[0, idx - 2] == user_id
                and input_ids[0, idx - 1] == newline_id
            ):
                user_begin.append(idx)
            if input_ids[0, idx] == im_end_id and len(user_begin) == len(user_end) + 1:
                user_end.append(idx)

        for s, e in zip(user_begin, user_end):
            user_labels[0, s:e] = input_ids[0, s:e]

        inputs["user_labels"] = [user_labels]

        # Raw image tensor (gen_transform equivalent)
        gen_transform = _build_image_transform(resolution)
        raw_img = Image.open(image_path).convert("RGB")
        inputs["raw_images"] = [gen_transform(raw_img)]

        # raw_captions
        inputs["raw_captions"] = [prompt if contain_text else ""]

        # target_images: PAD task types reuse the source image as latent condition
        if "PAD" in task_type:
            inputs["target_images"] = [gen_transform(raw_img)]
        else:
            inputs["target_images"] = None

        return dict(inputs)

    def _move_inputs_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Move tensor inputs to the appropriate compute device."""
        # When device_map="auto" the MLLM parameters may span multiple devices;
        # we send token tensors to the first MLLM layer's device and image
        # tensors to the decoder's device.
        try:
            mllm_device = next(self._model.mllm.parameters()).device
            if mllm_device.type == "meta":
                mllm_device = self.device
        except StopIteration:
            mllm_device = self.device

        try:
            decoder_device = next(self._model.decoder.parameters()).device
            if decoder_device.type == "meta":
                decoder_device = self.device
        except StopIteration:
            decoder_device = self.device

        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # Token/attention tensors → MLLM device
                moved[key] = value.to(mllm_device)
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        # Image tensors → decoder device
                        new_list.append(item.to(decoder_device))
                    else:
                        new_list.append(item)
                moved[key] = new_list
            else:
                moved[key] = value
        return moved

    def seed_for(self) -> int:
        value = self.config.get("seed")
        if value is None:
            return self.default_seed
        return int(value)

    @contextmanager
    def _dim_runtime_env(self):
        overrides = {
            "VT_DIM_NUM_INFERENCE_STEPS": str(self._num_inference_steps),
            "VT_DIM_GUIDANCE_SCALE": str(self._guidance_scale),
            "VT_DIM_SEED": str(self._seed),
        }
        previous = {key: os.environ.get(key) for key in overrides}
        try:
            for key, value in overrides.items():
                os.environ[key] = value
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _run_single_edit(self, image_path: str, prompt: str) -> Image.Image:
        # 1. Get CoT from designer (MLLM used as its own designer)
        cot_prompt = self._model.get_cot_from_designer(
            image_path=image_path,
            instruction=prompt,
        )

        # 2. Build tokenised inputs
        inputs = self._build_processor_inputs(
            processor=self._processor,
            image_path=image_path,
            prompt=cot_prompt,
            resolution=self._resolution,
            task_type=self._task_type,
            force_resolution=True,
        )
        inputs = self._move_inputs_to_device(inputs)

        # 3. Generate
        with torch.inference_mode():
            with self._dim_runtime_env():
                gen_tensor = self._model.generate(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    task_type=self._task_type,
                )

        # gen_tensor: 1×3×H×W, value range [-1, 1]
        return _tensor_to_pil(gen_tensor, self._resolution, self._resolution)

    def generate_batch(self, *, jobs: Sequence[Any], **_kwargs: Any) -> list[GeneratedImage]:
        outputs: list[GeneratedImage] = []
        for job in jobs:
            pil_image = self._run_single_edit(
                image_path=str(job.source_image_path),
                prompt=job.prompt,
            )
            outputs.append(GeneratedImage(image=pil_image))
        return outputs


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DIMEditOptions(EditGenerationOptions):
    backend: str | None = "dim-edit"
    model_id: str | None = DEFAULT_MODEL_ID
    designer_model_id: str | None = DEFAULT_DESIGNER_MODEL_ID
    seed: int | None = 233
    max_condition_length: int | None = 8192
    gen_resolution: int | None = 1024
    task_type: str | None = "MM-PAD"
    num_inference_steps: int | None = 30
    guidance_scale: float | None = 7.5
    width: int | None = 1024
    height: int | None = 1024


class DIMEditModel(BaseEditGenerationModel[DIMEditOptions]):
    description = "DIM image-edit wrapper (in-process, stdKonjac/DIM-4.6B-Edit)."
    options_cls = DIMEditOptions
    supports_nf4_precision = True
    supports_managed_auto_device = True
    backend_aliases = {
        **DEFAULT_EDIT_BACKEND_ALIASES,
        "dim-edit": "cvsuite.common.fm.providers.edit.dim_edit:DIMEditBackend",
        "dim_edit": "cvsuite.common.fm.providers.edit.dim_edit:DIMEditBackend",
    }

    def validate_auto_device_request(self, request: FMRequest, *, require_supported: bool = True) -> None:
        if utils.normalize_device_pref(request.device) != "auto":
            return
        utils.require_cuda_available("cuda")
        if getattr(request, "max_gpu_memory", None):
            raise RuntimeError("`--max-gpu-memory` is not supported for model 'dim_edit'.")


MODEL = DIMEditModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
