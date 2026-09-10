from __future__ import annotations

import inspect
from functools import wraps
from typing import Any

import torch


class QwenImageDiffusersMixin:
    diffusers_config_ignored_attrs = {"QwenImageTransformer2DModel": ("pooled_projection_dim",)}
    managed_auto_diffusers_components = {"transformer": "QwenImageTransformer2DModel"}
    managed_auto_transformers_components = {"text_encoder": "Qwen2_5_VLForConditionalGeneration"}
    managed_auto_cuda_runtime_reserve_fraction = 0.25
    managed_auto_cuda_runtime_reserve_bytes = 4 * 1024**3

    @staticmethod
    def _module_parameter_device(module: Any) -> torch.device | None:
        if not hasattr(module, "parameters"):
            return None
        try:
            for param in module.parameters():
                device = getattr(param, "device", None)
                if device is not None and str(device) != "meta":
                    return torch.device(device)
        except Exception:
            return None
        return None

    @classmethod
    def _text_encoder_input_device(cls, pipeline: Any) -> torch.device | None:
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is None:
            return None
        if hasattr(text_encoder, "get_input_embeddings"):
            try:
                device = cls._module_parameter_device(text_encoder.get_input_embeddings())
                if device is not None:
                    return device
            except Exception:
                pass
        return cls._module_parameter_device(text_encoder)

    @classmethod
    def patch_prompt_encoder_input_device(cls, pipeline_cls: Any) -> None:
        if getattr(pipeline_cls, "_cvsuite_qwen_prompt_device_patch", False):
            return
        original = getattr(pipeline_cls, "_get_qwen_prompt_embeds", None)
        if original is None:
            return
        signature = inspect.signature(original)

        @wraps(original)
        def _get_qwen_prompt_embeds_compat(pipeline, *args, **kwargs):
            bound = signature.bind(pipeline, *args, **kwargs)
            requested_device = bound.arguments.get("device") or getattr(pipeline, "_execution_device", None)
            input_device = cls._text_encoder_input_device(pipeline)
            if input_device is not None:
                bound.arguments["device"] = input_device
            prompt_embeds, encoder_attention_mask = original(*bound.args, **bound.kwargs)
            if requested_device is not None:
                prompt_embeds = prompt_embeds.to(device=requested_device)
                if encoder_attention_mask is not None:
                    encoder_attention_mask = encoder_attention_mask.to(device=requested_device)
            return prompt_embeds, encoder_attention_mask

        pipeline_cls._cvsuite_original_get_qwen_prompt_embeds = original
        pipeline_cls._get_qwen_prompt_embeds = _get_qwen_prompt_embeds_compat
        pipeline_cls._cvsuite_qwen_prompt_device_patch = True

    @classmethod
    def patch_vae_image_latent_device(cls, pipeline_cls: Any) -> None:
        if getattr(pipeline_cls, "_cvsuite_qwen_vae_latent_device_patch", False):
            return
        original = getattr(pipeline_cls, "_encode_vae_image", None)
        if original is None:
            return
        signature = inspect.signature(original)

        @wraps(original)
        def _encode_vae_image_compat(pipeline, *args, **kwargs):
            bound = signature.bind(pipeline, *args, **kwargs)
            requested_device = getattr(pipeline, "_execution_device", None)
            vae_device = cls._module_parameter_device(getattr(pipeline, "vae", None))
            image = bound.arguments.get("image")
            if vae_device is not None and torch.is_tensor(image):
                bound.arguments["image"] = image.to(device=vae_device)
            image_latents = original(*bound.args, **bound.kwargs)
            if requested_device is not None and torch.is_tensor(image_latents):
                image_latents = image_latents.to(device=requested_device)
            return image_latents

        pipeline_cls._cvsuite_original_encode_vae_image = original
        pipeline_cls._encode_vae_image = _encode_vae_image_compat
        pipeline_cls._cvsuite_qwen_vae_latent_device_patch = True

    @classmethod
    def patch_vae_decode_input_device(cls, pipeline: Any) -> None:
        vae = getattr(pipeline, "vae", None)
        if vae is None or getattr(vae, "_cvsuite_qwen_decode_device_patch", False):
            return
        original = getattr(vae, "decode", None)
        if original is None:
            return

        @wraps(original)
        def _decode_compat(*args, **kwargs):
            vae_device = cls._module_parameter_device(vae)
            if vae_device is not None:
                args = tuple(arg.to(device=vae_device) if torch.is_tensor(arg) else arg for arg in args)
                kwargs = {
                    key: value.to(device=vae_device) if torch.is_tensor(value) else value
                    for key, value in kwargs.items()
                }
            return original(*args, **kwargs)

        vae._cvsuite_original_decode = original
        vae.decode = _decode_compat
        vae._cvsuite_qwen_decode_device_patch = True

    def prepare_managed_auto_pipeline(self, pipeline: Any) -> Any:
        self.patch_prompt_encoder_input_device(pipeline.__class__)
        self.patch_vae_image_latent_device(pipeline.__class__)
        self.patch_vae_decode_input_device(pipeline)
        return pipeline

    def build_call_kwargs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        call_kwargs = super().build_call_kwargs(*args, **kwargs)
        if not call_kwargs.get("negative_prompt"):
            call_kwargs["true_cfg_scale"] = 1.0
        return call_kwargs
