from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from cvsuite.common.fm.core import utils
from cvsuite.common.fm.providers.bases.diffusion import (
    BaseDiffusersEditBackend,
    BaseGenerationBackend,
)
from cvsuite.common.fm.providers.bases.gen import GeneratedImage


class Step1XEditBackend(BaseDiffusersEditBackend):
    default_model_id = "stepfun-ai/Step1X-Edit-v1p1-diffusers"
    pipeline_class_name = "DiffusionPipeline"
    default_num_inference_steps = 28
    default_guidance_scale = 6.0


class OvisU1EditBackend(BaseGenerationBackend):
    default_model_id = "AIDC-AI/Ovis-U1-3B"
    default_num_inference_steps = 50

    @staticmethod
    def patch_remote_code_compat(snapshot_path: Path | None) -> None:
        if snapshot_path is None:
            return
        OvisU1EditBackend._patch_flash_attn_fallback(snapshot_path / "modeling_aimv2.py")
        OvisU1EditBackend._patch_yak_meta_device_fallback(snapshot_path / "modeling_yak.py")
        for rel_path in ("configuration_ovis_u1.py", "modeling_ovis_u1.py"):
            OvisU1EditBackend._patch_auto_register_compat(snapshot_path / rel_path)

    @staticmethod
    def _patch_auto_register_compat(target: Path) -> None:
        if not target.exists():
            return
        text = target.read_text(encoding="utf-8")
        patched_lines: list[str] = []
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if (
                stripped.startswith("AutoConfig.register(")
                or stripped.startswith("AutoModel.register(")
            ) and "exist_ok=" not in stripped:
                line = line.rstrip("\r\n")[:-1] + ", exist_ok=True)" + ("\r\n" if line.endswith("\r\n") else "\n")
            patched_lines.append(line)
        patched = "".join(patched_lines)
        if patched != text:
            target.write_text(patched, encoding="utf-8")

    @staticmethod
    def _patch_flash_attn_fallback(target: Path) -> None:
        if not target.exists():
            return
        text = target.read_text(encoding="utf-8")
        if "_cvsuite_flash_attn_varlen_func" in text:
            patched = text.replace(
                "    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)\n"
                "    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)\n",
                "    q_embed = apply_rotary_emb(q, cos, sin).type_as(q)\n"
                "    k_embed = apply_rotary_emb(k, cos, sin).type_as(k)\n",
                1,
            )
            patched = patched.replace(
                "        while cos.ndim < x_rot.ndim:\n"
                "            cos = cos.unsqueeze(0 if cos.ndim == 1 else -2)\n"
                "            sin = sin.unsqueeze(0 if sin.ndim == 1 else -2)\n",
                "        if cos.ndim == 2 and x_rot.ndim == 4:\n"
                "            cos = cos.unsqueeze(0).unsqueeze(2)\n"
                "            sin = sin.unsqueeze(0).unsqueeze(2)\n"
                "        else:\n"
                "            while cos.ndim < x_rot.ndim:\n"
                "                cos = cos.unsqueeze(0)\n"
                "                sin = sin.unsqueeze(0)\n",
                1,
            )
            if patched != text:
                target.write_text(patched, encoding="utf-8")
            return
        patch = """try:
    from flash_attn.layers.rotary import apply_rotary_emb
    from flash_attn import flash_attn_varlen_func
except ImportError:
    def _cvsuite_rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb(x, cos, sin):
        rotary_dim = int(cos.shape[-1]) * 2
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        cos = torch.cat((cos, cos), dim=-1).to(device=x.device, dtype=x.dtype)
        sin = torch.cat((sin, sin), dim=-1).to(device=x.device, dtype=x.dtype)
        if cos.ndim == 2 and x_rot.ndim == 4:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        else:
            while cos.ndim < x_rot.ndim:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
        return torch.cat((x_rot * cos + _cvsuite_rotate_half(x_rot) * sin, x_pass), dim=-1)

    def _cvsuite_flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k):
        outputs = []
        for idx in range(int(cu_q.numel()) - 1):
            q_start, q_end = int(cu_q[idx]), int(cu_q[idx + 1])
            k_start, k_end = int(cu_k[idx]), int(cu_k[idx + 1])
            q_i = q[q_start:q_end].permute(1, 0, 2).unsqueeze(0)
            k_i = k[k_start:k_end].permute(1, 0, 2).unsqueeze(0)
            v_i = v[k_start:k_end].permute(1, 0, 2).unsqueeze(0)
            out = F.scaled_dot_product_attention(q_i, k_i, v_i)
            outputs.append(out.squeeze(0).permute(1, 0, 2))
        return torch.cat(outputs, dim=0)

    flash_attn_varlen_func = _cvsuite_flash_attn_varlen_func
"""
        patched = text.replace(
            "from flash_attn.layers.rotary import apply_rotary_emb\nfrom flash_attn import flash_attn_varlen_func\n",
            patch,
            1,
        )
        if patched == text:
            patched = text.replace(
                "from flash_attn.layers.rotary import apply_rotary_emb\r\nfrom flash_attn import flash_attn_varlen_func\r\n",
                patch,
                1,
            )
        if patched == text:
            return
        patched = patched.replace(
            "    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)\n"
            "    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)\n",
            "    q_embed = apply_rotary_emb(q, cos, sin).type_as(q)\n"
            "    k_embed = apply_rotary_emb(k, cos, sin).type_as(k)\n",
            1,
        )
        target.write_text(patched, encoding="utf-8")

    @staticmethod
    def _patch_yak_meta_device_fallback(target: Path) -> None:
        if not target.exists():
            return
        text = target.read_text(encoding="utf-8")
        helper = """
def _cvsuite_module_device_dtype(module, fallback_device):
    for param in module.parameters():
        if param.device.type != "meta":
            return param.device, param.dtype
    return fallback_device, torch.float32

def _cvsuite_ensure_tensor_on_device(tensor, device, dtype=None):
    if isinstance(tensor, torch.Tensor):
        result = tensor.to(device=device)
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result
    elif isinstance(tensor, (int, float)):
        return torch.tensor(tensor, device=device, dtype=dtype or torch.float32)
    return tensor

"""
        if "def _cvsuite_module_device_dtype(" not in text:
            text = text.replace("\ndef get_noise(\n", helper + "\ndef get_noise(\n", 1)
        elif "def _cvsuite_ensure_tensor_on_device(" not in text:
            # Add only the second helper
            helper2 = """def _cvsuite_ensure_tensor_on_device(tensor, device, dtype=None):
    if isinstance(tensor, torch.Tensor):
        result = tensor.to(device=device)
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result
    elif isinstance(tensor, (int, float)):
        return torch.tensor(tensor, device=device, dtype=dtype or torch.float32)
    return tensor

"""
            text = text.replace("\ndef get_noise(\n", helper2 + "\ndef get_noise(\n", 1)
        patched = text.replace(
            "        torch_device = next(self.backbone.parameters()).device\n",
            "        torch_device = next(self.backbone.parameters()).device\n"
            "        if torch_device.type == 'meta':\n"
            "            torch_device = torch.device('cpu')\n",
            1,
        )
        patched = patched.replace(
            "        if \"vae_pixel_values\" in cond:\n"
            "            img_vae_cond, cond_ids = self.compute_vae_encodings(\n"
            "                pixel_values=cond[\"vae_pixel_values\"], with_ids=True, time=1.0)\n",
            "        if \"vae_pixel_values\" in cond:\n"
            "            img_vae_cond, cond_ids = self.compute_vae_encodings(\n"
            "                pixel_values=cond[\"vae_pixel_values\"].to(torch_device), with_ids=True, time=1.0)\n",
            1,
        )
        patched = patched.replace(
            "        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()\n"
            "        pixel_values = pixel_values.to(self.vae.device, dtype=self.vae.dtype)\n",
            "        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()\n"
            "        vae_device = self.vae.device\n"
            "        if vae_device.type == 'meta':\n"
            "            vae_device = torch.device('cpu')\n"
            "        pixel_values = pixel_values.to(vae_device, dtype=self.vae.dtype)\n",
            1,
        )
        patched = patched.replace(
            "        if txt.shape[0] == 1 and bs > 1:\n"
            "            txt = repeat(txt, \"1 ... -> bs ...\", bs=bs)\n"
            "        txt_ids = torch.zeros(bs, txt.shape[1], 3).to(img.device)\n",
            "        txt = txt.to(torch_device)\n"
            "        if txt.shape[0] == 1 and bs > 1:\n"
            "            txt = repeat(txt, \"1 ... -> bs ...\", bs=bs)\n"
            "        txt_ids = torch.zeros(bs, txt.shape[1], 3).to(img.device)\n",
            1,
        )
        patched = patched.replace(
            "        no_both_txt = no_both_cond[\"txt\"]\n"
            "        if no_txt_cond is not None:\n"
            "            no_txt_txt = no_txt_cond[\"txt\"]\n",
            "        no_both_txt = no_both_cond[\"txt\"].to(torch_device)\n"
            "        if no_txt_cond is not None:\n"
            "            no_txt_txt = no_txt_cond[\"txt\"].to(torch_device)\n",
            1,
        )
        patched = patched.replace(
            "        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):\n"
            "            t_vec = torch.full((bs,), t_curr, dtype=input_img.dtype, device=input_img.device)\n"
            "            txt_ids = torch.zeros(bs, txt.shape[1], 3).to(txt.device)\n",
            "        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):\n"
            "            backbone_device, backbone_dtype = _cvsuite_module_device_dtype(self.backbone, input_img.device)\n"
            "            input_img = input_img.to(device=backbone_device, dtype=backbone_dtype)\n"
            "            img_ids = img_ids.to(device=backbone_device)\n"
            "            txt = txt.to(device=backbone_device, dtype=backbone_dtype)\n"
            "            neg_txt = neg_txt.to(device=backbone_device, dtype=backbone_dtype)\n"
            "            t_vec = torch.full((bs,), t_curr, dtype=backbone_dtype, device=backbone_device)\n"
            "            txt_ids = torch.zeros(bs, txt.shape[1], 3).to(backbone_device)\n",
            1,
        )
        patched = patched.replace(
            "            input_img = input_img + (t_prev - t_curr) * pred\n",
            "            pred = pred.to(device=input_img.device, dtype=input_img.dtype)\n"
            "            input_img = input_img + (t_prev - t_curr) * pred\n",
            1,
        )
        patched = patched.replace(
            "            txt_ids = torch.zeros(bs, neg_txt.shape[1], 3).to(neg_txt.device)\n",
            "            txt_ids = torch.zeros(bs, neg_txt.shape[1], 3).to(backbone_device)\n",
            1,
        )
        patched = patched.replace(
            "        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):\n"
            "            t_vec = torch.full((bs * 1,), t_curr, dtype=input_img.dtype, device=input_img.device)\n"
            "            txt_ids = torch.zeros(bs, txt.shape[1], 3).to(txt.device)\n",
            "        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):\n"
            "            backbone_device, backbone_dtype = _cvsuite_module_device_dtype(self.backbone, input_img.device)\n"
            "            input_img = input_img.to(device=backbone_device, dtype=backbone_dtype)\n"
            "            img_ids = img_ids.to(device=backbone_device)\n"
            "            txt = txt.to(device=backbone_device, dtype=backbone_dtype)\n"
            "            no_txt_txt = no_txt_txt.to(device=backbone_device, dtype=backbone_dtype)\n"
            "            no_both_txt = no_both_txt.to(device=backbone_device, dtype=backbone_dtype)\n"
            "            img_cond = img_cond.to(device=backbone_device, dtype=backbone_dtype)\n"
            "            cond_img_ids = cond_img_ids.to(device=backbone_device)\n"
            "            t_vec = torch.full((bs * 1,), t_curr, dtype=backbone_dtype, device=backbone_device)\n"
            "            txt_ids = torch.zeros(bs, txt.shape[1], 3).to(backbone_device)\n",
            1,
        )
        patched = patched.replace(
            "            input_img = input_img + (t_prev - t_curr) * pred\n",
            "            input_img = _cvsuite_ensure_tensor_on_device(input_img, input_img.device)\n"
            "            pred = _cvsuite_ensure_tensor_on_device(pred, input_img.device, input_img.dtype)\n"
            "            t_diff = _cvsuite_ensure_tensor_on_device((t_prev - t_curr), input_img.device, input_img.dtype)\n"
            "            input_img = input_img + t_diff * pred\n"
        )
        patched = patched.replace(
            "            txt_ids = torch.zeros(bs, no_both_txt.shape[1], 3).to(no_both_txt.device)\n",
            "            txt_ids = torch.zeros(bs, no_both_txt.shape[1], 3).to(backbone_device)\n",
            1,
        )
        patched = patched.replace(
            "            txt_ids = torch.zeros(bs, no_txt_txt.shape[1], 3).to(no_txt_txt.device)\n",
            "            txt_ids = torch.zeros(bs, no_txt_txt.shape[1], 3).to(backbone_device)\n",
            1,
        )
        if patched != text:
            target.write_text(patched, encoding="utf-8")

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

    def _ovis_auto_gpu_memory_budget(self, current: Any) -> int | str:
        configured = self.config.get("auto_max_gpu_memory")
        if configured is not None:
            return configured
        current_bytes = self._memory_budget_to_int(current)
        if current_bytes is None:
            return current
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return current
        total_bytes = int(torch.cuda.get_device_properties(self.device).total_memory)
        runtime_reserve = max(8 * 1024**3, int(total_bytes * 0.35))
        budget = max(1, total_bytes - runtime_reserve)
        return min(int(current_bytes), int(budget))

    def _cap_ovis_auto_gpu_memory(self, load_kwargs: dict[str, Any], placement: Any) -> None:
        if str(load_kwargs.get("device_map", "")) != "auto":
            return
        device_map_override = self.config.get("auto_device_map")
        if device_map_override is None:
            offload_visual_generator = self.config.get("auto_offload_visual_generator", True)
            if offload_visual_generator:
                device_map_override = {"": 0, "visual_generator": "cpu"}
        if device_map_override is not None:
            load_kwargs["device_map"] = device_map_override
            try:
                self.model_wrapper._last_hf_load_placement = replace(
                    placement,
                    device_map=device_map_override,
                    managed_device_map_active=True,
                    memory_budget_source="ovis-explicit-map",
                )
            except Exception:
                pass
            return
        if self.config.get("auto_max_gpu_memory") is None and getattr(self.ctx.request, "max_gpu_memory", None):
            return
        max_memory = load_kwargs.get("max_memory")
        if not isinstance(max_memory, dict):
            return
        capped = dict(max_memory)
        changed = False
        for key, value in list(capped.items()):
            if self._cuda_index_from_memory_key(key) is None:
                continue
            new_value = self._ovis_auto_gpu_memory_budget(value)
            if new_value != value:
                capped[key] = new_value
                changed = True
        if not changed:
            return
        load_kwargs["max_memory"] = capped
        try:
            self.model_wrapper._last_hf_load_placement = replace(
                placement,
                max_memory=capped,
                memory_budget_source="ovis-runtime-reserve",
            )
        except Exception:
            pass

    def __init__(self, *, config: dict[str, Any] | None = None, dataset=None, ctx=None, mode: str = "", model=None) -> None:
        super().__init__(config=config, dataset=dataset, ctx=ctx, mode=mode, model=model)
        from transformers import AutoModelForCausalLM

        self.source = utils.materialize_hf_model_source(
            self.model_id,
            hub_dir=ctx.hub_dir,
            stage_dir=getattr(ctx, "stage_dir", None),
            caller_cwd=getattr(ctx, "caller_cwd", Path.cwd()),
        )
        self.patch_remote_code_compat(self.source.snapshot_path)
        load_kwargs, _placement = self.model_wrapper.build_hf_load_kwargs(
            ctx,
            self.device,
            self.precision,
            dtype_key="dtype",
            default_offload_folder=True,
        )
        self._cap_ovis_auto_gpu_memory(load_kwargs, _placement)
        self.model = self.model_wrapper.load_hf_pretrained_model(
            AutoModelForCausalLM.from_pretrained,
            self.precision,
            self.source.load_arg,
            **utils.build_hf_source_kwargs(
                self.source,
                trust_remote_code=True,
                **load_kwargs,
            ),
        )
        self.model = self.model.eval()
        if "device_map" not in load_kwargs and hasattr(self.model, "to"):
            self.model.to(self.device)
            if self.precision.quantization_mode is None:
                self.model.to(self.precision.compute_dtype)

    @staticmethod
    def _module_real_device(module: Any, fallback: torch.device) -> torch.device:
        device = getattr(module, "device", None)
        if device is not None:
            device = torch.device(device)
            if device.type != "meta":
                return device
        if hasattr(module, "parameters"):
            try:
                param_device = next(module.parameters()).device
                if param_device.type != "meta":
                    return param_device
            except StopIteration:
                pass
            except Exception:
                pass
        return fallback

    def _text_input_device(self) -> torch.device:
        return self._module_real_device(getattr(self.model, "llm", self.model), self.device)

    def _visual_generator_input_device(self) -> torch.device:
        visual_generator = getattr(self.model, "visual_generator", None)
        fallback = torch.device("cpu") if self.config.get("auto_offload_visual_generator", True) else self.device
        if visual_generator is None:
            return fallback
        return self._module_real_device(visual_generator, fallback)

    @staticmethod
    def _blank_image(width: int, height: int) -> Image.Image:
        return Image.new("RGB", (width, height), (255, 255, 255)).convert("RGB")

    def _build_inputs(self, prompt: str, pil_image: Image.Image, target_width: int, target_height: int):
        text_tokenizer = self.model.get_text_tokenizer()
        visual_tokenizer = self.model.get_visual_tokenizer()
        target_size = (int(target_width), int(target_height))
        pil_image, vae_pixel_values, cond_img_ids = self.model.visual_generator.process_image_aspectratio(pil_image, target_size)
        cond_img_ids[..., 0] = 1.0
        vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=self._visual_generator_input_device())
        width = pil_image.width
        height = pil_image.height
        resized_height, resized_width = visual_tokenizer.smart_resize(
            height,
            width,
            max_pixels=visual_tokenizer.image_processor.min_pixels,
        )
        pil_image = pil_image.resize((resized_width, resized_height))

        prompt, input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            prompt,
            [pil_image],
            generation_preface=None,
            return_labels=False,
            propagate_exception=False,
            multimodal_type="single_image",
            fix_sample_overall_length_navit=False,
        )
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        text_device = self._text_input_device()
        input_ids = input_ids.unsqueeze(0).to(device=text_device)
        attention_mask = attention_mask.unsqueeze(0).to(device=text_device)
        if pixel_values is not None:
            pixel_values = torch.cat([pixel_values.to(device=visual_tokenizer.device, dtype=self.precision.compute_dtype)], dim=0)
        if grid_thws is not None:
            grid_thws = torch.cat([grid_thws.to(device=visual_tokenizer.device)], dim=0)
        return text_tokenizer, input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values

    def _run_edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        width, height = image.size
        visual_tokenizer = self.model.get_visual_tokenizer()
        height, width = visual_tokenizer.smart_resize(height, width, factor=32)
        txt_cfg = float(self.config.get("txt_guidance_scale", self.guidance_for() or 6.0))
        img_cfg = float(self.config.get("img_guidance_scale", 1.5))
        steps = self.steps_for()
        text_tokenizer = self.model.get_text_tokenizer()
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=text_tokenizer.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True,
            height=height,
            width=width,
            num_steps=steps,
            seed=seed,
            img_cfg=img_cfg,
            txt_cfg=txt_cfg,
        )
        uncond_image = self._blank_image(width, height)
        uncond_prompt = "<image>\nGenerate an image."
        (
            _tokenizer,
            input_ids,
            pixel_values,
            attention_mask,
            grid_thws,
            _vae_pixel_values,
        ) = self._build_inputs(uncond_prompt, uncond_image, width, height)
        with torch.inference_mode():
            no_both_cond = self.model.generate_condition(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                grid_thws=grid_thws,
                **gen_kwargs,
            )

        image = image.resize((width, height))
        (
            _tokenizer,
            input_ids,
            pixel_values,
            attention_mask,
            grid_thws,
            _ignored,
        ) = self._build_inputs(uncond_prompt, image, width, height)
        with torch.inference_mode():
            no_txt_cond = self.model.generate_condition(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                grid_thws=grid_thws,
                **gen_kwargs,
            )

        (
            _tokenizer,
            input_ids,
            pixel_values,
            attention_mask,
            grid_thws,
            vae_pixel_values,
        ) = self._build_inputs("<image>\n" + prompt.strip(), image, width, height)
        with torch.inference_mode():
            cond = self.model.generate_condition(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                grid_thws=grid_thws,
                **gen_kwargs,
            )
            cond["vae_pixel_values"] = vae_pixel_values
            images = self.model.generate_img(cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs)
        return images[0]

    def generate_batch(self, *, jobs: Sequence[Any], **_kwargs: Any):
        outputs: list[GeneratedImage] = []
        for job in jobs:
            with Image.open(job.source_image_path) as source:
                image = source.convert("RGB")
            generator = self.generator_for(job)
            seed_value = generator.initial_seed() if generator is not None else (self.config.get("seed") or 42)
            seed = int(seed_value)
            outputs.append(GeneratedImage(image=self._run_edit(image, job.prompt, seed)))
        return outputs
