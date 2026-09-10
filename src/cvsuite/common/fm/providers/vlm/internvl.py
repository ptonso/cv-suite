from __future__ import annotations

import importlib
import importlib.util
import io
import sys
import warnings
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.dynamic_module_utils import HF_MODULES_CACHE, _sanitize_module_name, init_hf_modules

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import utils
from cvsuite.common.fm.core.paths import hf_repo_cache_dir
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseVLMModel
from cvsuite.common.fm.providers.bases.vlm import VLMJob

DEFAULT_MODEL_ID = "OpenGVLab/InternVL2_5-1B"
PARAMS = "1B"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
            continue
        if ratio_diff == best_ratio_diff:
            threshold = 0.5 * image_size * image_size * ratio[0] * ratio[1]
            if area > threshold:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int = 1,
    max_num: int = 12,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = sorted(
        {(cols, rows) for count in range(min_num, max_num + 1) for cols in range(1, count + 1) for rows in range(1, count + 1) if min_num <= cols * rows <= max_num},
        key=lambda item: item[0] * item[1],
    )
    cols, rows = _find_closest_aspect_ratio(aspect_ratio, list(target_ratios), width, height, input_size)
    target_width = input_size * cols
    target_height = input_size * rows
    resized = image.resize((target_width, target_height))
    processed: list[Image.Image] = []
    for index in range(cols * rows):
        x0 = (index % cols) * input_size
        y0 = (index // cols) * input_size
        box = (x0, y0, x0 + input_size, y0 + input_size)
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((input_size, input_size)))
    return processed


def _load_pixel_values(image: Image.Image, input_size: int, max_num: int) -> torch.Tensor:
    transform = _build_transform(input_size)
    patches = _dynamic_preprocess(image, input_size=input_size, max_num=max_num)
    return torch.stack([transform(patch) for patch in patches])


def _build_generation_config(tokenizer: AutoTokenizer, max_new_tokens: int) -> dict[str, object]:
    generation_config: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generation_config["pad_token_id"] = pad_token_id
    elif eos_token_id is not None:
        generation_config["pad_token_id"] = eos_token_id
    return generation_config


class _FilteredLineStream(io.TextIOBase):
    def __init__(self, target, suppressed_lines: tuple[str, ...]) -> None:
        self._target = target
        self._suppressed = set(suppressed_lines)
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while True:
            newline_idx = self._buffer.find("\n")
            if newline_idx == -1:
                break
            line = self._buffer[: newline_idx + 1]
            self._buffer = self._buffer[newline_idx + 1 :]
            if line.rstrip("\n") not in self._suppressed:
                self._target.write(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer and self._buffer not in self._suppressed:
            self._target.write(self._buffer)
        self._buffer = ""
        self._target.flush()


@contextmanager
def _suppress_known_internvl_loader_noise():
    filtered_stdout = _FilteredLineStream(sys.stdout, ("FlashAttention2 is not installed.",))
    filtered_stderr = _FilteredLineStream(sys.stderr, ("FlashAttention2 is not installed.",))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
            category=FutureWarning,
        )
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = filtered_stdout
        sys.stderr = filtered_stderr
        try:
            yield
        finally:
            filtered_stdout.flush()
            filtered_stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


@dataclass(frozen=True)
class InternVLOptions:
    model_id: str | None = None
    max_new_tokens: int | None = None
    trust_remote_code: bool | None = None
    use_fast: bool | None = None
    input_size: int | None = None
    max_num: int | None = None
    use_flash_attn: bool | None = None
    revision: str | None = None


@dataclass(frozen=True)
class InternVLConfig:
    model_id: str = DEFAULT_MODEL_ID
    max_new_tokens: int = 64
    trust_remote_code: bool = True
    use_fast: bool = False
    input_size: int = 448
    max_num: int = 12
    use_flash_attn: bool = True
    revision: str | None = None


@dataclass
class InternVLRuntime:
    cfg: InternVLConfig
    revision: str | None
    device: torch.device
    dtype: torch.dtype
    model: AutoModel
    tokenizer: AutoTokenizer
    model_device: torch.device
    vision_input_device: torch.device
    vision_input_dtype: torch.dtype
    precision: utils.ResolvedPrecision | None = None


def _resolve_vision_input_spec(
    model: AutoModel,
    fallback_device: torch.device,
    fallback_dtype: torch.dtype,
) -> tuple[torch.device, torch.dtype]:
    candidates = [
        getattr(getattr(getattr(model, "vision_model", None), "embeddings", None), "patch_embedding", None),
        getattr(model, "vision_model", None),
        model,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        tensor = None
        for attr_name in ("bias", "weight"):
            value = getattr(candidate, attr_name, None)
            if torch.is_tensor(value):
                tensor = value
                break
        if tensor is None and hasattr(candidate, "parameters"):
            with suppress(Exception):
                tensor = next(candidate.parameters())
        if tensor is None:
            continue
        device = tensor.device if getattr(tensor, "device", None) is not None else fallback_device
        if device.type == "meta":
            device = fallback_device
        dtype = tensor.dtype if getattr(tensor, "dtype", None) is not None else fallback_dtype
        return device, dtype
    return fallback_device, fallback_dtype


def _resolve_model_revision(model_id: str, revision: str | None, hub_dir: Path | None) -> str | None:
    if revision:
        return revision
    if hub_dir is None or "/" not in model_id:
        return None
    ref_path = hf_repo_cache_dir(hub_dir, model_id) / "refs" / "main"
    try:
        cached_revision = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return cached_revision or None


def _build_drop_path_rates(drop_path_rate: float, num_hidden_layers: int) -> list[float]:
    layers = max(0, int(num_hidden_layers))
    if layers == 0:
        return []
    if layers == 1:
        return [0.0]
    stop = float(drop_path_rate)
    step = stop / float(layers - 1)
    return [step * idx for idx in range(layers)]


def _resolve_language_input_device(model: AutoModel, fallback_device: torch.device) -> torch.device:
    language_model = getattr(model, "language_model", None)
    get_input_embeddings = getattr(language_model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        with suppress(Exception):
            embedding = get_input_embeddings()
            weight = getattr(embedding, "weight", None)
            if torch.is_tensor(weight) and getattr(weight, "device", None) is not None and weight.device.type != "meta":
                return weight.device

    for candidate in (language_model, model):
        if candidate is None or not hasattr(candidate, "parameters"):
            continue
        with suppress(Exception):
            tensor = next(candidate.parameters())
            if getattr(tensor, "device", None) is not None and tensor.device.type != "meta":
                return tensor.device
    return fallback_device


@contextmanager
def _suppress_generate_device_mismatch_warning(model: AutoModel, input_ids: torch.Tensor | None):
    model_device = None
    with suppress(Exception):
        model_device = model.device
    if (
        model_device is None
        or getattr(model_device, "type", None) != "meta"
        or not torch.is_tensor(input_ids)
        or input_ids.device.type == "meta"
    ):
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"You are calling \.generate\(\) with the `input_ids` being on a device type different than your model's device\..*",
            category=UserWarning,
            module=r"transformers\.generation\.utils",
        )
        yield


def _patch_internvl_chat_class(chat_model_cls, chat_module) -> None:
    if chat_model_cls is None or getattr(chat_model_cls, "_cvsuite_text_device_patched", False):
        return

    def _patched_batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            print("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print("Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.")

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        queries = []
        template = None
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = chat_module.get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
        text_device = _resolve_language_input_device(self, self.device)
        input_ids = model_inputs["input_ids"].to(text_device)
        attention_mask = model_inputs["attention_mask"].to(text_device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def _patched_chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):
        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = chat_module.get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        text_device = _resolve_language_input_device(self, self.device)
        input_ids = model_inputs["input_ids"].to(text_device)
        attention_mask = model_inputs["attention_mask"].to(text_device)
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
        query_to_print = query_to_print.replace(f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>")
        if verbose:
            print(query_to_print, response)
        return response

    @torch.no_grad()
    def _patched_generate(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        visual_features=None,
        generation_config=None,
        output_hidden_states=None,
        **generate_kwargs,
    ):
        if input_ids is None:
            raise ValueError(
                "InternVL generate requires `input_ids` so generation stays on the language-model device "
                "when using managed offload."
            )

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            flat_input_ids = input_ids.reshape(B * N)
            selected = flat_input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        with _suppress_generate_device_mismatch_warning(self.language_model, input_ids):
            outputs = self.language_model.generate(
                input_ids=input_ids,
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                **generate_kwargs,
            )

        return outputs

    chat_model_cls.batch_chat = _patched_batch_chat
    chat_model_cls.chat = _patched_chat
    chat_model_cls.generate = _patched_generate
    chat_model_cls._cvsuite_text_device_patched = True


def _patch_loaded_internvl_chat_model(model: AutoModel) -> None:
    chat_model_cls = getattr(model, "__class__", None)
    if chat_model_cls is None:
        return
    module_name = getattr(chat_model_cls, "__module__", None)
    if not module_name:
        return
    try:
        chat_module = sys.modules.get(module_name) or importlib.import_module(module_name)
    except Exception:
        return
    _patch_internvl_chat_class(chat_model_cls, chat_module)


def _patch_remote_internvl_encoder(model_id: str, revision: str | None) -> None:
    if revision is None or "/" not in model_id:
        return
    namespace, repo = model_id.split("/", 1)
    vit_module_name = ".".join(
        (
            "transformers_modules",
            _sanitize_module_name(namespace),
            _sanitize_module_name(repo),
            revision,
            "modeling_intern_vit",
        )
    )
    chat_module_name = ".".join(
        (
            "transformers_modules",
            _sanitize_module_name(namespace),
            _sanitize_module_name(repo),
            revision,
            "modeling_internvl_chat",
        )
    )
    try:
        init_hf_modules()
        if not Path(HF_MODULES_CACHE).exists():
            return
        vit_module = importlib.import_module(vit_module_name)
        chat_module = importlib.import_module(chat_module_name)
        encoder_cls = getattr(vit_module, "InternVisionEncoder", None)
        encoder_layer_cls = getattr(vit_module, "InternVisionEncoderLayer", None)
        attention_cls = getattr(vit_module, "InternAttention", None)
        chat_model_cls = getattr(chat_module, "InternVLChatModel", None)
        if encoder_cls is None or encoder_layer_cls is None:
            return

        if not getattr(encoder_cls, "_cvsuite_drop_path_patched", False):
            def _patched_init(self, config) -> None:
                torch.nn.Module.__init__(self)
                self.config = config
                dpr = _build_drop_path_rates(config.drop_path_rate, config.num_hidden_layers)
                self.layers = torch.nn.ModuleList(
                    [encoder_layer_cls(config, dpr[idx]) for idx in range(int(config.num_hidden_layers))]
                )
                self.gradient_checkpointing = True

            encoder_cls.__init__ = _patched_init
            encoder_cls._cvsuite_drop_path_patched = True

        if (
            attention_cls is not None
            and not getattr(attention_cls, "_cvsuite_sdpa_patched", False)
            and hasattr(torch.nn.functional, "scaled_dot_product_attention")
        ):
            def _patched_naive_attn(self, x):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)

                if self.qk_normalization:
                    B_, H_, N_, D_ = q.shape
                    q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
                    k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

                dropout_p = self.attn_drop.p if self.training else 0.0
                try:
                    x = torch.nn.functional.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=dropout_p,
                        scale=self.scale,
                    )
                except TypeError:
                    x = torch.nn.functional.scaled_dot_product_attention(
                        q * self.scale,
                        k,
                        v,
                        dropout_p=dropout_p,
                    )
                x = x.transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x

            attention_cls._naive_attn = _patched_naive_attn
            attention_cls._cvsuite_sdpa_patched = True

        _patch_internvl_chat_class(chat_model_cls, chat_module)
    except Exception:
        return


@contextmanager
def _force_cpu_linspace_when_default_is_meta():
    original_linspace = torch.linspace

    def _patched_linspace(start, end, steps, *args, **kwargs):
        value = original_linspace(start, end, steps, *args, **kwargs)
        if isinstance(value, torch.Tensor) and value.device.type == "meta":
            retry_kwargs = dict(kwargs)
            retry_kwargs["device"] = "cpu"
            retry_kwargs.pop("out", None)
            return original_linspace(start, end, steps, *args, **retry_kwargs)
        return value

    torch.linspace = _patched_linspace
    try:
        yield
    finally:
        torch.linspace = original_linspace


class InternVLModel(BaseVLMModel[InternVLOptions, InternVLRuntime]):
    description = "InternVL wrapper: runs image-chat prompts over a VisionDataset JSON."
    options_cls = InternVLOptions
    supports_nf4_precision = True
    supports_managed_auto_device = True

    @staticmethod
    def _clear_cuda_cache(device: torch.device) -> None:
        if device.type != "cuda":
            return
        try:
            torch.cuda.empty_cache()
        except Exception:
            return

    def _raise_oom(
        self,
        exc: torch.OutOfMemoryError,
        *,
        runtime: InternVLRuntime,
        ctx: RuntimeContext[InternVLOptions],
        job: VLMJob | None,
        num_patches: int,
    ) -> None:
        scope = f"image idx={job.record_idx}" if job is not None else "batch"
        raise RuntimeError(
            "InternVL ran out of CUDA memory while processing "
            f"{scope} with {num_patches} tile(s) "
            f"(hf_model={runtime.cfg.model_id}, precision={ctx.request.precision}, "
            f"input_size={runtime.cfg.input_size}, max_num={runtime.cfg.max_num}, batch_size={ctx.batch_size}). "
            "Try `--batch 1`, lowering `--model-arg max_num=<n>` or `--model-arg input_size=<n>`, "
            "or installing `flash-attn` in the InternVL venv."
        ) from exc

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[InternVLOptions]) -> InternVLRuntime:
        cfg = self.load_dataclass_config(ctx.config_path, InternVLConfig, ctx.options)
        source = utils.materialize_hf_model_source(
            cfg.model_id,
            hub_dir=ctx.hub_dir,
            stage_dir=getattr(ctx, "stage_dir", None),
            caller_cwd=getattr(ctx, "caller_cwd", Path.cwd()),
            revision=cfg.revision,
        )
        device = utils.select_device(ctx.request.device)
        precision = self.resolve_runtime_precision(dataset, ctx, device)
        dtype = precision.compute_dtype
        default_device_map = self.resolve_device_map(ctx.request.device)
        use_flash_attn = bool(cfg.use_flash_attn and device.type == "cuda" and _flash_attn_available())
        shared_kwargs = utils.build_hf_source_kwargs(
            source,
            trust_remote_code=cfg.trust_remote_code,
        )

        with _suppress_known_internvl_loader_noise():
            config = AutoConfig.from_pretrained(source.load_arg, **shared_kwargs)
        revision = (
            cfg.revision
            or getattr(config, "_commit_hash", None)
            or source.revision
            or _resolve_model_revision(cfg.model_id, None, ctx.hub_dir)
        )
        if revision is not None and "revision" not in shared_kwargs:
            shared_kwargs["revision"] = revision

        _patch_remote_internvl_encoder(cfg.model_id, revision)

        with _suppress_known_internvl_loader_noise(), _force_cpu_linspace_when_default_is_meta():
            placement_kwargs, placement = self.build_hf_load_kwargs(
                ctx,
                device,
                precision,
                dtype_key="dtype",
                default_device_map=default_device_map,
            )
            load_kwargs = {
                "config": config,
                "low_cpu_mem_usage": placement.managed_device_map_active,
                "use_flash_attn": use_flash_attn,
                **shared_kwargs,
                **placement_kwargs,
            }
            model = self.load_hf_pretrained_model(
                AutoModel.from_pretrained,
                precision,
                source.load_arg,
                **load_kwargs,
            )
        _patch_loaded_internvl_chat_model(model)
        if not placement.managed_device_map_active and precision.quantization_mode is None:
            model = model.to(device=device, dtype=dtype)
        model.eval()
        with _suppress_known_internvl_loader_noise():
            tokenizer = AutoTokenizer.from_pretrained(
                source.load_arg,
                use_fast=cfg.use_fast,
                **shared_kwargs,
            )
        model_device = next(model.parameters()).device
        vision_input_device, vision_input_dtype = _resolve_vision_input_spec(model, model_device, dtype)
        return InternVLRuntime(
            cfg=cfg,
            revision=revision,
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=tokenizer,
            model_device=model_device,
            vision_input_device=vision_input_device,
            vision_input_dtype=vision_input_dtype,
            precision=precision,
        )

    def process_batch(
        self,
        dataset: VisionDataset,
        batch: list[VLMJob],
        runtime: InternVLRuntime,
        ctx: RuntimeContext[InternVLOptions],
    ) -> BatchResult:
        warnings: list[str] = []
        modified: list[int] = []
        questions: list[str] = []
        pixel_batches: list[torch.Tensor] = []
        num_patches_list: list[int] = []
        ready_jobs, images, warnings = self.load_batch_images(dataset, batch)
        active_jobs: list[VLMJob] = []
        for job, image in zip(ready_jobs, images):
            try:
                pixel_values = _load_pixel_values(image, runtime.cfg.input_size, runtime.cfg.max_num)
            except Exception as exc:
                warnings.append(f"[internvl] failed to prepare image idx={job.record_idx} path={job.image_path}: {exc}")
                continue

            active_jobs.append(job)
            questions.append(self.ensure_image_token(job.question))
            pixel_batches.append(pixel_values)
            num_patches_list.append(int(pixel_values.shape[0]))

        if not active_jobs:
            return BatchResult(modified_record_indices=modified, warnings=warnings)

        generation_config = _build_generation_config(runtime.tokenizer, runtime.cfg.max_new_tokens)

        if len(active_jobs) > 1 and hasattr(runtime.model, "batch_chat"):
            batch_pixel_values = None
            try:
                batch_pixel_values = torch.cat(pixel_batches, dim=0).to(
                    runtime.vision_input_device,
                    dtype=runtime.vision_input_dtype,
                )
                with torch.inference_mode():
                    answers = runtime.model.batch_chat(
                        runtime.tokenizer,
                        batch_pixel_values,
                        num_patches_list=num_patches_list,
                        questions=questions,
                        generation_config=generation_config,
                    )
            except torch.OutOfMemoryError:
                batch_pixel_values = None
                self._clear_cuda_cache(runtime.model_device)
                warnings.append(
                    "[internvl] batch_chat ran out of CUDA memory; retrying serial chat "
                    f"for {len(active_jobs)} job(s) / {sum(num_patches_list)} tile(s)."
                )
            else:
                if not isinstance(answers, (list, tuple)):
                    raise TypeError(
                        f"InternVL batch_chat returned {type(answers).__name__}, expected a list/tuple for {len(active_jobs)} jobs."
                    )
                if len(answers) != len(active_jobs):
                    raise ValueError(
                        f"InternVL batch_chat returned {len(answers)} answers for {len(active_jobs)} jobs."
                    )
                modified.extend(self.append_answers(dataset, active_jobs, answers))
                return BatchResult(modified_record_indices=modified, warnings=warnings)

        for job, pixel_values, question in zip(active_jobs, pixel_batches, questions):
            num_patches = int(pixel_values.shape[0])
            try:
                pixel_values = pixel_values.to(runtime.vision_input_device, dtype=runtime.vision_input_dtype)
                with torch.inference_mode():
                    answer = runtime.model.chat(runtime.tokenizer, pixel_values, question, generation_config)
            except torch.OutOfMemoryError as exc:
                pixel_values = None
                self._clear_cuda_cache(runtime.model_device)
                self._raise_oom(exc, runtime=runtime, ctx=ctx, job=job, num_patches=num_patches)
            modified.extend(self.append_answers(dataset, [job], [answer]))

        return BatchResult(modified_record_indices=modified, warnings=warnings)

    def build_fm_meta(
        self,
        dataset: VisionDataset,
        runtime: InternVLRuntime,
        ctx: RuntimeContext[InternVLOptions],
    ) -> dict[str, object]:
        return self.build_vlm_meta(
            runtime=runtime,
            ctx=ctx,
            hf_model_id=runtime.cfg.model_id,
            precision=runtime.precision,
            extra={
                "max_new_tokens": runtime.cfg.max_new_tokens,
                "input_size": runtime.cfg.input_size,
                "max_num": runtime.cfg.max_num,
                "revision": runtime.revision,
            },
        )


MODEL = InternVLModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
