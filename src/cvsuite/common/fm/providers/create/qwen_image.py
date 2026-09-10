from __future__ import annotations

from dataclasses import dataclass

from cvsuite.common.fm.providers.bases.create import BaseCreateGenerationModel, CreateGenerationOptions, DEFAULT_CREATE_BACKEND_ALIASES
from cvsuite.common.fm.providers.bases.diffusion import BaseDiffusersCreateBackend, build_diffusers_nf4_pipeline
from cvsuite.common.fm.providers.bases.qwen_image import QwenImageDiffusersMixin

DEFAULT_MODEL_ID = "Qwen/Qwen-Image"
PARAMS = "20B"


@dataclass(frozen=True)
class QwenImageOptions(CreateGenerationOptions):
    backend: str | None = "qwen-image"
    model_id: str | None = DEFAULT_MODEL_ID
    width: int | None = 1024
    height: int | None = 1024
    num_inference_steps: int | None = 28
    guidance_scale: float | None = None
    max_sequence_length: int | None = 512
    true_cfg_scale: float | None = None


class QwenImageCreateBackend(QwenImageDiffusersMixin, BaseDiffusersCreateBackend):
    default_model_id = DEFAULT_MODEL_ID
    pipeline_class_name = "DiffusionPipeline"
    default_num_inference_steps = 28
    default_guidance_scale = None
    diffusers_execution_device_components = {"QwenImagePipeline": "transformer"}

    def load_quantized_pipeline(self):
        return build_diffusers_nf4_pipeline(
            backend=self,
            pipeline_class_name="QwenImagePipeline",
            diffusers_components={"transformer": "QwenImageTransformer2DModel"},
            transformers_components={"text_encoder": "Qwen2_5_VLForConditionalGeneration"},
        )


class QwenImageModel(BaseCreateGenerationModel[QwenImageOptions]):
    description = "Qwen Image text-to-image wrapper."
    options_cls = QwenImageOptions
    supports_managed_auto_device = True
    supports_nf4_precision = True
    backend_aliases = {
        **DEFAULT_CREATE_BACKEND_ALIASES,
        "qwen-image": "cvsuite.common.fm.providers.create.qwen_image:QwenImageCreateBackend",
        "qwen_image": "cvsuite.common.fm.providers.create.qwen_image:QwenImageCreateBackend",
    }


MODEL = QwenImageModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
