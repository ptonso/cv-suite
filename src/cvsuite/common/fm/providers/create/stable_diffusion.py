from __future__ import annotations

import warnings
from dataclasses import dataclass

from cvsuite.common.fm.providers.bases.create import BaseCreateGenerationModel, CreateGenerationOptions, DEFAULT_CREATE_BACKEND_ALIASES
from cvsuite.common.fm.providers.bases.diffusion import BaseDiffusersCreateBackend, build_diffusers_nf4_pipeline

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PARAMS = "3.5B"


@dataclass(frozen=True)
class StableDiffusionOptions(CreateGenerationOptions):
    backend: str | None = "stable-diffusion"
    model_id: str | None = DEFAULT_MODEL_ID
    width: int | None = 1024
    height: int | None = 1024
    num_inference_steps: int | None = 30
    guidance_scale: float | None = 7.0


class StableDiffusionCreateBackend(BaseDiffusersCreateBackend):
    default_model_id = DEFAULT_MODEL_ID
    pipeline_class_name = "AutoPipelineForText2Image"
    default_num_inference_steps = 30
    default_guidance_scale = 7.0

    def load_pipeline(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers\..*",
                category=FutureWarning,
                module=r"transformers\.utils\.hub",
            )
            return super().load_pipeline()

    def load_quantized_pipeline(self):
        return build_diffusers_nf4_pipeline(
            backend=self,
            pipeline_class_name="StableDiffusionXLPipeline",
            diffusers_components={"unet": "UNet2DConditionModel"},
            transformers_components={
                "text_encoder": "CLIPTextModel",
                "text_encoder_2": "CLIPTextModelWithProjection",
            },
        )


class StableDiffusionModel(BaseCreateGenerationModel[StableDiffusionOptions]):
    description = "Stable Diffusion XL create wrapper."
    options_cls = StableDiffusionOptions
    supports_managed_auto_device = True
    supports_nf4_precision = True
    backend_aliases = {
        **DEFAULT_CREATE_BACKEND_ALIASES,
        "stable-diffusion": "cvsuite.common.fm.providers.create.stable_diffusion:StableDiffusionCreateBackend",
        "stable_diffusion": "cvsuite.common.fm.providers.create.stable_diffusion:StableDiffusionCreateBackend",
    }


MODEL = StableDiffusionModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
