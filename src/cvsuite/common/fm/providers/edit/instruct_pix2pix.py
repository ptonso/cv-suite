from __future__ import annotations

import warnings
from dataclasses import dataclass

from cvsuite.common.fm.providers.bases.diffusion import BaseDiffusersEditBackend, build_diffusers_nf4_pipeline
from cvsuite.common.fm.providers.bases.edit import BaseEditGenerationModel, DEFAULT_EDIT_BACKEND_ALIASES, EditGenerationOptions

DEFAULT_MODEL_ID = "timbrooks/instruct-pix2pix"
PARAMS = "1B"


@dataclass(frozen=True)
class InstructPix2PixOptions(EditGenerationOptions):
    backend: str | None = "instruct-pix2pix"
    model_id: str | None = DEFAULT_MODEL_ID
    num_inference_steps: int | None = 100
    guidance_scale: float | None = 7.5
    image_guidance_scale: float | None = 1.5


class InstructPix2PixEditBackend(BaseDiffusersEditBackend):
    default_model_id = DEFAULT_MODEL_ID
    pipeline_class_name = "StableDiffusionInstructPix2PixPipeline"
    default_num_inference_steps = 100
    default_guidance_scale = 7.5

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
            pipeline_class_name="StableDiffusionInstructPix2PixPipeline",
            diffusers_components={"unet": "UNet2DConditionModel"},
            transformers_components={"text_encoder": "CLIPTextModel"},
        )


class InstructPix2PixModel(BaseEditGenerationModel[InstructPix2PixOptions]):
    description = "Stable Diffusion InstructPix2Pix image-edit wrapper."
    options_cls = InstructPix2PixOptions
    supports_managed_auto_device = True
    supports_nf4_precision = True
    backend_aliases = {
        **DEFAULT_EDIT_BACKEND_ALIASES,
        "instruct-pix2pix": "cvsuite.common.fm.providers.edit.instruct_pix2pix:InstructPix2PixEditBackend",
        "instruct_pix2pix": "cvsuite.common.fm.providers.edit.instruct_pix2pix:InstructPix2PixEditBackend",
        "pix2pix": "cvsuite.common.fm.providers.edit.instruct_pix2pix:InstructPix2PixEditBackend",
    }


MODEL = InstructPix2PixModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
