from __future__ import annotations

from dataclasses import dataclass

from cvsuite.common.fm.providers.bases.diffusion import BaseDiffusersEditBackend, build_diffusers_nf4_pipeline
from cvsuite.common.fm.providers.bases.edit import BaseEditGenerationModel, DEFAULT_EDIT_BACKEND_ALIASES, EditGenerationOptions
from cvsuite.common.fm.providers.bases.qwen_image import QwenImageDiffusersMixin

DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit"
PARAMS = "20B"


@dataclass(frozen=True)
class QwenImageEditOptions(EditGenerationOptions):
    backend: str | None = "qwen-image-edit"
    model_id: str | None = DEFAULT_MODEL_ID
    num_inference_steps: int | None = 28
    guidance_scale: float | None = 4.0
    strength: float | None = 0.8
    max_sequence_length: int | None = 512
    true_cfg_scale: float | None = 4.0


class QwenImageEditBackend(QwenImageDiffusersMixin, BaseDiffusersEditBackend):
    default_model_id = DEFAULT_MODEL_ID
    pipeline_class_name = "DiffusionPipeline"
    default_num_inference_steps = 28
    default_guidance_scale = 4.0
    diffusers_execution_device_components = {"QwenImageEditPipeline": "transformer"}

    def load_quantized_pipeline(self):
        return build_diffusers_nf4_pipeline(
            backend=self,
            pipeline_class_name="QwenImageEditPipeline",
            diffusers_components={"transformer": "QwenImageTransformer2DModel"},
            transformers_components={"text_encoder": "Qwen2_5_VLForConditionalGeneration"},
        )


class QwenImageEditModel(BaseEditGenerationModel[QwenImageEditOptions]):
    description = "Qwen Image Edit wrapper."
    options_cls = QwenImageEditOptions
    supports_managed_auto_device = True
    supports_nf4_precision = True
    backend_aliases = {
        **DEFAULT_EDIT_BACKEND_ALIASES,
        "qwen-image-edit": "cvsuite.common.fm.providers.edit.qwen_image_edit:QwenImageEditBackend",
        "qwen_image_edit": "cvsuite.common.fm.providers.edit.qwen_image_edit:QwenImageEditBackend",
    }


MODEL = QwenImageEditModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
