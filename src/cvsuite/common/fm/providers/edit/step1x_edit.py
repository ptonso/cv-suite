from __future__ import annotations

from dataclasses import dataclass

from cvsuite.common.fm.providers.bases.diffusion import BaseDiffusersEditBackend
from cvsuite.common.fm.providers.bases.edit import BaseEditGenerationModel, DEFAULT_EDIT_BACKEND_ALIASES, EditGenerationOptions

DEFAULT_MODEL_ID = "stepfun-ai/Step1X-Edit-v1p1-diffusers"
PARAMS = "unknown"


@dataclass(frozen=True)
class Step1XEditOptions(EditGenerationOptions):
    backend: str | None = "step1x-edit"
    model_id: str | None = DEFAULT_MODEL_ID
    num_inference_steps: int | None = 28
    guidance_scale: float | None = 6.0
    strength: float | None = 0.8
    width: int | None = 1024
    height: int | None = 1024


class Step1XEditBackend(BaseDiffusersEditBackend):
    default_model_id = DEFAULT_MODEL_ID
    pipeline_class_name = "DiffusionPipeline"
    default_num_inference_steps = 28
    default_guidance_scale = 6.0


class Step1XEditModel(BaseEditGenerationModel[Step1XEditOptions]):
    description = "Step1X edit wrapper."
    options_cls = Step1XEditOptions
    backend_aliases = {
        **DEFAULT_EDIT_BACKEND_ALIASES,
        "step1x-edit": "cvsuite.common.fm.providers.edit.step1x_edit:Step1XEditBackend",
        "step1x_edit": "cvsuite.common.fm.providers.edit.step1x_edit:Step1XEditBackend",
    }


MODEL = Step1XEditModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
