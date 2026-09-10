from __future__ import annotations

from dataclasses import dataclass

from cvsuite.common.fm.providers.bases.create import BaseCreateGenerationModel, CreateGenerationOptions, DEFAULT_CREATE_BACKEND_ALIASES
from cvsuite.common.fm.providers.bases.diffusion import BaseDiffusersCreateBackend

DEFAULT_MODEL_ID = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
PARAMS = "1.6B"


@dataclass(frozen=True)
class SanaOptions(CreateGenerationOptions):
    backend: str | None = "sana"
    model_id: str | None = DEFAULT_MODEL_ID
    width: int | None = 1024
    height: int | None = 1024
    num_inference_steps: int | None = 20
    guidance_scale: float | None = 5.0


class SanaCreateBackend(BaseDiffusersCreateBackend):
    default_model_id = DEFAULT_MODEL_ID
    pipeline_class_name = "DiffusionPipeline"
    default_num_inference_steps = 20
    default_guidance_scale = 5.0


class SanaModel(BaseCreateGenerationModel[SanaOptions]):
    description = "Sana 1.6B text-to-image wrapper."
    options_cls = SanaOptions
    supports_managed_auto_device = True
    backend_aliases = {
        **DEFAULT_CREATE_BACKEND_ALIASES,
        "sana": "cvsuite.common.fm.providers.create.sana:SanaCreateBackend",
        "sana-1.6b": "cvsuite.common.fm.providers.create.sana:SanaCreateBackend",
        "sana_1_6b": "cvsuite.common.fm.providers.create.sana:SanaCreateBackend",
    }


MODEL = SanaModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
