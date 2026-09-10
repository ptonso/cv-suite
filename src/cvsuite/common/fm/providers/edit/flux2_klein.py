from __future__ import annotations

from dataclasses import dataclass

from cvsuite.common.fm.providers.bases.diffusion import BaseDiffusersEditBackend
from cvsuite.common.fm.providers.bases.edit import BaseEditGenerationModel, DEFAULT_EDIT_BACKEND_ALIASES, EditGenerationOptions

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
PARAMS = "4B"


@dataclass(frozen=True)
class Flux2KleinOptions(EditGenerationOptions):
    backend: str | None = "flux2-klein"
    model_id: str | None = DEFAULT_MODEL_ID
    num_inference_steps: int | None = 28
    guidance_scale: float | None = None
    strength: float | None = 0.8
    max_sequence_length: int | None = 512


class Flux2KleinEditBackend(BaseDiffusersEditBackend):
    default_model_id = DEFAULT_MODEL_ID
    pipeline_class_name = "Flux2KleinPipeline"
    default_num_inference_steps = 28
    default_guidance_scale = None


class Flux2KleinModel(BaseEditGenerationModel[Flux2KleinOptions]):
    description = "FLUX.2 klein image-edit wrapper."
    options_cls = Flux2KleinOptions
    supports_managed_auto_device = True
    supports_nf4_precision = True
    backend_aliases = {
        **DEFAULT_EDIT_BACKEND_ALIASES,
        "flux2-klein": "cvsuite.common.fm.providers.edit.flux2_klein:Flux2KleinEditBackend",
        "flux2_klein": "cvsuite.common.fm.providers.edit.flux2_klein:Flux2KleinEditBackend",
        "flux.2-klein": "cvsuite.common.fm.providers.edit.flux2_klein:Flux2KleinEditBackend",
    }


MODEL = Flux2KleinModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
