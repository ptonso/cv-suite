from __future__ import annotations

from dataclasses import dataclass

from cvsuite.common.fm.providers.bases.create import BaseCreateGenerationModel, CreateGenerationOptions, DEFAULT_CREATE_BACKEND_ALIASES
from cvsuite.common.fm.providers.bases.diffusion import BaseDiffusersCreateBackend, build_flux_nf4_pipeline

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"
PARAMS = "12B"


@dataclass(frozen=True)
class FluxOptions(CreateGenerationOptions):
    backend: str | None = "flux"
    model_id: str | None = DEFAULT_MODEL_ID
    width: int | None = 1024
    height: int | None = 1024
    num_inference_steps: int | None = 28
    guidance_scale: float | None = 3.5
    max_sequence_length: int | None = 512


class FluxCreateBackend(BaseDiffusersCreateBackend):
    default_model_id = DEFAULT_MODEL_ID
    pipeline_class_name = "DiffusionPipeline"
    default_num_inference_steps = 28
    default_guidance_scale = 3.5

    def load_quantized_pipeline(self):
        return build_flux_nf4_pipeline(backend=self, pipeline_class_name=self.pipeline_class_name)


class FluxModel(BaseCreateGenerationModel[FluxOptions]):
    description = "FLUX.1 text-to-image wrapper."
    options_cls = FluxOptions
    supports_managed_auto_device = True
    supports_nf4_precision = True
    backend_aliases = {
        **DEFAULT_CREATE_BACKEND_ALIASES,
        "flux": "cvsuite.common.fm.providers.create.flux:FluxCreateBackend",
        "flux-dev": "cvsuite.common.fm.providers.create.flux:FluxCreateBackend",
        "flux_dev": "cvsuite.common.fm.providers.create.flux:FluxCreateBackend",
    }


MODEL = FluxModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
