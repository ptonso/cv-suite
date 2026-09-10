from __future__ import annotations

from dataclasses import dataclass

from cvsuite.common.fm.providers.bases.edit import BaseEditGenerationModel, DEFAULT_EDIT_BACKEND_ALIASES, EditGenerationOptions

DEFAULT_MODEL_ID = "AIDC-AI/Ovis-U1-3B"
PARAMS = "3B"


@dataclass(frozen=True)
class OvisU1EditOptions(EditGenerationOptions):
    backend: str | None = "ovis-u1"
    model_id: str | None = DEFAULT_MODEL_ID
    num_inference_steps: int | None = 50
    guidance_scale: float | None = 6.0
    width: int | None = 1024
    height: int | None = 1024
    auto_max_gpu_memory: str | None = None
    auto_offload_visual_generator: bool | None = True
    auto_device_map: dict[str, str | int] | None = None


class OvisU1EditModel(BaseEditGenerationModel[OvisU1EditOptions]):
    description = "Ovis-U1 edit wrapper."
    options_cls = OvisU1EditOptions
    supports_managed_auto_device = True
    backend_aliases = {
        **DEFAULT_EDIT_BACKEND_ALIASES,
        "ovis-u1": "cvsuite.common.fm.providers.bases.edit_custom:OvisU1EditBackend",
        "ovis_u1": "cvsuite.common.fm.providers.bases.edit_custom:OvisU1EditBackend",
        "ovis_u1_3b": "cvsuite.common.fm.providers.bases.edit_custom:OvisU1EditBackend",
    }


MODEL = OvisU1EditModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
