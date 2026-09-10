"""Shared family base classes and helpers for FM wrappers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BaseClassificationModel": "cvsuite.common.fm.providers.bases.classification",
    "BaseCreateGenerationModel": "cvsuite.common.fm.providers.bases.create",
    "BaseEditGenerationModel": "cvsuite.common.fm.providers.bases.edit",
    "BaseGenerationModel": "cvsuite.common.fm.providers.bases.gen",
    "BaseGroundModel": "cvsuite.common.fm.providers.bases.ground",
    "BaseOCRModel": "cvsuite.common.fm.providers.bases.ocr",
    "BaseRecordImageModel": "cvsuite.common.fm.providers.bases.record_image",
    "BaseVLMModel": "cvsuite.common.fm.providers.bases.vlm",
    "CreateGenerationJob": "cvsuite.common.fm.providers.bases.create",
    "EditGenerationJob": "cvsuite.common.fm.providers.bases.edit",
    "GEN_OUTPUT_KEY_ATTR": "cvsuite.common.fm.providers.bases.gen",
    "GEN_PROMPT_ATTR": "cvsuite.common.fm.providers.bases.gen",
    "GEN_PROMPT_INDEX_ATTR": "cvsuite.common.fm.providers.bases.gen",
    "GEN_SAMPLE_INDEX_ATTR": "cvsuite.common.fm.providers.bases.gen",
    "GEN_SOURCE_IMAGE_ATTR": "cvsuite.common.fm.providers.bases.gen",
    "GEN_SOURCE_KEY_ATTR": "cvsuite.common.fm.providers.bases.gen",
    "GEN_SOURCE_RECORD_IDX_ATTR": "cvsuite.common.fm.providers.bases.gen",
    "GeneratedImage": "cvsuite.common.fm.providers.bases.gen",
    "GenerationRuntime": "cvsuite.common.fm.providers.bases.gen",
    "PromptCardCreateBackend": "cvsuite.common.fm.providers.bases.create",
    "PromptCardEditBackend": "cvsuite.common.fm.providers.bases.edit",
    "RecordImageJob": "cvsuite.common.fm.providers.bases.record_image",
    "VLMJob": "cvsuite.common.fm.providers.bases.vlm",
    "build_generation_runtime_extra": "cvsuite.common.fm.providers.bases.gen",
    "ensure_image_token": "cvsuite.common.fm.providers.bases.vlm",
    "generation_record_triplet": "cvsuite.common.fm.providers.bases.gen",
    "load_dataclass_config": "cvsuite.common.fm.providers.bases.vlm",
    "load_generation_runtime": "cvsuite.common.fm.providers.bases.gen",
    "load_rgb_image": "cvsuite.common.fm.providers.bases.vlm",
    "materialize_generated_output": "cvsuite.common.fm.providers.bases.gen",
    "normalize_generated_output": "cvsuite.common.fm.providers.bases.gen",
    "parse_prompt_payload": "cvsuite.common.fm.providers.bases.gen",
    "resolve_device_map": "cvsuite.common.fm.providers.bases.vlm",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
