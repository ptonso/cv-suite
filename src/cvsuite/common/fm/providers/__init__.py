"""FM providers, registry, and shared utilities."""

from importlib import import_module
from typing import Any

from .registry import (
    MODEL_MODULES,
    default_model_id_for_provider,
    format_available_providers,
    model_registry_items,
    provider_choices_for_family,
    provider_registry_items,
    resolve_model_module,
    resolve_provider_module,
    resolve_provider_spec,
)

__all__ = [
    "MODEL_MODULES",
    "default_model_id_for_provider",
    "format_available_providers",
    "model_registry_items",
    "provider_choices_for_family",
    "provider_registry_items",
    "resolve_model_module",
    "resolve_provider_module",
    "resolve_provider_spec",
    "utils",
]


def __getattr__(name: str) -> Any:
    if name != "utils":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("cvsuite.common.fm.core.utils")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
