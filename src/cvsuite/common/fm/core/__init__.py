"""Shared FM runtime for dataset-in/dataset-out model execution."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "dump_dataset",
    "load_dataset",
    "FMRunner",
    "RunnerConfig",
]

_EXPORTS = {
    "dump_dataset": (".export", "dump_dataset"),
    "load_dataset": (".export", "load_dataset"),
    "FMRunner": (".runner", "FMRunner"),
    "RunnerConfig": (".runner", "RunnerConfig"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
