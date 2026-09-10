from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .fs import PREP_MANIFEST_NAME


def _yaml_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _yaml_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_value(item) for item in value]
    return value


def emit_prep_manifest(config: Any, command: str, stats: dict[str, Any] | None = None) -> None:
    data = asdict(config)
    options = {
        key: _yaml_value(value)
        for key, value in data.items()
        if key not in {"src", "dst", "manifest"}
    }
    subdir = getattr(config, "dst_subdir", None)
    out_dir = config.dst / subdir if subdir else config.dst
    path = out_dir / PREP_MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "branch": "prep",
        "command": command,
        "src": str(config.src),
        "dst": str(config.dst),
        "options": options,
    }
    if stats is not None:
        payload["stats"] = {key: _yaml_value(value) for key, value in stats.items()}
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
