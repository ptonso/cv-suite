from __future__ import annotations
import json, yaml
import types as _types
import dataclasses
import typing
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar, get_args, get_origin

from cvsuite.common.core.types import ImageInfo

T = TypeVar("T")

def write_json(path: Path, obj: Dict[str, Any], overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_csv(path: Path, rows: Dict[str, Any], overwrite: bool = False) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    keys = sorted({k for r in rows.values() for k in (r.keys() if isinstance(r, dict) else [])})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", *keys])
        w.writeheader()
        for k, v in rows.items():
            row = {"id": k}
            if isinstance(v, dict):
                row.update(v)
            w.writerow(row)


def write_yaml(path: Path, obj: Dict[str, Any], overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")



# --- DatasetRecord save and load helpers


def _encode(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):  # type: ignore[name-defined]
        return {f.name: _encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}  # type: ignore[name-defined]
    if isinstance(obj, ImageInfo):
        # Plain slotted class (lazy dims); serialize raw slots so encoding stays side-effect-free.
        return {
            "path": str(obj.path),
            "source": None if obj.source is None else str(obj.source),
            "width": obj._width,
            "height": obj._height,
        }
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(_encode(k)): _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(x) for x in obj]
    return obj


def _is_optional(tp: Any) -> bool:
    origin = get_origin(tp)
    if origin is _types.UnionType:
        return any(a is type(None) for a in get_args(tp))
    if origin is None:
        return False
    # typing.Union
    return any(a is type(None) for a in get_args(tp))


def _strip_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin in (_types.UnionType, None) and not get_args(tp):
        return tp
    if origin is _types.UnionType or origin is getattr(__import__("typing"), "Union", None):
        return next(a for a in get_args(tp) if a is not type(None))
    return tp


def _decode_value(tp: Any, val: Any) -> Any:
    if val is None:
        return None

    if _is_optional(tp):
        inner = _strip_optional(tp)
        return _decode_value(inner, val)

    origin = get_origin(tp)
    args = get_args(tp)

    if origin in (list, List):
        inner = args[0] if args else Any
        return [_decode_value(inner, x) for x in (val or [])]

    if origin in (dict, Dict):
        k_tp, v_tp = args if len(args) == 2 else (Any, Any)
        return {
            _decode_value(k_tp, k): _decode_value(v_tp, v)
            for k, v in (val or {}).items()
        }

    if dataclasses.is_dataclass(tp):  # type: ignore[name-defined]
        return _decode_dataclass(tp, val)

    if isinstance(tp, type) and issubclass(tp, Enum):
        return tp(val)

    if tp is Path:
        return Path(val)

    if isinstance(tp, type) and issubclass(tp, ImageInfo):
        return ImageInfo(
            path=Path(val["path"]),
            width=val.get("width"),
            height=val.get("height"),
            source=Path(val["source"]) if val.get("source") else None,
        )

    if tp is Any:
        return val

    if tp in (int, float, str, bool):
        return tp(val)

    return val


def _decode_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    hints = typing.get_type_hints(cls, include_extras=True)  # type: ignore[name-defined]
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):  # type: ignore[name-defined]
        if f.name not in data:
            continue
        tp = hints.get(f.name, f.type)
        kwargs[f.name] = _decode_value(tp, data[f.name])
    return cls(**kwargs)
