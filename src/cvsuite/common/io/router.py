"""Public IO router and registry for `cvsuite`.

This module exposes the package-level dataset entrypoints:

- `read_dataset(...)`
- `write_dataset(...)`
- `register_reader(...)`
- `register_writer(...)`
- `list_readers()`
- `list_writers()`

Adapters are registered by public format id, and the router dispatches from a
filesystem source into canonical `VisionDataset` / `VisionRecord` objects.

Example sources this router may dispatch:
    dataset/
      data.yaml
      train/
        images/
        labels/

    dataset/
      cat/
        a.jpg
      dog/
        b.jpg
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from cvsuite.common.core import VisionDataset

# Runtime registries keyed by the public format id, 
# for example `yolo` or `labelme`.
_READERS: dict[str, type] = {}
_WRITERS: dict[str, type] = {}
_PRIORITY: dict[str, int] = {}
_BUILTINS_REGISTERED = False

# Each builtin entry is:
# (
#   public_format_id,
#   module_import_path,
#   adapter_class_name,
#   has_reader,
#   has_writer,
#   auto_detect_priority,
# )
#
# Lower `auto_detect_priority` wins during `format="auto"` detection, so the
# table encodes the detection order described by the IO specs.
_BUILTINS: tuple[tuple[str, str, str, bool, bool, int], ...] = (
    ("semseg_mask", "cvsuite.common.io.semseg_mask", "SemSegMaskAdapter", True, False, 5),
    ("yolo", "cvsuite.common.io.yolo", "YoloAdapter", True, True, 10),
    ("coco", "cvsuite.common.io.coco", "CocoAdapter", True, True, 10),
    ("vqa_style", "cvsuite.common.io.vqa_style", "VQAStyleAdapter", True, True, 10),
    ("labelme", "cvsuite.common.io.labelme", "LabelMeAdapter", True, True, 20),
    ("shards_vlm", "cvsuite.common.io.shards_vlm", "ShardsVlmAdapter", True, True, 30),
    ("flat_vlm_json", "cvsuite.common.io.flat_vlm_json", "FlatVlmJsonAdapter", True, True, 40),
    ("class-dir", "cvsuite.common.io.class_dir", "ClassDirAdapter", True, True, 50),
    ("flat", "cvsuite.common.io.flat", "FlatAdapter", True, False, 60),
)


def register_reader(name: str, adapter: type, *, priority: int = 100) -> None:
    """Register a reader adapter under one public format id."""
    _READERS[name] = adapter
    _PRIORITY[name] = priority


def register_writer(name: str, adapter: type) -> None:
    """Register a writer adapter under one public format id."""
    _WRITERS[name] = adapter


def list_readers() -> list[str]:
    """Return the registered public reader format ids."""
    _ensure_builtins_registered()
    return list(_READERS.keys())


def list_writers() -> list[str]:
    """Return the registered public writer format ids."""
    _ensure_builtins_registered()
    return list(_WRITERS.keys())


def _ensure_builtins_registered() -> None:
    """Import and register builtin adapters exactly once."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    for name, module_name, symbol_name, has_reader, has_writer, priority in _BUILTINS:
        module = import_module(module_name)
        adapter = getattr(module, symbol_name)
        if has_reader:
            register_reader(name, adapter, priority=priority)
        if has_writer:
            register_writer(name, adapter)
    _BUILTINS_REGISTERED = True


def _select_reader_for_auto(src: Path) -> tuple[str, type]:
    """Pick one reader for `format="auto"` or fail on ambiguity."""
    matches: list[tuple[int, str, type]] = []
    for name, adapter in _READERS.items():
        if adapter.matches(src):
            matches.append((_PRIORITY.get(name, 100), name, adapter))
    if not matches:
        raise RuntimeError(f"Could not infer dataset format from: {src}")
    matches.sort(key=lambda item: (item[0], item[1]))
    best_priority = matches[0][0]
    best = [item for item in matches if item[0] == best_priority]
    if len(best) > 1:
        names = ", ".join(name for _priority, name, _adapter in best)
        raise RuntimeError(f"Ambiguous dataset format for {src}: {names}")
    _priority, name, adapter = best[0]
    return name, adapter


def read_dataset(
    src: str | Path,
    *,
    format: str = "auto",
    task: str = "auto",
    splits: list[str] | None = None,
    **adapter_options: Any,
) -> VisionDataset:
    """Read a filesystem dataset into the canonical in-memory contract."""
    _ensure_builtins_registered()
    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")

    if format == "auto":
        name, adapter = _select_reader_for_auto(path)
    else:
        if format not in _READERS:
            raise KeyError(f"Unknown reader format: {format}")
        name, adapter = format, _READERS[format]

    dataset: VisionDataset = adapter.read(path, task=task, splits=splits, **adapter_options)
    dataset.source_format = name  # built-in readers are canonical public formats
    for record in dataset.records:
        record.source_format = name
    return dataset


def write_dataset(
    dataset: VisionDataset,
    dst: str | Path,
    *,
    format: str,
    splits: list[str] | None = None,
    **writer_options: Any,
) -> None:
    """Write a canonical dataset through one named format adapter."""
    _ensure_builtins_registered()
    if format not in _WRITERS:
        raise KeyError(f"Unknown or unsupported writer format: {format}")
    writer = _WRITERS[format]
    writer.write(dataset, Path(dst), splits=splits, **writer_options)
