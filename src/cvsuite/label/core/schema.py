from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from cvsuite.common.core.enums import Task


@dataclass
class DataYaml:
    path: Path
    task: Task
    names: List[str]
    root: Path
    splits: Dict[str, List[Path]]
    raw: Dict[str, Any]
    task_raw: str | None = None


def _resolve_root(path: Path, data: Dict[str, Any]) -> Path:
    base = path.parent
    root_field = data.get("path")
    if not root_field:
        return base
    root_path = Path(root_field)
    if root_path.is_absolute():
        return root_path
    candidate = (base / root_path).resolve()
    return candidate if candidate.exists() else base


def _resolve_split_paths(yaml_path: Path, base_root: Path, data: Dict[str, Any]) -> Dict[str, List[Path]]:
    yaml_dir = yaml_path.parent

    def _pick_path(key: str, entry: str) -> Path:
        p = Path(entry)
        if p.is_absolute():
            return p
        alt_key = {"val": "valid", "valid": "val"}.get(key)
        candidates = [
            (yaml_dir / p).resolve(),
            (base_root / p).resolve(),
            (yaml_dir / key / "images").resolve(),
            (base_root / key / "images").resolve(),
        ]
        if alt_key:
            candidates.extend(
                [
                    (yaml_dir / alt_key / "images").resolve(),
                    (base_root / alt_key / "images").resolve(),
                ]
            )
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    splits: Dict[str, List[Path]] = {}

    def _paths_for(key: str) -> List[Path]:
        val = data.get(key)
        if val is None:
            return []
        entries = val if isinstance(val, list) else [val]
        return [_pick_path(key, e) for e in entries]

    for key in ("train", "val", "valid", "test"):
        paths = _paths_for(key)
        if not paths:
            continue
        norm = "val" if key == "valid" else key
        splits[norm] = paths
    return splits


def read_data_yaml(path: Path) -> DataYaml:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    def _normalize_task(val: Any) -> tuple[Task, str | None]:
        if val is None:
            return Task.det, None
        raw = str(val).lower()
        aliases = {
            "detect": "det",
            "detection": "det",
            "segment": "seg",
            "segmentation": "seg",
            "pose": "pose",
            "keypoint": "pose",
            "keypoints": "pose",
        }
        norm = aliases.get(raw, raw)
        try:
            return Task(norm), raw
        except ValueError:
            return Task.det, raw

    task, task_raw = _normalize_task(data.get("task"))
    names = list(data.get("names", []))
    root = _resolve_root(path, data)
    splits = _resolve_split_paths(path, root, data)
    return DataYaml(path=path, task=task, names=names, root=root, splits=splits, raw=data, task_raw=task_raw)

def write_data_yaml(path: Path, root: Path, classes: List[str], task: str, split_paths: Dict[str, str] | None = None) -> None:
    obj: Dict[str, Any] = {
        "path": str(root),
        "names": classes,
        "task": task,
    }
    obj.update(split_paths or {"train": "train/images", "val": "val/images", "test": "test/images"})
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")

def infer_classes_from_labelme(records: List[Any]) -> List[str]:
    labels = []
    seen = set()
    for r in records:
        for s in r.shapes:
            if s.label not in seen:
                labels.append(s.label)
                seen.add(s.label)
    return labels



def resolve_data_yaml(src: Path) -> Path:
    if src.is_file() and src.suffix.lower() in {".yaml", ".yml"}:
        return src
    if src.is_dir():
        for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
            cand = src / name
            if cand.exists():
                return cand
        hit = next(src.rglob("data.yaml"), None)
        if hit:
            return hit
    raise FileNotFoundError(str(src))
