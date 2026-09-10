from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from ..actions import Action, apply_actions
from ..config import SampleConfig
from ..fs import PREP_ARTIFACT_NAMES, is_image
from ..logging import Logger


@dataclass(frozen=True)
class SampleEntry:
    src: Path
    name: str


def _load_json_safe(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _looks_like_annotated_dataset(src: Path) -> bool:
    if not src.exists():
        return False

    if src.is_file():
        suffix = src.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return True
        if suffix == ".json":
            data = _load_json_safe(src) or {}
            return ("images" in data and "annotations" in data) or ("shapes" in data)
        return False

    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        if (src / name).exists():
            return True

    direct_children = list(src.iterdir())
    lower_dir_names = {child.name.lower() for child in direct_children if child.is_dir()}
    if {"train", "val", "valid", "test"} & lower_dir_names:
        for split_name in ("train", "val", "valid", "test"):
            split_dir = src / split_name
            if not split_dir.exists():
                continue
            if (split_dir / "images").exists() and any(
                (split_dir / label_dir).exists()
                for label_dir in ("labels", "labels_det", "labels_seg", "labels_pose")
            ):
                return True

    direct_jsons = sorted(child for child in direct_children if child.is_file() and child.suffix.lower() == ".json")
    for json_path in direct_jsons[:5]:
        data = _load_json_safe(json_path) or {}
        if ("images" in data and "annotations" in data) or ("shapes" in data):
            return True
    return False


def _collect_entries(config: SampleConfig, logger: Logger) -> list[SampleEntry]:
    entries: list[SampleEntry] = []
    for src in config.srcs:
        if not src.exists():
            raise SystemExit(f"Source not found: {src}")
        if not src.is_dir():
            raise SystemExit(f"`cvsuite prep sample` expects directories only: {src}")
        if _looks_like_annotated_dataset(src):
            raise SystemExit(f"Annotated datasets are not supported by `cvsuite prep sample`. Use `cvsuite label sample` instead: {src}")

        children = [c for c in sorted(src.iterdir()) if c.name not in PREP_ARTIFACT_NAMES]
        nested = [child for child in children if child.is_dir()]
        if nested:
            raise SystemExit(
                f"`cvsuite prep sample` expects a flat raw image folder with no nested directories. "
                f"Use `cvsuite class sample` for class-organized folders: {src}"
            )

        src_entries = [
            SampleEntry(src=child, name=child.name)
            for child in children
            if child.is_file() and is_image(child)
        ]
        if not src_entries:
            raise SystemExit(f"No images found in source folder: {src}")
        logger.info(f"Discovered {len(src_entries)} images in {src}.")
        entries.extend(src_entries)

    logger.info(f"Pooled {len(entries)} images across {len(config.srcs)} source folder(s).")
    return entries


def _choose_indices(*, total: int, count: int | None, frac: float | None, rng: random.Random) -> list[int]:
    if total <= 0:
        return []
    if count is not None:
        if count > total:
            raise SystemExit(f"Requested --count {count} but only {total} pooled images are available.")
        return rng.sample(range(total), count)
    if frac is None:
        return []
    if frac < 0.0 or frac > 1.0:
        raise SystemExit("--frac must be between 0 and 1.")
    chosen = int(math.floor(total * frac))
    if frac > 0.0:
        chosen = max(1, chosen)
    chosen = min(chosen, total)
    return rng.sample(range(total), chosen)


def _next_flat_dst(dst: Path, used_names: set[str], name: str) -> Path:
    candidate = dst / name
    if candidate.name not in used_names:
        used_names.add(candidate.name)
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        alt = dst / f"{stem}_{counter}{suffix}"
        if alt.name not in used_names:
            used_names.add(alt.name)
            return alt
        counter += 1


def run_sample_core(config: SampleConfig) -> None:
    logger = Logger(verbose=config.verbose)
    entries = _collect_entries(config, logger)
    rng = random.Random(config.seed) if config.seed is not None else random.Random()
    indices = _choose_indices(
        total=len(entries),
        count=config.count,
        frac=config.frac,
        rng=rng,
    )
    selected = sorted((entries[idx] for idx in indices), key=lambda entry: str(entry.src))
    logger.info(f"Selected {len(selected)} pooled images for sampling.")

    used_names: set[str] = set()
    actions: list[Action] = []
    kind = "link" if config.hardlink else "copy"
    link_mode = "hard" if config.hardlink else "copy"
    for entry in selected:
        dst = _next_flat_dst(config.dst, used_names, entry.name)
        actions.append(Action(kind=kind, src=entry.src, dst=dst, reason=None))

    apply_actions(actions, link_mode, config.dry_run, logger)
