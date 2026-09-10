from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from cvsuite.common.core import VisionDataset
from cvsuite.common.core.utils import write_yaml

_PREFERRED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ExportStatsContext:
    branch: str
    command: str
    dst: Path
    stats_path: Path
    primary_artifacts: list[str] = field(default_factory=list)
    writer_details: dict[str, Any] = field(default_factory=dict)
    run_meta: dict[str, Any] = field(default_factory=dict)


def resolve_stats_path(dst: Path) -> Path:
    return dst.parent / "stats.yaml" if dst.suffix else dst / "stats.yaml"


def make_export_context(
    *,
    branch: str,
    command: str,
    dst: Path,
    primary_artifacts: Iterable[Path | str] | None = None,
    writer_details: Mapping[str, Any] | None = None,
    run_meta: Mapping[str, Any] | None = None,
) -> ExportStatsContext:
    dst_path = Path(dst)
    return ExportStatsContext(
        branch=str(branch),
        command=str(command),
        dst=dst_path,
        stats_path=resolve_stats_path(dst_path),
        primary_artifacts=[str(Path(item)) for item in (primary_artifacts or [])],
        writer_details=dict(writer_details or {}),
        run_meta=dict(run_meta or {}),
    )


def emit_stats_yaml(stats: dict[str, Any], ctx: ExportStatsContext) -> Path:
    payload = prune_empty(stats)
    if not isinstance(payload, dict):
        raise TypeError(f"stats payload must be a mapping, got {type(payload)!r}")
    write_yaml(ctx.stats_path, payload, overwrite=True)
    return ctx.stats_path


def preferred_counts(raw: Mapping[str, int]) -> dict[str, int]:
    ordered: dict[str, int] = {}
    seen: set[str] = set()
    for name in _PREFERRED_SPLITS:
        value = int(raw.get(name, 0))
        if value <= 0:
            continue
        ordered[name] = value
        seen.add(name)
    for name in sorted(raw):
        if name in seen:
            continue
        value = int(raw[name])
        if value <= 0:
            continue
        ordered[name] = value
    return ordered


def records_by_split(dataset: VisionDataset) -> dict[str, int]:
    return preferred_counts({name: len(records) for name, records in dataset.by_split().items()})


def count_values(values: Iterable[str | None]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        counts[text] += 1
    return preferred_counts(counts) if set(counts).issubset(set(_PREFERRED_SPLITS)) else dict(sorted(counts.items()))


def score_summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    scores = [float(value) for value in values if value is not None]
    if not scores:
        return {"available": 0}
    return {
        "available": len(scores),
        "min": round(min(scores), 6),
        "mean": round(sum(scores) / len(scores), 6),
        "max": round(max(scores), 6),
    }


def task_name(task: object) -> str | None:
    if task is None:
        return None
    value = getattr(task, "value", task)
    text = str(value or "").strip()
    return text or None


def label_name(label: object, cls_idx: object, classes: list[str]) -> str:
    text = str(label or "").strip()
    if text:
        return text
    if isinstance(cls_idx, int) and 0 <= cls_idx < len(classes):
        return str(classes[cls_idx])
    return str(cls_idx)


def image_key(record) -> str:
    rel = getattr(record, "rel_image_path", None)
    if rel is not None:
        return str(rel)
    return str(record.image.path)


def base_stats_document(dataset: VisionDataset, ctx: ExportStatsContext) -> dict[str, Any]:
    output: dict[str, Any] = {"root": str(ctx.stats_path.parent)}
    if ctx.primary_artifacts:
        output["primary_artifacts"] = list(ctx.primary_artifacts)
    return {
        "schema_version": 1,
        "branch": ctx.branch,
        "command": ctx.command,
        "output": output,
        "dataset": {
            "records": len(dataset.records),
            "splits": records_by_split(dataset),
        },
    }


def prune_empty(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        value = dict(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            pruned = prune_empty(item)
            if pruned is None:
                continue
            if pruned == {} or pruned == [] or pruned == "":
                continue
            out[str(key)] = pruned
        return out
    if isinstance(value, (list, tuple, set)):
        out_list = []
        for item in value:
            pruned = prune_empty(item)
            if pruned is None:
                continue
            if pruned == {} or pruned == [] or pruned == "":
                continue
            out_list.append(pruned)
        return out_list
    return value
