"""`report.yaml`: what the run did. Contract: `specs/eda/report.md#layout`."""

from __future__ import annotations

from pathlib import Path

import yaml

from cvsuite.common.core import VisionDataset
from cvsuite.eda.core.sections import Context


def build_report(dataset: VisionDataset, ctx: Context, options: dict, results: dict[str, dict]) -> dict:
    return {
        "sources": list(dataset.meta.get("sources") or [str(dataset.root)]),
        "records": len(dataset.records),
        "steps": list(dataset.meta.get("steps", [])),
        "options": options,
        "sections": results,
        "unreadable_images": [{"path": p, "error": e} for p, e in ctx.table.unreadable],
        "figures_not_drawn": ctx.omitted,
        "samples": ctx.samples,
        "notes": ctx.notes,
    }


def write_report(path: Path, report: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(report, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
