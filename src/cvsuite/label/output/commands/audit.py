"""Run audits on a VisionDataset and print or write a YAML report."""

from pathlib import Path
import argparse
import yaml

from cvsuite.label.core.audit import run_checks
from cvsuite.common.core.utils import write_yaml
from cvsuite.common.core import VisionDataset


def attach(p: argparse.ArgumentParser) -> None:
    p.add_argument("dst", nargs="?", type=Path, help="Optional YAML report path; prints to stdout when omitted.")
    p.add_argument("--split", default=None, help="Comma-separated splits to include (default: all).")
    p.add_argument("--thresholds", type=str, default="", help="Optional comma-separated key=value overrides for audit thresholds.")


def run(ds: VisionDataset, a: argparse.Namespace):
    splits = [s.strip() for s in a.split.split(",") if s.strip()] if a.split else None
    thresholds = {} if not a.thresholds else dict(kv.split("=") for kv in a.thresholds.split(",") if "=" in kv)
    subset = ds
    if splits:
        subset = VisionDataset(
            records=[r for r in ds.records if r.split in splits],
            classes=ds.classes,
            task=ds.task,
            root=ds.root,
            meta=ds.meta,
        )
    report = run_checks(subset, thresholds=thresholds)

    if a.dst:
        dst = a.dst
        if dst.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            dst = dst / "audit.yaml"
        elif dst.suffix.lower() not in {".yml", ".yaml"}:
            dst = dst.with_suffix(".yaml")
            dst.parent.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(dst, report, overwrite=True)
    else:
        print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    return ds
