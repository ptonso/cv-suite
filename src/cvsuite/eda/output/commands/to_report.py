"""Write an exploratory report: figures and tables per section, one image table and a run summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvsuite.common.io.common import OutputClaims
from cvsuite.common.io.folds import require_single_fold
from cvsuite.eda.core import report as report_mod
from cvsuite.eda.core.sections import NAMES, Context, safe_name as _safe, sections
from cvsuite.eda.core.split import resolve_threshold
from cvsuite.eda.core.table import build_table

GROUPED = ("profile", "structure")


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("out", type=Path, help="Report folder to create.")
    parser.add_argument("--group-by", action="append", default=[], metavar="KEY", help="Key spec (regex:, dir:, attr:, file:, similar:). Adds per-group panels for the largest groups.")
    parser.add_argument("--top", type=int, default=20, help="Largest classes or groups drawn per view; the rest become one `other (k)` entry (default 20).")
    parser.add_argument("--only", default=None, metavar="SECTION[,SECTION...]", help=f"Run only these sections: {', '.join(NAMES)}.")
    parser.add_argument("--measure", choices=["hash", "ssim"], default="hash", help="Similarity for the split section.")
    parser.add_argument("--twin-threshold", type=float, default=None, help="Twin threshold: bits for hash (default 6), an index for ssim (default 0.6).")
    parser.add_argument("--max-images", type=int, default=20000, help="Bound on the records a pairwise section compares; larger sets are sampled.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the samples and baselines.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the section folders and files this run writes.")


def _selected(only: str | None) -> list[str]:
    if not only:
        return list(NAMES)
    chosen = [s.strip() for s in only.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in NAMES]
    if unknown:
        raise SystemExit(f"Unknown section(s) in --only: {', '.join(unknown)}. Valid: {', '.join(NAMES)}.")
    return [s for s in NAMES if s in chosen]


def _group_panels(ctx: Context, module, base: Path) -> list[str]:
    sizes: dict[str, list[int]] = {}
    for i, row in enumerate(ctx.table.rows):
        sizes.setdefault(row["group"], []).append(i)
    written = []
    for group in sorted(sizes, key=lambda g: (-len(sizes[g]), g))[: ctx.top]:
        sub = ctx.subset(sizes[group])
        folder = base / "by_group" / _safe(group)
        written += [f"by_group/{_safe(group)}/{f}" for f in module.write(sub, folder)]
    return written


def run(dataset, args: argparse.Namespace):
    dataset = require_single_fold(dataset)
    chosen = _selected(args.only)
    if args.top < 1:
        raise SystemExit("--top must be at least 1.")
    out: Path = args.out
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} already holds output; pass --overwrite to replace what this run writes.")
    resolve_threshold(args.measure, args.twin_threshold)
    registry = sections()
    ctx_args = dict(top=args.top, seed=args.seed, group_by=list(args.group_by), max_images=args.max_images, measure=args.measure, twin_threshold=args.twin_threshold)

    table = build_table(dataset, list(args.group_by))
    ctx = Context(dataset=dataset, table=table, **ctx_args)
    results: dict[str, dict] = {}
    runnable: list[str] = []
    for name in NAMES:
        module, needs = registry[name]
        reason = needs(ctx)
        if reason is not None:
            if name in chosen and args.only:
                raise SystemExit(f"Section `{name}` cannot run: {reason}.")
            results[name] = {"status": "skipped", "reason": reason}
        elif name in chosen:
            runnable.append(name)
        else:
            results[name] = {"status": "skipped", "reason": "not selected by --only"}

    claims = OutputClaims(overwrite=args.overwrite)
    out.mkdir(parents=True, exist_ok=True)
    claims.claim(out / "images.csv")
    claims.claim(out / "report.yaml")
    for name in runnable:
        claims.claim(out / name)
    table.write(out / "images.csv", with_group=bool(args.group_by))
    for name in runnable:
        module = registry[name][0]
        files = module.write(ctx, out / name)
        if args.group_by and name in GROUPED:
            files += _group_panels(ctx, module, out / name)
        results[name] = {"status": "ran", "files": sorted(files)}

    ordered = {name: results[name] for name in NAMES}
    options = {
        "group_by": list(args.group_by), "top": args.top, "only": args.only, "measure": args.measure,
        "twin_threshold": args.twin_threshold, "max_images": args.max_images, "seed": args.seed,
    }
    report_mod.write_report(out / "report.yaml", report_mod.build_report(dataset, ctx, options, ordered))
    for name, result in ordered.items():
        if result["status"] == "skipped":
            print(f"skipped {name}: {result['reason']}")
    print(f"wrote {out}")
    return dataset
