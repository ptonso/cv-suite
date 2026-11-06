"""structure: how much of what. Figures `01` to `07` and tables `structure.csv`, `box_sizes.csv`."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cvsuite.common.transforms.split import record_labels
from cvsuite.eda.core.sections import Context, split_order
from cvsuite.eda.core.style import bars, new_figure, save_figure, spans_orders, top_entries

QUANTILES = (5, 25, 50, 75, 95)


def needs(ctx: Context) -> str | None:
    return None if ctx.dataset.records else "no records"


def class_tables(ctx: Context) -> tuple[dict[str, Counter], dict[str, Counter], list[str]]:
    """Instances and images per class and split."""
    instances: dict[str, Counter] = defaultdict(Counter)
    images: dict[str, Counter] = defaultdict(Counter)
    for record in ctx.dataset.records:
        labels = record_labels(record, ctx.dataset.classes)
        split = record.split or "train"
        for name in labels:
            instances[name][split] += 1
        for name in set(labels):
            images[name][split] += 1
    return instances, images, split_order({r.split or "train" for r in ctx.dataset.records})


def box_stats(ctx: Context) -> dict[str, list[tuple[float, float, float, float]]]:
    """Per class: `(px size, area share, cx, cy)` for every box and polygon, a polygon and its derived box counted once."""
    out: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    classes = ctx.dataset.classes
    for record, row in zip(ctx.dataset.records, ctx.table.rows):
        if row["width"] == "":
            continue
        width, height = row["width"], row["height"]

        def name(item) -> str:
            return classes[item.cls] if 0 <= item.cls < len(classes) else (item.label or str(item.cls))

        for box in record.boxes:
            if box.kind != "seg":
                out[name(box)].append((math.sqrt(box.w * width * box.h * height), box.w * box.h, box.cx, box.cy))
        for poly in record.polys:
            xs, ys = [p[0] for p in poly.points], [p[1] for p in poly.points]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            out[name(poly)].append((math.sqrt(w * width * h * height), w * h, min(xs) + w / 2, min(ys) + h / 2))
    return out


def _tables(ctx: Context, base: Path, instances, images, splits, boxes) -> list[str]:
    classes = sorted(set(instances) | set(images), key=lambda c: (-sum(instances[c].values()), c))
    path = base / "tables" / "structure.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", *[f"{s}_{kind}" for s in splits for kind in ("images", "instances")], "images", "instances"])
        for name in classes:
            writer.writerow([name, *[v for s in splits for v in (images[name][s], instances[name][s])], sum(images[name].values()), sum(instances[name].values())])
    written = ["tables/structure.csv"]
    if boxes:
        path = base / "tables" / "box_sizes.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class", "n", *[f"px_q{q}" for q in QUANTILES], *[f"area_frac_q{q}" for q in QUANTILES]])
            for name in sorted(boxes, key=lambda c: (-len(boxes[c]), c)):
                arr = np.asarray(boxes[name])
                writer.writerow([name, len(arr), *[round(float(np.percentile(arr[:, 0], q)), 2) for q in QUANTILES], *[round(float(np.percentile(arr[:, 1], q)), 6) for q in QUANTILES]])
        written.append("tables/box_sizes.csv")
    return written


def _class_counts(ctx, base, instances) -> str:
    totals = {name: float(sum(c.values())) for name, c in instances.items()}
    entries, _ = top_entries(totals, ctx.top)
    fig, ax = new_figure(7, 0.28 * len(entries) + 1.4)
    bars(ax, entries, log=spans_orders([v for _, v in entries]))
    ax.set_title(f"How many instances does each class have? (top {ctx.top} of {len(totals)})")
    ax.set_xlabel("instances")
    save_figure(fig, base / "figures" / "01_class_counts.png")
    return "figures/01_class_counts.png"


def _class_by_split(ctx, base, instances, splits) -> str | None:
    if len(splits) < 2:
        ctx.omit("figures/02_class_by_split.png", "the dataset has a single split")
        return None
    totals = {name: float(sum(c.values())) for name, c in instances.items()}
    kept, _ = top_entries(totals, ctx.top)
    names = [n for n, _ in kept if not n.startswith("other (")]
    rows = [(n, [float(instances[n][s]) for s in splits]) for n in names]
    if len(kept) > len(names):
        rest = [c for c in instances if c not in names]
        rows.append((kept[-1][0], [float(sum(instances[c][s] for c in rest)) for s in splits]))
    fig, ax = new_figure(8, 4.5)
    width = 0.8 / len(splits)
    positions = np.arange(len(rows))
    for j, split in enumerate(splits):
        ax.bar(positions + j * width, [max(v[j], 0.5) for _, v in rows], width, label=split)
    ax.set_xticks(positions + 0.4 - width / 2, [n for n, _ in rows], rotation=60, ha="right")
    ax.set_yscale("log")
    ax.set_title(f"How are the classes spread across splits? (top {ctx.top} of {len(instances)})")
    ax.set_ylabel("instances (log)")
    ax.legend()
    save_figure(fig, base / "figures" / "02_class_by_split.png")
    return "figures/02_class_by_split.png"


def _long_tail(ctx, base, instances) -> str:
    counts = sorted((sum(c.values()) for c in instances.values()), reverse=True)
    fig, ax = new_figure(6, 4)
    ax.loglog(range(1, len(counts) + 1), counts, marker="o", ms=3)
    ax.set_title(f"How long is the tail? ({len(counts)} classes, rank against instances)")
    ax.set_xlabel("class rank (log)")
    ax.set_ylabel("instances (log)")
    save_figure(fig, base / "figures" / "03_long_tail.png")
    return "figures/03_long_tail.png"


def _box_size(ctx, base, boxes) -> str:
    totals = {n: float(len(v)) for n, v in boxes.items()}
    kept, _ = top_entries(totals, ctx.top)
    names = [n for n, _ in kept if not n.startswith("other (")]
    groups = [(n, np.asarray(boxes[n])) for n in names]
    if len(kept) > len(names):
        groups.append((kept[-1][0], np.asarray([b for c, v in boxes.items() if c not in names for b in v])))
    groups.append(("all", np.asarray([b for v in boxes.values() for b in v])))
    plt_fig, (ax_px, ax_frac) = _two_panels(len(groups))
    for ax, column, label in ((ax_px, 0, "box size sqrt(w*h), pixels"), (ax_frac, 1, "box area, share of the image")):
        log = spans_orders([float(np.percentile(arr[:, column], q)) for _, arr in groups for q in QUANTILES], factor=20.0)
        for y, (name, arr) in enumerate(groups):
            lo, q1, med, q3, hi = (float(np.percentile(arr[:, column], q)) for q in QUANTILES)
            ax.plot([max(lo, 1e-6), hi], [y, y], color="#888888", lw=1)
            ax.plot([max(q1, 1e-6), q3], [y, y], color="#4c78a8", lw=4)
            ax.plot([max(med, 1e-6)], [y], "o", color="#f58518")
        ax.set_xscale("log" if log else "linear")
        ax.set_xlabel(label)
    ax_px.set_yticks(range(len(groups)), [n if n.startswith("other (") else f"{n} ({len(a)})" for n, a in groups])
    ax_px.invert_yaxis()
    plt_fig.suptitle(f"How big are the boxes? (median, quartiles, 5-95%; top {ctx.top} classes)")
    save_figure(plt_fig, base / "figures" / "04_box_size.png")
    return "figures/04_box_size.png"


def _two_panels(rows: int):
    from cvsuite.eda.core.style import apply_style, ensure_matplotlib

    plt = ensure_matplotlib()
    apply_style()
    return plt.subplots(1, 2, figsize=(10, 0.3 * rows + 1.6), sharey=True)


def _instances_per_image(ctx, base) -> str:
    counts = Counter(row["n_instances"] for row in ctx.table.rows)
    entries = sorted(counts.items())
    fig, ax = new_figure(6, 3.6)
    ax.bar([k for k, _ in entries], [v for _, v in entries], color="#4c78a8")
    ax.set_yscale("log")
    ax.set_title("How many instances does an image hold?")
    ax.set_xlabel("instances per image")
    ax.set_ylabel("images (log)")
    save_figure(fig, base / "figures" / "05_instances_per_image.png")
    return "figures/05_instances_per_image.png"


def _box_centers(ctx, base, boxes) -> str:
    arr = np.asarray([(b[2], b[3]) for v in boxes.values() for b in v])
    fig, ax = new_figure(5, 4.4)
    image = ax.hist2d(arr[:, 0], arr[:, 1], bins=40, range=[[0, 1], [0, 1]], cmap="viridis")[3]
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_title("Where do the boxes sit in the image?")
    ax.set_xlabel("x center (share of width)")
    ax.set_ylabel("y center (share of height)")
    fig.colorbar(image, ax=ax, label="boxes")
    save_figure(fig, base / "figures" / "06_box_centers.png")
    return "figures/06_box_centers.png"


def _resolutions(ctx, base) -> str | None:
    rows = [r for r in ctx.table.rows if r["width"] != ""]
    if not rows:
        ctx.omit("figures/07_resolutions.png", "no readable image")
        return None
    mix = Counter(f"{r['width']}x{r['height']}" for r in rows)
    entries, _ = top_entries({k: float(v) for k, v in mix.items()}, ctx.top)
    ratios = [r["width"] / r["height"] for r in rows]
    from cvsuite.eda.core.style import apply_style, ensure_matplotlib

    plt = ensure_matplotlib()
    apply_style()
    fig, (ax_mix, ax_ratio) = plt.subplots(1, 2, figsize=(11, 0.28 * len(entries) + 1.8))
    bars(ax_mix, entries, log=spans_orders([v for _, v in entries]))
    ax_mix.set_title(f"Which resolutions occur? (top {ctx.top} of {len(mix)})")
    ax_ratio.hist(ratios, bins=30, color="#4c78a8")
    ax_ratio.axvline(1.0, color="#888888", ls="--")
    ax_ratio.set_title("Aspect ratio (width / height); dashed = square")
    save_figure(fig, base / "figures" / "07_resolutions.png")
    return "figures/07_resolutions.png"


def write(ctx: Context, base: Path) -> list[str]:
    instances, images, splits = class_tables(ctx)
    boxes = box_stats(ctx)
    written = _tables(ctx, base, instances, images, splits, boxes)
    figures: list[str | None] = []
    if instances:
        figures += [_class_counts(ctx, base, instances), _class_by_split(ctx, base, instances, splits), _long_tail(ctx, base, instances)]
    else:
        ctx.omit("figures/01_class_counts.png", "no labels")
    if boxes:
        figures += [_box_size(ctx, base, boxes), _instances_per_image(ctx, base), _box_centers(ctx, base, boxes)]
    else:
        for name in ("04_box_size.png", "05_instances_per_image.png", "06_box_centers.png"):
            ctx.omit(f"figures/{name}", "the dataset has no boxes or polygons")
    figures.append(_resolutions(ctx, base))
    return written + [f for f in figures if f]
