"""split: can the split be trusted. Figures `01` to `06` and tables `twins.csv`, `class_balance.csv`, `units.csv`, `missing_classes.csv`, `straddle.csv`."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cvsuite.common import similarity
from cvsuite.common.io.common import resolve_record_image_path
from cvsuite.common.sampling import take
from cvsuite.common.transforms.split import EMPTY, record_labels
from cvsuite.eda.core.sections import Context, needs_split, safe_name, sampling_keys, split_order
from cvsuite.eda.core.structure import class_tables
from cvsuite.eda.core.style import BASELINE, OTHER, PALETTE, apply_style, ensure_matplotlib, save_figure, top_entries, write_image
from cvsuite.eda.core.table import hash_array

DEFAULT_BITS = 6
DEFAULT_SSIM = 0.6
CANDIDATES = 25
EXAMPLE_PERCENTS = (0, 1, 5, 25, 75)
EXAMPLE_SIZE = 256
LOW_ROWS = (5, 20)
Z95 = 1.96


def resolve_threshold(measure: str, value: float | None) -> float:
    """The twin threshold in the unit of the measure: whole bits for `hash`, an index for `ssim`."""
    try:
        if measure == "hash":
            if value is not None and value != int(value):
                raise ValueError(f"A hash threshold is a whole number of bits, got {value}.")
            return float(similarity.check_bits(DEFAULT_BITS if value is None else int(value)))
        return similarity.check_ssim(DEFAULT_SSIM if value is None else float(value))
    except ValueError as exc:
        raise SystemExit(f"--twin-threshold: {exc}") from None


def floor_score(measure: str, threshold: float) -> float:
    """The smallest score that counts as a twin: matching bits for `hash`, the index for `ssim`."""
    return similarity.MAX_BITS - threshold if measure == "hash" else threshold


def eval_splits(ctx: Context) -> list[str]:
    return [s for s in split_order({r["split"] for r in ctx.table.rows}) if s != "train"]


def needs(ctx: Context) -> str | None:
    reason = needs_split(ctx)
    if reason:
        return reason
    readable = {ctx.table.rows[i]["split"] for i in ctx.table.readable()}
    if "train" not in readable or not set(eval_splits(ctx)) & readable:
        return "needs a readable image in train and in another split"
    return None


@dataclass
class Twins:
    """Best similarity of each row of one split to the reference rows: table indices, scores, closest reference rows."""

    split: str
    rows: np.ndarray
    scores: np.ndarray
    closest: np.ndarray

    def ranked(self) -> np.ndarray:
        return -np.sort(-self.scores)


def _bits(ctx: Context, indices: np.ndarray) -> np.ndarray:
    return hash_array([ctx.table.rows[i]["phash"] for i in indices])


def measure_twins(ctx: Context, split: str, rows: np.ndarray, reference: np.ndarray) -> Twins:
    """Best similarity of `rows` to `reference`: the hash distance, or SSIM over the nearest hash candidates."""
    if ctx.measure == "hash":
        distance, index = similarity.nearest(_bits(ctx, rows), _bits(ctx, reference), 1)
        return Twins(split, rows, (similarity.MAX_BITS - distance[:, 0]).astype(float), reference[index[:, 0]])
    _, candidates = similarity.nearest(_bits(ctx, rows), _bits(ctx, reference), CANDIDATES)
    scores, closest = np.empty(len(rows)), np.empty(len(rows), dtype=np.int64)
    for n, (row, near) in enumerate(zip(rows, candidates)):
        found = [similarity.ssim(ctx.table.thumbs[row], ctx.table.thumbs[reference[j]]) for j in near]
        best = int(np.argmax(found))
        scores[n], closest[n] = found[best], reference[near[best]]
    return Twins(split, rows, scores, closest)


def _sample(ctx: Context, name: str, indices: np.ndarray, keys: list[str]) -> np.ndarray:
    if len(indices) <= ctx.max_images:
        return indices
    chosen = take([keys[i] for i in indices], ctx.seed, ctx.max_images)
    ctx.samples.append({"section": "split", "split": name, "of": len(indices), "taken": len(chosen), "seed": ctx.seed, "method": "nested hash sampling"})
    return indices[chosen]


def _rows_by_split(ctx: Context) -> dict[str, np.ndarray]:
    readable = set(ctx.table.readable())
    out: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(ctx.table.rows):
        if i in readable:
            out[row["split"]].append(i)
    return {s: np.asarray(v, dtype=np.int64) for s, v in out.items()}


def _share(scores: np.ndarray, floor: float) -> float:
    return float((scores >= floor).mean()) if len(scores) else 0.0


def _measure_all(ctx: Context) -> tuple[dict[str, Twins], dict[str, Twins], str]:
    """Twins of every eval split against train, and the same measurement on a random split of the same sizes."""
    keys = sampling_keys(ctx.dataset)
    by_split = {s: _sample(ctx, s, rows, keys) for s, rows in _rows_by_split(ctx).items()}
    evals = [s for s in eval_splits(ctx) if s in by_split]
    twins = {s: measure_twins(ctx, s, by_split[s], by_split["train"]) for s in evals}
    pool = np.concatenate([by_split[s] for s in ["train", *evals]])
    shuffled = pool[np.random.default_rng(ctx.seed).permutation(len(pool))]
    cuts = np.cumsum([len(by_split[s]) for s in ["train", *evals]])
    parts = dict(zip(["train", *evals], np.split(shuffled, cuts[:-1])))
    baseline = {s: measure_twins(ctx, s, parts[s], parts["train"]) for s in evals}
    return twins, baseline, "train"


def _plt():
    plt = ensure_matplotlib()
    apply_style()
    return plt


def _curve_axes(ax, ctx: Context, curves: dict[str, np.ndarray], baselines: dict[str, np.ndarray], floor: float, title: str) -> None:
    unit = "matching bits of 64" if ctx.measure == "hash" else "SSIM"
    for k, (name, scores) in enumerate(curves.items()):
        if len(scores):
            ax.plot(np.linspace(0, 100, len(scores)), scores, color=PALETTE[k % len(PALETTE)], label=f"{name} (n={len(scores)})")
        if name in baselines and len(baselines[name]):
            ax.plot(np.linspace(0, 100, len(baselines[name])), baselines[name], color=PALETTE[k % len(PALETTE)], ls="--", alpha=0.7, label=f"random split, {name}")
    ax.axhline(floor, color=BASELINE, ls=":", label=f"threshold ({floor:g})")
    if ctx.measure == "hash":
        ax.set_ylim(top=similarity.MAX_BITS + 0.5)
    ax.set_xlabel("eval images, most similar first (%)")
    ax.set_ylabel(f"best similarity to train ({unit})")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


def _twin_title(ctx: Context, twins: dict[str, np.ndarray], baselines: dict[str, np.ndarray], floor: float, prefix: str = "") -> str:
    parts = [f"{s} {100 * _share(v, floor):.1f}% (random {100 * _share(baselines[s], floor):.1f}%)" for s, v in twins.items()]
    return f"{prefix}Share of eval images at or above the twin threshold: " + ", ".join(parts)


def _twin_rank(ctx, base, twins, baseline, floor) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ranked = {s: t.ranked() for s, t in twins.items()}
    base_ranked = {s: t.ranked() for s, t in baseline.items()}
    _curve_axes(ax, ctx, ranked, base_ranked, floor, _twin_title(ctx, ranked, base_ranked, floor))
    save_figure(fig, base / "figures" / "01_twin_rank.png")
    return "figures/01_twin_rank.png"


def _by_group_curves(ctx, base, twins, baseline, floor) -> list[str]:
    sizes: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(ctx.table.rows):
        sizes[row["group"]].append(i)
    written = []
    plt = _plt()
    for group in sorted(sizes, key=lambda g: (-len(sizes[g]), g))[: ctx.top]:
        members = set(sizes[group])
        mine = {s: np.sort(t.scores[[r in members for r in t.rows]])[::-1] for s, t in twins.items()}
        mine = {s: v for s, v in mine.items() if len(v)}
        relative = f"by_group/{safe_name(group)}/figures/01_twin_rank.png"
        if not mine:
            ctx.omit(relative, f"group {group} has no eval images")
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        _curve_axes(ax, ctx, mine, {s: baseline[s].ranked() for s in mine}, floor, _twin_title(ctx, mine, {s: baseline[s].ranked() for s in mine}, floor, f"Group {group}: "))
        save_figure(fig, base / relative)
        written.append(relative)
    return written


def _twins_table(ctx, base, twins, floor) -> str:
    path = base / "tables" / "twins.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "split", "class", "closest_train", "score", "is_twin"])
        for name, t in twins.items():
            for n in np.argsort(-t.scores, kind="stable"):
                row = ctx.table.rows[t.rows[n]]
                score = int(t.scores[n]) if ctx.measure == "hash" else round(float(t.scores[n]), 4)
                writer.writerow([row["path"], name, row["class"], ctx.table.rows[t.closest[n]]["path"], score, int(t.scores[n] >= floor)])
    return "tables/twins.csv"


def _color(ctx: Context, index: int) -> np.ndarray:
    image = cv2.imread(str(resolve_record_image_path(ctx.dataset, ctx.dataset.records[index])))
    if image is None:
        return np.zeros((EXAMPLE_SIZE, EXAMPLE_SIZE, 3), dtype=np.uint8)
    return cv2.resize(image, (EXAMPLE_SIZE, EXAMPLE_SIZE), interpolation=cv2.INTER_AREA)


def _twin_examples(ctx, base, twins) -> str:
    rows = np.concatenate([t.rows for t in twins.values()])
    scores = np.concatenate([t.scores for t in twins.values()])
    closest = np.concatenate([t.closest for t in twins.values()])
    names = [s for s, t in twins.items() for _ in t.rows]
    order = np.argsort(-scores, kind="stable")
    percent: dict[int, int] = {}
    for p in EXAMPLE_PERCENTS:
        percent.setdefault(int(order[round(p / 100 * (len(order) - 1))]), p)
    picks = list(percent)
    label_width = 240
    canvas = np.zeros((len(picks) * EXAMPLE_SIZE, label_width + 2 * EXAMPLE_SIZE, 3), dtype=np.uint8)
    for r, pick in enumerate(picks):
        top = r * EXAMPLE_SIZE
        score = f"{scores[pick]:.0f}/64 bits" if ctx.measure == "hash" else f"SSIM {scores[pick]:.3f}"
        lines = [f"rank {percent[pick]}% of eval", f"{names[pick]}: {ctx.table.rows[rows[pick]]['path'].rsplit('/', 1)[-1][:24]}", f"train: {ctx.table.rows[closest[pick]]['path'].rsplit('/', 1)[-1][:24]}", score]
        for k, line in enumerate(lines):
            cv2.putText(canvas, line, (6, top + 40 + 36 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[top : top + EXAMPLE_SIZE, label_width : label_width + EXAMPLE_SIZE] = _color(ctx, int(rows[pick]))
        canvas[top : top + EXAMPLE_SIZE, label_width + EXAMPLE_SIZE :] = _color(ctx, int(closest[pick]))
    write_image(base / "figures" / "02_twin_examples.jpg", canvas)
    return "figures/02_twin_examples.jpg"


def _balance(ctx: Context, base: Path) -> list[str]:
    instances, _, splits = class_tables(ctx)
    if not instances:
        ctx.omit("figures/03_class_balance.png", "no labels")
        return []
    totals = {s: float(sum(c[s] for c in instances.values())) for s in splits}
    grand = sum(totals.values())
    share_all = {s: totals[s] / grand for s in splits}
    class_total = {n: float(sum(c.values())) for n, c in instances.items()}
    path = base / "tables" / "class_balance.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    deviation: dict[str, float] = {}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "split", "instances", "class_share", "split_share", "deviation"])
        for name in sorted(instances, key=lambda c: (-class_total[c], c)):
            deviation[name] = max(abs(instances[name][s] / class_total[name] - share_all[s]) for s in splits)
            for s in splits:
                share = instances[name][s] / class_total[name]
                writer.writerow([name, s, instances[name][s], round(share, 4), round(share_all[s], 4), round(share - share_all[s], 4)])
    kept, _ = top_entries(class_total, ctx.top)
    named = [n for n, _ in kept if not n.startswith("other (")]
    rows = [(n, [instances[n][s] for s in splits]) for n in named]
    if len(kept) > len(named):
        rest = [c for c in instances if c not in named]
        rows.append((kept[-1][0], [sum(instances[c][s] for c in rest) for s in splits]))
    plt = _plt()
    fig, (ax, hist) = plt.subplots(1, 2, figsize=(12, 0.3 * len(rows) + 2.6), gridspec_kw={"width_ratios": [3, 2]}, layout="constrained")
    for r, (name, counts) in enumerate(rows):
        left, total = 0.0, float(sum(counts)) or 1.0
        for k, (s, count) in enumerate(zip(splits, counts)):
            width = count / total
            ax.barh(r, width, left=left, color=OTHER if name.startswith("other (") and k == 0 else PALETTE[k % len(PALETTE)], label=s if r == 0 else None)
            if width > 0.05:
                ax.text(left + width / 2, r, f"{count}", ha="center", va="center", fontsize=7, color="white")
            left += width
    edge = 0.0
    for s in splits[:-1]:
        edge += share_all[s]
        ax.axvline(edge, color="black", ls="--", lw=0.8)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the class's instances (dashed: each split's share of all instances)")
    ax.set_title(f"Does each class follow the split shares? ({min(ctx.top, len(instances))} of {len(instances)} classes)", fontsize=9)
    ax.legend(fontsize=7, ncol=len(splits), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    hist.hist(list(deviation.values()), bins=20, color=PALETTE[0])
    hist.set_xlabel("largest gap between a split's share of the class and of all instances")
    hist.set_ylabel("classes")
    hist.set_title(f"All {len(deviation)} classes", fontsize=9)
    save_figure(fig, base / "figures" / "03_class_balance.png")
    return ["tables/class_balance.csv", "figures/03_class_balance.png"]


def _half_width(units: int) -> float | None:
    return Z95 * math.sqrt(0.25 / units) if units else None


def _units(ctx: Context, base: Path, evals: dict[str, np.ndarray]) -> list[str]:
    link = int(ctx.twin_threshold) if ctx.measure == "hash" else DEFAULT_BITS
    per_split = {}
    for name, rows in evals.items():
        comp = similarity.components(len(rows), similarity.pairs_within(_bits(ctx, rows), link))
        members: dict[str, set[int]] = defaultdict(set)
        for n, i in enumerate(rows):
            for label in set(record_labels(ctx.dataset.records[i], ctx.dataset.classes)):
                members[label].add(n)
        per_split[name] = (members, comp)
    ctx.notes.setdefault("split", {})["unit_link_bits"] = link
    pooled_rows = defaultdict(int)
    for members, _ in per_split.values():
        for c, m in members.items():
            pooled_rows[c] += len(m)
    if not pooled_rows:
        ctx.omit("figures/04_independent_units.png", "no labeled eval images")
        return []
    path = base / "tables" / "units.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table: dict[str, list[tuple[str, int, int, float | None]]] = {}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "class", "rows", "units", "half_width_95"])
        for name, (members, comp) in per_split.items():
            entries = [("(all)", len(comp), len(set(comp)))]
            entries += [(c, len(m), len({comp[n] for n in m})) for c, m in sorted(members.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
            table[name] = [(c, r, u, _half_width(u)) for c, r, u in entries]
            for c, r, u in entries:
                hw = _half_width(u)
                writer.writerow([name, c, r, u, "" if hw is None else round(hw, 4)])
    kept, _ = top_entries({c: float(v) for c, v in pooled_rows.items()}, ctx.top)
    named = [n for n, _ in kept if not n.startswith("other (")]
    plt = _plt()
    fig = plt.figure(figsize=(5.5 * len(per_split), 0.32 * (len(named) + 2) + 4.6), layout="constrained")
    grid = fig.add_gridspec(2, len(per_split), height_ratios=[max(2.0, 0.32 * (len(named) + 2)), 1.6])
    for col, (name, (members, comp)) in enumerate(per_split.items()):
        ax = fig.add_subplot(grid[0, col])
        rest = set().union(*[m for c, m in members.items() if c not in named]) if len(kept) > len(named) else set()
        rows = [("(all)", len(comp), len(set(comp)))] + [(c, len(members.get(c, ())), len({comp[n] for n in members.get(c, ())})) for c in named]
        if len(kept) > len(named):
            rows.append((kept[-1][0], len(rest), len({comp[n] for n in rest})))
        positions = np.arange(len(rows))
        ax.barh(positions, [r for _, r, _ in rows], color=OTHER, label="images")
        ax.barh(positions, [u for _, _, u in rows], color=PALETTE[0], height=0.5, label="independent units")
        for p, (_, r, u) in zip(positions, rows):
            hw = _half_width(u)
            ax.text(r, p, f" {u}/{r}" + ("" if hw is None else f" ±{100 * hw:.1f}%"), va="center", fontsize=7)
        ax.set_yticks(positions, [c for c, _, _ in rows])
        ax.invert_yaxis()
        ax.set_title(f"{name}: units/images and worst-case 95% half-width ({min(ctx.top, len(pooled_rows))} of {len(pooled_rows)} classes)", fontsize=8)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax = fig.add_subplot(grid[1, :])
    for k, (name, entries) in enumerate(table.items()):
        widths = [100 * hw for c, _, _, hw in entries if c != "(all)" and hw is not None]
        ax.hist(widths, bins=20, alpha=0.6, color=PALETTE[k % len(PALETTE)], label=f"{name} ({len(widths)} classes)")
    ax.set_xlabel("worst-case 95% half-width of an accuracy on a class's units (percentage points)")
    ax.set_ylabel("classes")
    ax.set_title("All classes", fontsize=9)
    ax.legend(fontsize=7)
    save_figure(fig, base / "figures" / "04_independent_units.png")
    return ["tables/units.csv", "figures/04_independent_units.png"]


def _missing(ctx: Context, base: Path, evals: list[str]) -> list[str]:
    universe = {c for c in ctx.dataset.classes}
    per_split: dict[str, dict[str, int]] = {s: defaultdict(int) for s in evals}
    for record, row in zip(ctx.dataset.records, ctx.table.rows):
        labels = set(record_labels(record, ctx.dataset.classes)) - {EMPTY}
        universe |= labels
        if row["split"] in per_split:
            for c in labels:
                per_split[row["split"]][c] += 1
    if not universe:
        ctx.omit("figures/05_missing_classes.png", "no labels")
        return []
    buckets = (("0", 1), (f"<{LOW_ROWS[0]}", LOW_ROWS[0]), (f"<{LOW_ROWS[1]}", LOW_ROWS[1]))
    path = base / "tables" / "missing_classes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, list[int]] = {}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "bucket", "classes", "names"])
        for s in evals:
            counts[s] = []
            for label, limit in buckets:
                names = sorted(c for c in universe if per_split[s][c] < limit)
                counts[s].append(len(names))
                writer.writerow([s, label, len(names), ";".join(names) if label == "0" else ""])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.8 / len(evals)
    for k, s in enumerate(evals):
        pos = np.arange(len(buckets)) + k * width
        ax.bar(pos, counts[s], width, color=PALETTE[k % len(PALETTE)], label=s)
        for p, v in zip(pos, counts[s]):
            ax.text(p, v, f"{v}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(buckets)) + 0.4 - width / 2, [f"{b} rows" for b, _ in buckets])
    ax.set_ylabel(f"classes (of {len(universe)})")
    ax.set_title("How many classes are missing or rare in each eval split?", fontsize=9)
    ax.legend()
    save_figure(fig, base / "figures" / "05_missing_classes.png")
    return ["tables/missing_classes.csv", "figures/05_missing_classes.png"]


def _straddle(ctx: Context, base: Path) -> list[str]:
    if not ctx.group_by:
        ctx.omit("figures/06_group_straddle.png", "no --group-by given")
        return []
    splits = split_order({r["split"] for r in ctx.table.rows})
    members: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in ctx.table.rows:
        members[row["group"]][row["split"]] += 1
    straddling = {g: m for g, m in members.items() if len(m) > 1}
    pairs: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for g, m in straddling.items():
        present = [s for s in splits if s in m]
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                key = f"{present[a]}+{present[b]}"
                pairs[key][0] += 1
                pairs[key][1] += m[present[a]] + m[present[b]]
    path = base / "tables" / "straddle.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(straddling, key=lambda g: (-sum(straddling[g].values()), g))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "name", "groups", "rows", *[f"rows_{s}" for s in splits]])
        for key, (groups, rows) in sorted(pairs.items()):
            writer.writerow(["pair", key, groups, rows, *[""] * len(splits)])
        for g in ordered:
            writer.writerow(["group", g, 1, sum(straddling[g].values()), *[straddling[g].get(s, 0) for s in splits]])
    plt = _plt()
    shown = ordered[: ctx.top]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, max(3.5, 0.3 * (len(shown) + 1) + 1.6)), gridspec_kw={"width_ratios": [2, 3]}, layout="constrained")
    names = sorted(pairs)
    positions = np.arange(len(names))
    left.barh(positions, [pairs[n][0] for n in names], color=PALETTE[0], label="groups")
    for p, n in zip(positions, names):
        left.text(pairs[n][0], p, f" {pairs[n][0]} groups, {pairs[n][1]} rows", va="center", fontsize=7)
    left.set_yticks(positions, names)
    left.set_title(f"{len(straddling)} of {len(members)} groups straddle splits", fontsize=9)
    left.set_xlabel("straddling groups per split pair")
    rest = ordered[ctx.top :]
    rows = [(g, [straddling[g].get(s, 0) for s in splits]) for g in shown]
    if rest:
        rows.append((f"other ({len(rest)})", [sum(straddling[g].get(s, 0) for g in rest) for s in splits]))
    for r, (name, counts) in enumerate(rows):
        start = 0
        for k, (s, count) in enumerate(zip(splits, counts)):
            right.barh(r, count, left=start, color=PALETTE[k % len(PALETTE)], label=s if r == 0 else None)
            if count:
                right.text(start + count / 2, r, str(count), ha="center", va="center", fontsize=7, color="white")
            start += count
    right.set_yticks(range(len(rows)), [n for n, _ in rows])
    right.invert_yaxis()
    right.set_xlabel("rows per split")
    right.set_title(f"The {len(shown)} largest straddling groups", fontsize=9)
    if rows:
        right.legend(fontsize=7)
    save_figure(fig, base / "figures" / "06_group_straddle.png")
    return ["tables/straddle.csv", "figures/06_group_straddle.png"]


def write(ctx: Context, base: Path) -> list[str]:
    ctx.twin_threshold = resolve_threshold(ctx.measure, ctx.twin_threshold)
    floor = floor_score(ctx.measure, ctx.twin_threshold)
    twins, baseline, _ = _measure_all(ctx)
    note = ctx.notes.setdefault("split", {})
    note.update(
        measure=ctx.measure,
        threshold=ctx.twin_threshold,
        twin_share={s: round(_share(t.scores, floor), 4) for s, t in twins.items()},
        baseline_share={s: round(_share(t.scores, floor), 4) for s, t in baseline.items()},
    )
    files = [_twin_rank(ctx, base, twins, baseline, floor), _twin_examples(ctx, base, twins), _twins_table(ctx, base, twins, floor)]
    files += _balance(ctx, base)
    evals = {s: t.rows for s, t in twins.items()}
    files += _units(ctx, base, evals)
    files += _missing(ctx, base, list(twins))
    files += _straddle(ctx, base)
    if ctx.group_by:
        files += _by_group_curves(ctx, base, twins, baseline, floor)
    return files
