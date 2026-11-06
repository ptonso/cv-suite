"""leak: can the label be guessed without the object. Figures `01` to `03` and tables `guess_accuracy.csv`, `cue_bins.csv`, `cue_contribution.csv`."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cvsuite.common import splitting
from cvsuite.common.io.common import resolve_record_image_path
from cvsuite.common.transforms.split import EMPTY, record_labels, weights
from cvsuite.eda.core import cues as cue_mod
from cvsuite.eda.core.sections import Context, needs_leak, split_order
from cvsuite.eda.core.style import BASELINE, OTHER, PALETTE, apply_style, ensure_matplotlib, save_figure, top_entries

SHUFFLES = 20
GROUP_SHUFFLES = 5
FOLDS = 5
COMBINED = "combined"
MAIN_CUES = 4


def needs(ctx: Context) -> str | None:
    reason = needs_leak(ctx)
    if reason:
        return reason
    splits = {ctx.table.rows[i]["split"] for i in ctx.table.readable()}
    if "train" in splits and len(splits) > 1:
        return None
    groups = {ctx.table.rows[i]["group"] if ctx.group_by else i for i in ctx.table.readable()}
    return None if len(groups) >= 2 else "needs at least two groups to cross-validate"


@dataclass
class Samples:
    """One prediction problem: the rows it draws on (indices into `Data.rows`) and the integer class of each sample."""

    name: str
    rows: np.ndarray
    y: np.ndarray


@dataclass
class Data:
    rows: np.ndarray
    labels: list[list[str]]
    raw: dict[str, list]
    problem: list[Samples]
    mode: str
    classes: list[str]

    def splits(self, ctx: Context) -> np.ndarray:
        return np.array([ctx.table.rows[i]["split"] for i in self.rows])


def _build(ctx: Context) -> Data:
    classes = ctx.dataset.classes
    labels = {i: sorted(set(record_labels(ctx.dataset.records[i], classes)) - {EMPTY}) for i in ctx.table.readable()}
    annotated = ctx.has_geometry()
    rows = np.array([i for i in labels if annotated or labels[i]], dtype=np.int64)
    kept = [labels[i] for i in rows]
    raw = cue_mod.extract([ctx.table.rows[i] for i in rows], [resolve_record_image_path(ctx.dataset, ctx.dataset.records[i]) for i in rows])
    images = Counter(name for names in kept for name in names)
    if annotated:
        named = [n for n, _ in sorted(images.items(), key=lambda kv: (-kv[1], kv[0]))[: ctx.top]]
        problem = [Samples(name, np.arange(len(rows)), np.array([int(name in names) for names in kept])) for name in named]
        return Data(rows, kept, raw, problem, f"one-versus-rest presence of the {len(named)} largest classes", named)
    names = sorted(images)
    index = {n: k for k, n in enumerate(names)}
    pairs = [(r, index[n]) for r, row in enumerate(kept) for n in row]
    problem = [Samples("(label)", np.array([r for r, _ in pairs]), np.array([y for _, y in pairs]))]
    return Data(rows, kept, raw, problem, f"label of the record, {len(names)} classes", names)


def _folds(ctx: Context, data: Data) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped five-fold masks through the splitter: `--group-by` keeps groups whole, each row is a group otherwise."""
    ids = [ctx.table.rows[i]["group"] if ctx.group_by else str(i) for i in data.rows]
    matrix = weights(ctx.dataset, "class")[0][data.rows]
    _, index, counts = splitting.group_counts(ids, matrix)
    k = min(FOLDS, len(counts))
    fold = splitting.pack(counts, [1.0 / k] * k, seed=ctx.seed, restarts=50).assignment[index]
    return [(fold != f, fold == f) for f in range(k)]


def _pairs(ctx: Context, data: Data) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    splits = data.splits(ctx)
    present = [s for s in split_order(set(splits)) if s != "train"]
    scoring = "test" if "test" in present else (present[0] if present else None)
    if scoring is not None and (splits == "train").any() and (splits == scoring).any():
        return f"fit on train, score on {scoring}", [(splits == "train", splits == scoring)]
    return f"grouped {FOLDS}-fold cross-validation", _folds(ctx, data)


def _balanced(ids_fit, y_fit, ids_score, y_score, k: int) -> float:
    """The likelihood rule: each cue value predicts the class with the largest share of its fitting records showing it, unseen values the majority class."""
    keys = ids_fit * k + y_fit
    unique, count = np.unique(keys, return_counts=True)
    code, cls = unique // k, unique % k
    share = count / np.maximum(np.bincount(y_fit, minlength=k)[cls], 1)
    order = np.lexsort((cls, -share, code))
    code_o = code[order]
    first = np.r_[True, code_o[1:] != code_o[:-1]]
    codes, classes = code_o[first], cls[order][first]
    pos = np.minimum(np.searchsorted(codes, ids_score), len(codes) - 1)
    predicted = np.where(codes[pos] == ids_score, classes[pos], np.bincount(y_fit, minlength=k).argmax())
    present = np.bincount(y_score, minlength=k)
    hits = np.bincount(y_score[predicted == y_score], minlength=k)
    seen = present > 0
    return float((hits[seen] / present[seen]).mean())


@dataclass
class Score:
    accuracy: float
    floor: float
    chance: float
    rows: int


@dataclass
class Evaluator:
    data: Data
    pairs: list[tuple[np.ndarray, np.ndarray]]
    seed: int
    cache: dict = field(default_factory=dict)

    def ids(self, pair: int, cues: tuple[str, ...], restrict: np.ndarray | None, key: str) -> np.ndarray:
        slot = (pair, cues, key)
        if slot not in self.cache:
            fit = self.pairs[pair][0] if restrict is None else self.pairs[pair][0] & restrict
            single = [cue_mod.encode(self.data.raw[c], c in cue_mod.NUMERIC, fit)[0] for c in cues]
            out = single[0]
            for other in single[1:]:
                out = cue_mod.joint(out, other)
            self.cache[slot] = out
        return self.cache[slot]

    def score(self, cues: tuple[str, ...], *, shuffles: int = SHUFFLES, restrict: np.ndarray | None = None, key: str = "") -> dict[str, Score] | None:
        """Per target, the mean over pairs of the balanced accuracy, its shuffled floor and its chance; None when no target can be scored."""
        k = max(int(s.y.max()) for s in self.data.problem) + 1
        sums: dict[str, list[tuple[float, float, float, int]]] = {}
        for p, (fit, score) in enumerate(self.pairs):
            if restrict is not None:
                fit, score = fit & restrict, score & restrict
            ids = self.ids(p, cues, restrict, key)
            rng = np.random.default_rng(self.seed)
            for s in self.data.problem:
                fi, si = fit[s.rows], score[s.rows]
                y_fit, y_score = s.y[fi], s.y[si]
                if len(np.unique(y_fit)) < 2 or len(np.unique(y_score)) < 2:
                    continue
                f_ids, s_ids = ids[s.rows][fi], ids[s.rows][si]
                accuracy = _balanced(f_ids, y_fit, s_ids, y_score, k)
                floor = float(np.mean([_balanced(f_ids, rng.permutation(y_fit), s_ids, y_score, k) for _ in range(shuffles)])) if shuffles else float("nan")
                sums.setdefault(s.name, []).append((accuracy, floor, 1.0 / len(np.unique(y_score)), int(si.sum())))
        if not sums:
            return None
        return {n: Score(*[float(np.mean([v[i] for v in vals])) for i in range(3)], sum(v[3] for v in vals)) for n, vals in sums.items()}


def _pooled(scores: dict[str, Score]) -> Score:
    return Score(*[float(np.mean([getattr(s, f) for s in scores.values()])) for f in ("accuracy", "floor", "chance")], sum(s.rows for s in scores.values()) // max(1, len(scores)))


def _group_masks(ctx: Context, data: Data) -> dict[str, np.ndarray]:
    names = np.array([ctx.table.rows[i]["group"] for i in data.rows])
    sizes = Counter(names.tolist())
    ordered = sorted(sizes, key=lambda g: (-sizes[g], g))
    masks = {g: names == g for g in ordered[: ctx.top]}
    if len(ordered) > ctx.top:
        masks[f"other ({len(ordered) - ctx.top})"] = ~np.isin(names, ordered[: ctx.top])
    return masks


def _plt():
    plt = ensure_matplotlib()
    apply_style()
    return plt


def _csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _cue_sets() -> dict[str, tuple[str, ...]]:
    return {**{c: (c,) for c in cue_mod.CUES}, COMBINED: cue_mod.COMBINED}


def _accuracy_table(base, results, groups) -> None:
    rows = []
    for cue, per_target in results.items():
        for target, s in per_target.items():
            rows.append([cue, target, round(s.accuracy, 4), round(s.floor, 4), round(s.chance, 4), s.rows, ""])
        pooled = _pooled(per_target)
        if len(per_target) > 1:
            rows.append([cue, "(pooled)", round(pooled.accuracy, 4), round(pooled.floor, 4), round(pooled.chance, 4), pooled.rows, ""])
    for cue, per_group in groups.items():
        for group, s in per_group.items():
            rows.append([cue, "(pooled)", round(s.accuracy, 4), round(s.floor, 4), round(s.chance, 4), s.rows, group])
    _csv(base / "tables" / "guess_accuracy.csv", ["cue", "target", "accuracy", "floor", "chance", "scoring_rows", "group"], rows)


def _cue_figure(base, data, pooled, groups, plan) -> str:
    plt = _plt()
    names = list(pooled)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 2))
    positions = np.arange(len(names))
    ax.barh(positions, [pooled[n].accuracy for n in names], color=[PALETTE[1] if n == COMBINED else PALETTE[0] for n in names], height=0.6)
    for p, n in zip(positions, names):
        ax.text(pooled[n].accuracy, p, f" {pooled[n].accuracy:.2f}", va="center", fontsize=7)
        ax.plot([pooled[n].floor], [p], marker="D", color="black", ms=4, ls="", label="shuffled-label floor" if p == 0 else None)
    chance = float(np.mean([pooled[n].chance for n in names]))
    ax.axvline(chance, color=BASELINE, ls="--", label=f"chance ({chance:.2f})")
    for k, group in enumerate(dict.fromkeys(g for per in groups.values() for g in per)):
        xs = [groups[n][group].accuracy for n in names if n in groups and group in groups[n]]
        ys = [p for p, n in zip(positions, names) if n in groups and group in groups[n]]
        ax.plot(xs, ys, marker="o", ms=3, ls="", color=OTHER if group.startswith("other (") else PALETTE[2], alpha=0.7, label="within a group" if k == 0 else None)
    ax.set_yticks(positions, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("balanced accuracy")
    ax.set_title(f"Can the label be guessed from the file? {data.mode}; {plan}", fontsize=8)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    save_figure(fig, base / "figures" / "01_metadata_cues.png")
    return "figures/01_metadata_cues.png"


def _bin_tables(base: Path, data: Data, plan_fit: np.ndarray) -> dict[str, tuple[list[str], np.ndarray]]:
    """Write the class counts per bin of every cue, with bins fitted on the first fitting set, and return the bins."""
    rows, per_cue = [], {}
    for cue in cue_mod.CUES:
        ids, labels = cue_mod.encode(data.raw[cue], cue in cue_mod.NUMERIC, plan_fit)
        counts: Counter = Counter()
        for r in np.nonzero(plan_fit)[0]:
            for name in data.labels[r] or [EMPTY]:
                counts[(int(ids[r]), name)] += 1
        for (b, name), n in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            rows.append([cue, labels[b], name, n])
        per_cue[cue] = (labels, ids)
    _csv(base / "tables" / "cue_bins.csv", ["cue", "bin", "class", "count"], rows)
    return per_cue


def _bins_figure(ctx, base, data, plan_fit, per_cue, best) -> str:
    plt = _plt()
    totals = Counter(name for r in np.nonzero(plan_fit)[0] for name in data.labels[r] or [EMPTY])
    shown = [n for n, _ in top_entries({k: float(v) for k, v in totals.items()}, min(ctx.top, len(PALETTE) - 1))[0] if not n.startswith("other (")]
    colors = {n: PALETTE[k] for k, n in enumerate(shown)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    grand = sum(totals.values())
    for ax, cue in zip(axes.ravel(), best):
        labels, ids = per_cue[cue]
        rows_in = np.nonzero(plan_fit)[0]
        per_bin: dict[int, Counter] = {}
        for r in rows_in:
            for name in data.labels[r] or [EMPTY]:
                per_bin.setdefault(int(ids[r]), Counter())[name if name in colors else "other"] += 1
        order = sorted(per_bin, key=lambda b: (-sum(per_bin[b].values()), b))
        kept = sorted(order[: ctx.top])
        rest = order[ctx.top :]
        entries = [(labels[b], per_bin[b]) for b in kept]
        if rest:
            entries.append((f"other ({len(rest)})", sum((per_bin[b] for b in rest), Counter())))
        for r, (label, counts) in enumerate(entries):
            total, start = sum(counts.values()), 0.0
            for name in [*shown, "other"]:
                width = counts.get(name, 0) / total
                ax.barh(r, width, left=start, color=colors.get(name, OTHER), label=name if r == 0 else None)
                start += width
            ax.text(1.0, r, f" n={total}", va="center", fontsize=7)
        edge = 0.0
        for name in shown:
            edge += totals[name] / grand
            ax.axvline(edge, color="black", ls="--", lw=0.6)
        ax.set_yticks(range(len(entries)), [l for l, _ in entries], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_title(f"{cue}: class share per bin", fontsize=9)
    for ax in axes.ravel()[len(best) :]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(shown) + 1)
    fig.suptitle("Which values carry the signal? (dashed: overall class shares; a bin that matches them carries none)", fontsize=9)
    save_figure(fig, base / "figures" / "02_cue_bins.png")
    return "figures/02_cue_bins.png"


def _forward(evaluator: Evaluator) -> list[tuple[int, str, float]]:
    chosen: list[str] = []
    steps = []
    while len(chosen) < len(cue_mod.CUES):
        best = None
        for cue in cue_mod.CUES:
            if cue in chosen:
                continue
            scores = evaluator.score(tuple([*chosen, cue]), shuffles=0)
            value = _pooled(scores).accuracy if scores else float("-inf")
            if best is None or value > best[1]:
                best = (cue, value)
        chosen.append(best[0])
        steps.append((len(chosen), best[0], best[1]))
    return steps


def _contribution_figure(base, pooled, steps) -> str:
    plt = _plt()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    names = sorted(pooled, key=lambda n: (n != COMBINED, -pooled[n].accuracy, n))
    positions = np.arange(len(names))
    left.barh(positions, [pooled[n].accuracy for n in names], color=[PALETTE[1] if n == COMBINED else PALETTE[0] for n in names], height=0.6)
    for p, n in zip(positions, names):
        left.text(pooled[n].accuracy, p, f" {pooled[n].accuracy:.2f}", va="center", fontsize=7)
    chance = float(np.mean([pooled[n].chance for n in names]))
    left.axvline(chance, color=BASELINE, ls="--", label=f"chance ({chance:.2f})")
    left.set_yticks(positions, names)
    left.invert_yaxis()
    left.set_xlim(0, 1.05)
    left.set_xlabel("balanced accuracy")
    left.set_title("Combined cue (file size, brightness, sharpness) against each cue alone", fontsize=8)
    left.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    xs = [s for s, _, _ in steps]
    right.plot(xs, [a for _, _, a in steps], marker="o", color=PALETTE[0])
    right.axhline(chance, color=BASELINE, ls="--", label="chance")
    for s, cue, a in steps:
        right.annotate(f"+{cue}\n{a:.2f}", (s, a), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6)
    right.set_xticks(xs)
    right.set_ylim(0, 1.1)
    right.set_xlabel("cues added, largest gain first")
    right.set_ylabel("balanced accuracy")
    right.set_title("What does each added cue contribute?", fontsize=8)
    right.legend(fontsize=7)
    save_figure(fig, base / "figures" / "03_cue_contribution.png")
    return "figures/03_cue_contribution.png"


def write(ctx: Context, base: Path) -> list[str]:
    data = _build(ctx)
    plan, pairs = _pairs(ctx, data)
    evaluator = Evaluator(data, pairs, ctx.seed)
    sets = _cue_sets()
    results = {name: scores for name, cues in sets.items() if (scores := evaluator.score(cues))}
    if not results:
        for name in ("01_metadata_cues.png", "02_cue_bins.png", "03_cue_contribution.png"):
            ctx.omit(f"figures/{name}", "no target has two outcomes in both its fitting and scoring rows")
        return []
    ctx.notes["leak"] = {"plan": plan, "target": data.mode, "rows": int(len(data.rows)), "cues": list(sets)}
    pooled = {name: _pooled(scores) for name, scores in results.items()}
    groups: dict[str, dict[str, Score]] = {}
    if ctx.group_by:
        for group, mask in _group_masks(ctx, data).items():
            for name, cues in sets.items():
                scores = evaluator.score(cues, shuffles=GROUP_SHUFFLES, restrict=mask, key=group)
                if scores:
                    groups.setdefault(name, {})[group] = _pooled(scores)
        if not groups:
            ctx.omit("figures/01_metadata_cues.png (within-group points)", "no group holds two classes in both its fitting and scoring rows")
    _accuracy_table(base, results, groups)
    files = ["tables/guess_accuracy.csv", _cue_figure(base, data, pooled, groups, plan)]
    best = sorted(cue_mod.CUES, key=lambda c: (-pooled[c].accuracy, c))[:MAIN_CUES]
    first_fit = pairs[0][0]
    files += ["tables/cue_bins.csv", _bins_figure(ctx, base, data, first_fit, _bin_tables(base, data, first_fit), best)]
    steps = _forward(evaluator)
    _csv(base / "tables" / "cue_contribution.csv", ["step", "cue_added", "accuracy"], [[s, c, round(a, 4)] for s, c, a in steps])
    files += ["tables/cue_contribution.csv", _contribution_figure(base, pooled, steps)]
    return files

