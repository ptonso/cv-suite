"""Group packing for the `split` transform, free of I/O. Contract: `specs/common/split.md`.

Groups are indivisible lumps with fixed counts per column. Randomized greedy bin-packing places whole
groups into splits so each split's share of every column approaches its target share. Pinned groups are
placed first and count toward their split; the targets of the free groups are `share * pool total`
plus what the split already holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPLITS = ("train", "val", "test")
WORK_BUDGET = 2e8
NOISE = (0.7, 1.3)
TOP_COLUMNS = 20


@dataclass(frozen=True)
class Packing:
    assignment: np.ndarray
    targets: np.ndarray
    assigned: np.ndarray
    totals: np.ndarray
    restarts: int
    requested: int


@dataclass(frozen=True)
class Report:
    label: str
    names: tuple[str, ...]
    columns: tuple[str, ...]
    target_share: np.ndarray
    achieved_share: np.ndarray
    totals: np.ndarray
    groups: tuple[int, ...]
    restarts: int
    requested: int
    checked: np.ndarray

    def max_deviation(self) -> tuple[float, str, str]:
        dev = np.abs(self.achieved_share - self.target_share) * self.checked[None, :]
        s, c = np.unravel_index(int(np.argmax(dev)), dev.shape)
        return float(dev[s, c]), self.columns[c], self.names[s]

    def info(self) -> dict:
        deviation, column, split = self.max_deviation()
        return {
            "label": self.label,
            "splits": list(self.names),
            "groups": dict(zip(self.names, self.groups)),
            "restarts": self.restarts,
            "restarts_requested": self.requested,
            "max_deviation": round(deviation, 4),
            "max_deviation_at": {"column": column, "split": split},
            "unchecked": [c for c, ok in zip(self.columns, self.checked) if not ok],
        }


def group_counts(group_ids: list[str], weights: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return the sorted unique group ids, the group index of each record and the (groups, columns) counts."""
    unique = sorted(set(group_ids))
    lookup = {gid: i for i, gid in enumerate(unique)}
    index = np.array([lookup[g] for g in group_ids], dtype=np.int64)
    counts = np.zeros((len(unique), weights.shape[1]), dtype=np.float64)
    np.add.at(counts, index, weights)
    return unique, index, counts


def pack(
    counts: np.ndarray,
    shares: list[float],
    *,
    pinned: dict[int, int] | None = None,
    seed: int = 0,
    restarts: int = 200,
) -> Packing:
    pinned = pinned or {}
    n_groups, n_cols = counts.shape
    n_splits = len(shares)
    share = np.asarray(shares, dtype=np.float64)
    totals = counts.sum(axis=0)
    base = np.zeros((n_splits, n_cols))
    for group, split in pinned.items():
        base[split] += counts[group]
    free = np.array([g for g in range(n_groups) if g not in pinned], dtype=np.int64)
    pool = counts[free].sum(axis=0) if len(free) else np.zeros(n_cols)
    targets = share[:, None] * pool[None, :] + base
    norm = np.maximum(totals, 1.0) ** 2
    blocked = np.where(share > 0, 0.0, np.inf)
    used = max(1, min(restarts, int(WORK_BUDGET // max(1, len(free) * n_splits * n_cols))))

    best_score, best = np.inf, None
    rng = np.random.default_rng(seed)
    sizes = counts[free].sum(axis=1)
    for _ in range(used):
        order = free[np.argsort(-(sizes * rng.uniform(*NOISE, size=len(free))), kind="stable")]
        assigned = base.copy()
        choice = np.full(n_groups, -1, dtype=np.int64)
        for group, split in pinned.items():
            choice[group] = split
        for group in order:
            n = counts[group]
            split = int(np.argmin((((2 * (assigned - targets) + n) * n) / norm).sum(axis=1) + blocked))
            assigned[split] += n
            choice[group] = split
        score = float((((assigned - targets) / np.maximum(targets, 1.0)) ** 2).sum())
        if score < best_score:
            best_score, best = score, (choice, assigned)
    assert best is not None
    return Packing(best[0], targets, best[1], totals, used, restarts)


def pack_folds(counts: np.ndarray, k: int, val_share: float, *, seed: int = 0, restarts: int = 200) -> tuple[np.ndarray, list[Packing]]:
    """Return the (k, groups) split index matrix over `SPLITS` and one packing per fold."""
    tests = pack(counts, [1.0 / k] * k, seed=seed, restarts=restarts)
    matrix = np.zeros((k, len(counts)), dtype=np.int64)
    packings: list[Packing] = []
    for i in range(k):
        rest = np.nonzero(tests.assignment != i)[0]
        inner = pack(counts[rest], [1.0 - val_share, val_share], seed=seed + i + 1, restarts=restarts)
        matrix[i, rest] = inner.assignment
        matrix[i, tests.assignment == i] = 2
        targets = np.vstack([inner.targets, tests.targets[i : i + 1]])
        assigned = np.vstack([inner.assigned, tests.assigned[i : i + 1]])
        packings.append(Packing(matrix[i], targets, assigned, tests.totals, min(tests.restarts, inner.restarts), restarts))
    return matrix, packings


def make_report(packing: Packing, names: tuple[str, ...], columns: tuple[str, ...], *, label: str = "") -> Report:
    """Report the shares of every split that holds a target or a group; unused splits are left out."""
    totals = np.maximum(packing.totals, 1.0)
    groups = [int((packing.assignment == s).sum()) for s in range(len(names))]
    used = [s for s in range(len(names)) if groups[s] or packing.targets[s].any()]
    return Report(
        label=label,
        names=tuple(names[s] for s in used),
        columns=columns,
        target_share=packing.targets[used] / totals,
        achieved_share=packing.assigned[used] / totals,
        totals=packing.totals,
        groups=tuple(groups[s] for s in used),
        restarts=packing.restarts,
        requested=packing.requested,
        checked=packing.totals >= 10 * len(used),
    )


def format_report(report: Report) -> str:
    order = np.argsort(-report.totals, kind="stable")[:TOP_COLUMNS]
    head = f"{report.label + ': ' if report.label else ''}target / achieved share per split"
    width = max([len(report.columns[c]) for c in order] + [len("(all)"), len("column")])
    cell = 15

    def row(name: str, total: str, cells: list[str], note: str = "") -> str:
        return f"{name:<{width}}  {total:>8}  " + "  ".join(f"{c:>{cell}}" for c in cells) + note

    lines = [head, row("column", "total", [f"{n} (tgt/got)" for n in report.names])]
    for c in order:
        cells = [f"{report.target_share[s, c]:.1%}/{report.achieved_share[s, c]:.1%}" for s in range(len(report.names))]
        lines.append(row(report.columns[c], f"{report.totals[c]:.0f}", cells, "" if report.checked[c] else "  (unchecked)"))
    grand = max(report.totals.sum(), 1.0)
    cells = [
        f"{(report.target_share[s] * report.totals).sum() / grand:.1%}/{(report.achieved_share[s] * report.totals).sum() / grand:.1%}"
        for s in range(len(report.names))
    ]
    lines.append(row("(all)", f"{report.totals.sum():.0f}", cells))
    lines.append(row("groups", "", [str(g) for g in report.groups]))
    if len(report.columns) > len(order):
        lines.append(f"{len(report.columns) - len(order)} smaller column(s) not shown")
    unchecked = [report.columns[c] for c in range(len(report.columns)) if not report.checked[c]]
    if unchecked:
        shown = ", ".join(unchecked[:TOP_COLUMNS]) + (" ..." if len(unchecked) > TOP_COLUMNS else "")
        lines.append(f"unchecked (total below {10 * len(report.names)}): {shown}")
    if report.restarts < report.requested:
        lines.append(f"restarts reduced from {report.requested} to {report.restarts} to bound the work")
    return "\n".join(lines)


def check_tolerance(reports: list[Report], tolerance: float) -> None:
    for report in reports:
        deviation, column, split = report.max_deviation()
        if deviation > tolerance:
            tables = "\n\n".join(format_report(r) for r in reports)
            raise SystemExit(
                f"{tables}\n\nSplit '{split}' misses its target on `{column}` by {deviation:.1%}"
                f"{' in ' + report.label if report.label else ''}, above --tolerance {tolerance:.1%}. "
                "Raise --tolerance, use finer --group-by keys, or relax --stratify-by."
            )
