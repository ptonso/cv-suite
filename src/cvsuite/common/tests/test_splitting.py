"""Group packing: `common/splitting.py`."""

import numpy as np
import pytest

from cvsuite.common import splitting as sp


def make(n_groups=60, n_cols=3, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"g{i:03d}" for i in range(n_groups)]
    return ids, rng.integers(1, 6, size=(n_groups, n_cols)).astype(float)


def test_group_counts_sums_records_per_sorted_group() -> None:
    weights = np.array([[1.0, 0], [0, 1], [2, 0], [1, 1]])
    ids, index, counts = sp.group_counts(["b", "a", "b", "c"], weights)
    assert ids == ["a", "b", "c"] and index.tolist() == [1, 0, 1, 2]
    assert counts.tolist() == [[0, 1], [3, 0], [1, 1]]


def test_pack_is_deterministic_and_reaches_the_targets() -> None:
    _, counts = make(200)
    a = sp.pack(counts, [0.6, 0.2, 0.2], seed=3)
    b = sp.pack(counts, [0.6, 0.2, 0.2], seed=3)
    assert (a.assignment == b.assignment).all()
    assert not (sp.pack(counts, [0.6, 0.2, 0.2], seed=4).assignment == a.assignment).all()
    report = sp.make_report(a, sp.SPLITS, ("a", "b", "c"))
    assert report.max_deviation()[0] < 0.03
    assert sum(report.groups) == 200


def test_assignment_does_not_depend_on_record_order() -> None:
    ids = [f"g{i % 40}" for i in range(120)]
    weights = np.random.default_rng(1).integers(0, 3, size=(120, 2)).astype(float)

    def split_of(order):
        gids, index, counts = sp.group_counts([ids[i] for i in order], weights[order])
        result = sp.pack(counts, [0.7, 0.3], seed=9).assignment
        return dict(zip(gids, result.tolist()))

    base = split_of(np.arange(120))
    assert split_of(np.random.default_rng(5).permutation(120)) == base


def test_known_groups_hand_made_case() -> None:
    counts = np.array([[10.0], [10.0], [10.0], [10.0]])
    result = sp.pack(counts, [0.5, 0.5], seed=0)
    assert sorted(np.bincount(result.assignment).tolist()) == [2, 2]


def test_pins_stay_and_count_toward_their_split() -> None:
    _, counts = make(100)
    result = sp.pack(counts, [0.8, 0.2, 0.0], pinned={0: 2, 1: 2, 2: 2}, seed=0)
    assert result.assignment[:3].tolist() == [2, 2, 2]
    assert (result.assignment[3:] != 2).all()
    assert np.allclose(result.assigned[2], counts[:3].sum(axis=0))


def test_tolerance_failure_prints_the_table() -> None:
    counts = np.array([[100.0, 0], [0, 100.0]])
    packing = sp.pack(counts, [0.5, 0.5], seed=0)
    report = sp.make_report(sp.Packing(np.array([0, 0]), packing.targets, np.array([[100.0, 100.0], [0, 0]]), packing.totals, 1, 1), ("train", "val"), ("x", "y"))
    with pytest.raises(SystemExit, match="tolerance"):
        sp.check_tolerance([report], 0.05)
    sp.check_tolerance([report], 0.6)


def test_small_columns_are_listed_as_unchecked() -> None:
    counts = np.array([[1.0, 50.0], [0.0, 50.0], [0.0, 50.0], [0.0, 50.0]])
    report = sp.make_report(sp.pack(counts, [0.5, 0.5], seed=0), ("train", "val"), ("rare", "common"))
    assert report.checked.tolist() == [False, True]
    assert "unchecked" in sp.format_report(report)


def test_restarts_are_bounded_and_reported(monkeypatch) -> None:
    monkeypatch.setattr(sp, "WORK_BUDGET", 1000.0)
    _, counts = make(20, 2)
    packing = sp.pack(counts, [0.5, 0.5], restarts=200)
    assert 1 <= packing.restarts < 200
    assert "restarts reduced" in sp.format_report(sp.make_report(packing, ("train", "val"), ("a", "b")))


def test_folds_partition_and_are_stable() -> None:
    _, counts = make(100)
    matrix, packings = sp.pack_folds(counts, 4, 0.25, seed=2)
    test_of = (matrix == 2)
    assert (test_of.sum(axis=0) == 1).all()
    assert set(np.unique(matrix)) <= {0, 1, 2}
    assert (matrix == sp.pack_folds(counts, 4, 0.25, seed=2)[0]).all()
    assert all(((m == 1).sum() > 0) for m in matrix)
    assert len(packings) == 4
    no_val, _ = sp.pack_folds(counts, 3, 0.0, seed=2)
    assert set(np.unique(no_val)) == {0, 2}
