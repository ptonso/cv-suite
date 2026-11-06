"""Cross-validation folds as a dataset sees them: `meta["folds"]` and `attributes["fold_splits"]`. Contract: `specs/common/split.md#writing-folds`."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterator

from cvsuite.common.core import VisionDataset

FOLD_SPLITS_ATTR = "fold_splits"
FOLDS_META = "folds"


def _fold(dataset: VisionDataset, index: int) -> VisionDataset:
    records = []
    for record in dataset.records:
        attributes = {k: v for k, v in record.attributes.items() if k != FOLD_SPLITS_ATTR}
        records.append(replace(record, split=record.attributes[FOLD_SPLITS_ATTR][index], attributes=attributes))
    return replace(dataset, records=records, meta={k: v for k, v in dataset.meta.items() if k != FOLDS_META})


def fold_targets(dataset: VisionDataset, dst: Path) -> Iterator[tuple[VisionDataset, Path]]:
    """Yield `(dataset, dst)` as it is, or one `(dataset_i, dst / "fold<i>")` per present fold."""
    folds = dataset.meta.get(FOLDS_META)
    if folds is None:
        yield dataset, Path(dst)
        return
    for index in folds["present"]:
        yield _fold(dataset, index), Path(dst) / f"fold{index}"


def require_single_fold(dataset: VisionDataset) -> VisionDataset:
    """For review outputs: the dataset itself, or its only fold; several folds raise."""
    folds = dataset.meta.get(FOLDS_META)
    if folds is None:
        return dataset
    if len(folds["present"]) != 1:
        raise SystemExit(f"This output reviews one dataset but the chain holds {len(folds['present'])} folds. Add `split --only-fold I` to pick one.")
    return _fold(dataset, folds["present"][0])
