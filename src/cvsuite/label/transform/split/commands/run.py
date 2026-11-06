"""Assign records to train, val and test, or to K cross-validation folds, keeping groups together."""

from cvsuite.common.transforms.split import attach, run

__all__ = ["attach", "run"]
