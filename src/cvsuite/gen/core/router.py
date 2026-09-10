from __future__ import annotations

from pathlib import Path

from cvsuite.common.core import VisionDataset
from cvsuite.label.core import router as label_router

Fmt = label_router.Fmt


def detect_format(src: Path, from_hint: str = "auto") -> Fmt:
    return label_router.detect_format(src, from_hint=from_hint)


def ingest_edit_source(src: Path, from_hint: str = "auto") -> VisionDataset:
    return label_router.ingest(src, from_hint=from_hint)
