"""Canonical shared dataset contract for cvsuite.

Do not re-export the `cvsuite.common.io` router here: `common.io` imports
`common.core`, so a re-export would create an import cycle. Import
`read_dataset` / `write_dataset` from `cvsuite.common.io` directly.
"""

from cvsuite.common.core.types import (
    BBox,
    Classification,
    Embedding,
    FMRequest,
    FormatId,
    ImageInfo,
    ImageRecord,
    KeypointValue,
    Keypoints,
    LabelMeShape,
    Point,
    Polygon,
    SemanticMaskRef,
    SplitId,
    TaskId,
    VQA,
)
from cvsuite.common.core.records import (
    Record,
    VisionDataset,
    VisionRecord,
    copy_dataset,
    copy_record,
    normalize_dataset,
)

__all__ = [
    "BBox",
    "Classification",
    "Embedding",
    "FMRequest",
    "FormatId",
    "ImageInfo",
    "ImageRecord",
    "KeypointValue",
    "Keypoints",
    "LabelMeShape",
    "Point",
    "Polygon",
    "Record",
    "SemanticMaskRef",
    "SplitId",
    "TaskId",
    "VQA",
    "VisionDataset",
    "VisionRecord",
    "copy_dataset",
    "copy_record",
    "normalize_dataset",
]
