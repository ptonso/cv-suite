"""Built-in dataset I/O for cvsuite.

Each module in this package implements one public format id and the read/write
behaviour documented in `specs/common/io/*.md`. The router in `router.py` dispatches
a filesystem source into canonical `VisionDataset` / `VisionRecord` objects.

Built-in formats: `semseg_mask`, `yolo`, `labelme`, `coco`, `class-dir`, `flat`,
`flat_vlm_json`, `vqa_style`, `shards_vlm`.
"""

from cvsuite.common.io.router import (
    list_readers,
    list_writers,
    read_dataset,
    register_reader,
    register_writer,
    write_dataset,
)

__all__ = [
    "list_readers",
    "list_writers",
    "read_dataset",
    "register_reader",
    "register_writer",
    "write_dataset",
]
