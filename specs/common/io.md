# Shared Dataset I/O

Related: **Read first** [records_dataset](records_dataset.md). **See also** [label adapters](../label/adapters.md), [vlm adapters](../vlm/adapters.md), [classify output](../classify/output.md).

## Overview

`cvsuite.common.io` reads a folder or file into the shared dataset and writes the dataset back out in a named on-disk format. It is a small router plus one adapter per format. New formats are added as new adapters without touching the rest of the system.

Branches keep their own public spellings and source-detection rules ([label adapters](../label/adapters.md), [vlm adapters](../vlm/adapters.md), [classify cli](../classify/cli.md)), then delegate parsing and writing here.

## Language

- **Adapter**: a class that reads and/or writes one format.
- **Format id**: the adapter's registered name, listed below.
- **Auto detection**: choosing a reader by asking each adapter's `matches()`, in priority order.

## Map

- `io/router.py`: registration, `read_dataset`, `write_dataset`, the built-in table.
- `io/<format>.py`: one adapter per format.
- `io/common.py`: shared mechanics and constants.

## Contracts

### Router (`io/router.py`)

```python
read_dataset(src, *, format="auto", task="auto", splits=None, **adapter_options) -> VisionDataset
write_dataset(dataset, dst, *, format, splits=None, **writer_options) -> None
register_reader(name, adapter, *, priority=100) -> None
register_writer(name, adapter) -> None
list_readers() -> list[str]
list_writers() -> list[str]
```

Import the router from `cvsuite.common.io`, never from `cvsuite.common.core`: `io` imports `core`, so re-exporting would create an import cycle.

`read_dataset` stamps `dataset.source_format` and each `record.source_format` with the resolved format id. Built-in adapters register lazily on the first router call from the `_BUILTINS` table.

### Adapter interface

- `matches(src: Path) -> bool`: cheap on-disk shape check used by auto detection.
- `read(path, *, task, splits, **opts) -> VisionDataset`
- `write(dataset, dst, *, splits, **opts) -> None` (writers only)

### Built-in formats

Auto detection picks the lowest priority number whose `matches()` is true.

| id | module | read | write | priority |
|---|---|---|---|---|
| `semseg_mask` | `io/semseg_mask.py` | yes | no | 5 |
| `yolo` | `io/yolo.py` | yes | yes | 10 |
| `coco` | `io/coco.py` | yes | yes | 10 |
| `vqa_style` | `io/vqa_style.py` | yes | yes | 10 |
| `labelme` | `io/labelme.py` | yes | yes | 20 |
| `shards_vlm` | `io/shards_vlm.py` | yes | yes | 30 |
| `flat_vlm_json` | `io/flat_vlm_json.py` | yes | yes | 40 |
| `class-dir` | `io/class_dir.py` | yes | yes | 50 |
| `flat` | `io/flat.py` | yes | no | 60 |

### Format details

- `flat`: loose image or tree into image-only records. Option `max_depth` (`-1` unlimited).
- `class-dir`: class-folder classification, single-label and multi-label duplicate-tree layouts. Options `class_name_mode` (`basename` | `full_path`), `max_depth`. Split-prefixed trees and threshold-based winner selection are handled inside.
- `yolo`: split trees and `data.yaml` for detection, instance segmentation and pose, plus optional `labels_vqa/` sidecars. Reads both the split-first layout (`<root>/<split>/images` + `<root>/<split>/labels*`) and the Ultralytics layout (`<root>/images/<split>` + `<root>/labels*/<split>`). `BBox.kind` (`det`, `seg`, `pose`) carries the box kind. Write option `hardlink`.
- `coco`: COCO JSON or YAML-driven layouts (`images`, `annotations`, `categories`); detection and instance-segmentation geometry normalized; per-image VQA extras. The writer always targets a directory. Write options `hardlink`, `coco_style`, `no_images`, `overwrite`.
- `labelme`: image plus JSON sidecars. Boxes, polygons, keypoints, base64 images, shape flags in `attributes["flags"]`, promoted `prompt` / `model` / `text` fields, full source shapes on `record.labelme_shapes` and `imageData` under `record.round_trip["labelme"]`. Read option `keypoints_file`. Write options `embed_image`, `hardlink`, `split_layout`.
- `semseg_mask`: semantic-segmentation manifests (`data.yaml` with per-split `images` / `masks` dirs) into records carrying `semantic_mask`. Read-only.
- `vqa_style`: normalized VQA manifests (`items` plus image paths) and classic raw VQA JSON keyed by `image_id` / `question_id`.
- `shards_vlm`: `.tar` shards of same-stem image and JSON members. The writer transcodes images to JPEG and reports a cleanup dir in metadata.
- `flat_vlm_json`: image plus same-stem JSON sidecar, recursive. A sidecar's image is resolved in its own directory first, then tree-wide, then by the parallel `labels*/` to `images/` mirror path. This keeps a bucket-partitioned export (`map-answers ... to-json`) re-ingestable.

### Shared constants (`io/common.py`)

`io/common.py` holds shared adapter mechanics (YAML/JSON parsing, lazy image metadata, path resolution, sample keys, `copy_or_link`) and constants, notably `CLASS_DIR_MULTI_LABELS_ATTR` (consumed by [classify output](../classify/output.md)) and `LABELME_META_KEY = "cvsuite"` (LabelMe JSON namespace for round-trip metadata).

## Constraints

- `format="auto"` MUST pick the adapter with the lowest priority number whose `matches()` is true; a tie at the same priority MUST raise.
- An unknown explicit `format` MUST raise `KeyError`. A missing `src` MUST raise `FileNotFoundError`. `write_dataset` with an unregistered or reader-only format MUST raise `KeyError`.
- Reading a YOLO source that resolves images but matches no label files MUST raise; image-only trees go through `flat`.
- In `flat_vlm_json`, only a same-directory `stem.jpg` plus `stem.png` clash MAY raise.
- Adapters MUST read and write only through the shared dataset and MUST NOT receive branch-specific adapters or CLI state.
- Geometry MUST be normalized in memory; adapters convert to and from format layouts.
- `coco` writing MUST target a directory.

## History

- [Shared dataset contract](../logs/2026-09-20_shared-dataset-contract.md): why format handling is decoupled from the rest of the system.
