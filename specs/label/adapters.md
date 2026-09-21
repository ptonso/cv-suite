# Label Adapters

Related: **Read first** [label cli](cli.md). **See also** [io](../common/io.md), [records_dataset](../common/records_dataset.md), [label output](output.md).

## Overview

The label adapter layer turns a source path into the shared dataset. It owns source-format detection and the public spellings; parsing and writing itself lives in [io](../common/io.md). It also handles multi-source merging, helper `data.yaml` emission, and task aliases that remain user-visible.

Ingest path: detect the public spelling in the label router, map it to a canonical format id, call `cvsuite.common.io.read_dataset(...)`, run `normalize_dataset(...)` to coerce task hints, and for several sources deep-copy records, absolutize image paths against each root and remap class ids by label.

## Language

- **Helper `data.yaml`**: YAML config written or read beside datasets (`DataYaml`).

## Map

- `label/core/router.py`: detection and ingest.
- `label/core/schema.py`: `DataYaml` and YAML helpers.

## Contracts

- `label.core.router.detect_format(...)`, `ingest(...)`, `ingest_many(...)`.
- `label.core.schema.DataYaml`, `read_data_yaml(...)`, `write_data_yaml(...)`, `resolve_data_yaml(...)`.

Inputs: YOLO datasets rooted by `data.yaml`-style configs; COCO JSON and COCO-via-YAML; LabelMe JSON trees; class-folder datasets; semantic-segmentation mask manifests; image-only trees.

Outputs: shared datasets with `image`, `boxes`, `polys`, `kpts`, `classification`, `semantic_mask`, `classes`, `task` and branch metadata; `DataYaml` objects; canonical formats written later by `cvsuite.common.io.write_dataset(...)`.

Canonical formats delegated to: `yolo`, `coco`, `labelme`, `class-dir`, `semseg_mask`, `flat`.

### Format detection

- YAML with `format: coco` is COCO; `format: labelme` is LabelMe; `format: semseg_mask` or `semseg-mask` is semantic-segmentation masks; split mappings like `{train: {images: ..., masks: ...}}` are semantic-segmentation masks; other YAML is YOLO.
- JSON with `images` and `annotations` is COCO; with `shapes` is LabelMe; other JSON falls back to COCO.
- A directory with `data.yaml` applies the YAML rules; otherwise it prefers LabelMe/COCO JSON sniffing, then mask manifests, then YOLO split structure (`images/` plus `labels*`), then image-only ingest if any image exists.
- An image file is image-only ingest.
- Reading YOLO that resolves images but zero label files raises ([io](../common/io.md)).

## Constraints

- Detection MUST follow the order above, with semantic-segmentation manifests before YOLO.
- Multi-source merges MUST remap class ids by resolved class name or label text, not by number.
- Public `semseg-mask` MUST translate to canonical `semseg_mask`.
- Task aliases MAY remain only where outputs still expose them (for example `--task seg`).
