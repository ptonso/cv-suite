# Shared Dataset Contract

Related: **Read first** [architecture](../architecture.md). **See also** [io](io.md), [fm_runtime](fm_runtime.md), [fm_provider](fm_provider.md).

## Overview

The shared dataset is the one in-memory model of images and their annotations. It lives in `cvsuite.common.core`. Every branch except `prep` reads its source into it, transformations and models work on it, and outputs write from it.

It is not a user-facing storage format. Its purpose is to let branches share transformations and models without sharing folder conventions. It is deliberately richer than any single export format: geometry is normalized, classification and question-answer payloads can sit on the same record, and data needed to round-trip a source format is carried beside the normalized annotations.

The main types are `VisionDataset` (a list of records plus dataset-wide information), `VisionRecord` (one image-centric sample) and `ImageInfo` (an image's path and size). `Record` and `ImageRecord` are aliases for `VisionRecord` and `ImageInfo`, not a second model.

A worked example: reading a YOLO folder gives a `VisionDataset` whose records each hold an `ImageInfo` and a list of `BBox`es with normalized coordinates. A model transformation attaches an `FMRequest` to the dataset, the model runtime serializes the dataset to JSON for the model process, the process adds boxes, and the runtime reloads the result.

The reason for having a shared model is in [the shared dataset note](../logs/2026-09-20_shared-dataset-contract.md).

## Language

- **Record**: one image-centric sample.
- **Split**: logical partition such as `train`, `val`, `test`. Some VLM paths keep richer strings such as `train2014`.
- **Task**: coarse dataset kind, the `Task` enum with canonical ids `det`, `inst-seg`, `pose`, `cls`, `multi-cls`.
- **Round trip data**: source-format details kept so a writer can reproduce the source (for example LabelMe shapes).
- **Overlay**: fields cvsuite adds on top of the portable schema (`fm_request`, `embeddings`, `labelme_shapes`, and some annotation fields).

## Map

- `common/core/types.py`: value types and runtime dataclasses.
- `common/core/records.py`: `VisionDataset`, `VisionRecord`, containers and helpers.
- `common/core/enums.py`: `Task`, `IMG_EXTS`.
- `common/core/utils.py`: JSON encoding and decoding.
- Format reading and writing: [io](io.md).

## Contracts

### Containers

`VisionDataset` fields: `records`, `classes`, `task`, `root`, `source_format`, `round_trip`, `meta`, and overlay `fm_request` (`FMRequest | None`).
Methods: `to_json(path)`, `from_json(path)` (classmethod), `by_split()`, `resized(size)`.

`VisionRecord` fields: `sample_id`, `image`, `split`, `task`, `boxes`, `polys`, `kpts`, `classification`, `vqas`, `semantic_mask`, `rel_image_path`, `rel_label_path`, `source_format`, `round_trip`, `attributes`, and overlay `embeddings` (`list[Embedding]`), `labelme_shapes` (`list[LabelMeShape]`).

Free functions in `cvsuite.common.core`: `normalize_dataset(dataset)`, `copy_dataset(dataset, **changes)`, `copy_record(record, **changes)`.

### Value types (`common/core/types.py`)

- `ImageInfo`: `path`, `source`, `width`, `height`. Dimensions are read lazily from `source`.
- `BBox`: required `cx`, `cy`, `w`, `h`, `cls`; optional `label`, `score`, `id`, `group_id`, `attributes`; overlay `kind` (default `"det"`), `prompt`, `model`, `text`.
- `Polygon`: required `points`, `cls`; optional `label`, `score`, `id`, `group_id`, `attributes`; overlay `prompt`, `model`.
- `Keypoints`: required `points`, `cls`; optional `label`, `score`, `id`, `group_id`, `attributes`; overlay `kind` (default `"pose"`).
- `Classification`: `label`, `score`, `probs`, `meta`.
- `VQA`: `question`, `answer`, `score`, `model`, `meta`.
- `SemanticMaskRef`: `path`, `encoding`, `ignore_index`, `attributes`.

### Runtime dataclasses (`common/core/types.py`)

- `Embedding`: `vector`, `model`, `task`, `meta`.
- `FMRequest`: `task`, `provider`, `prompt`, `config_path`, `device`, `precision`, `max_gpu_memory`, `batch_size`, `weights_dir`, `model_args`, `meta`. `model` is a property alias of `provider`.
- `LabelMeShape`: `label`, `shape_type`, `points`, `flags`, `group_id`, `description`, `other_data`.

### Task enum

`Task` has values including `det`, `seg`, `pose`, `cls`, `multi_cls`. `Task.seg.value` is the canonical `inst-seg`. `Task.from_raw(...)` accepts the older spellings and returns canonical values.

### Record and dataset conventions

- `record.image.path` may be absolute, root-relative, or rewritten relative to an extracted temporary root.
- `record.rel_image_path` and `record.rel_label_path` keep the source-relative identity that writers and stats need.
- `record.attributes`, `record.round_trip` and `dataset.meta` are open-ended and carry branch provenance. Each branch spec lists the keys it owns.
- `dataset.classes` holds class names in id order. `dataset.task` is the coarse task hint. `dataset.root` anchors relative paths. `dataset.source_format` is the canonical format id when known.

### JSON

`to_json` / `from_json` (`common/core/utils.py`) serialize every field, including overlay fields, as recursive dicts. `ImageInfo` writes its raw slots without reading dimensions. `Path` values become strings, enums their values, tuples arrays. Missing keys take field defaults. `from_json` runs `normalize_dataset`.

### Model request

`dataset.fm_request` carries the primary model inputs: provider, prompt, config path, batch size, device, precision, checkpoint overrides and related runtime instructions. See [fm_runtime](fm_runtime.md).

## Constraints

- The dataset model MUST be owned only by `cvsuite.common.core`. No branch MAY introduce a competing container hierarchy or a `RecordsDataset` alias.
- Geometry MUST be stored normalized: `BBox` as centre and size, `Polygon` and `Keypoints` points in `[0, 1]`. Writers convert to format-specific layouts.
- `BBox.kind`, `.prompt`, `.model`, `.text`, `Polygon.prompt`/`.model` and `Keypoints.kind` MUST be real fields, not derived from `attributes`. `attributes` is for open-ended source data.
- `Keypoints.points` MUST be the storage field for keypoints; there is no `kpts` alias on the annotation.
- Primary model inputs MUST go in `dataset.fm_request`, not in `dataset.meta`.
- `FMRequest.provider` MUST be the storage field, and `model` only an alias.
- `VisionDataset.resized(...)` MUST change image dimensions and write `resize_to` metadata on records and dataset meta, and MUST NOT change normalized annotation values.
- `record.labelme_shapes` MUST coexist with normalized annotations so LabelMe round-trips keep source shapes.
- Models MUST read and write only through this contract. They MUST NOT receive format adapters.
- No branch is forced to expose the dataset as a user-visible format.
- Older spellings are accepted only at compatibility or CLI boundaries: `Task.seg` maps to `inst-seg`; `images` / `unstructured` label inputs map to `flat`; VLM `json`, `shards`, `vqa-style` map to `flat_vlm_json`, `shards_vlm`, `vqa_style`.
- Overlay fields MUST NOT be stored under `_cvsuite_*` keys.
- `normalize_dataset` and `copy_*` MUST re-normalize task hints (`common/tests/test_contracts.py`).

## History

- [Shared dataset contract](../logs/2026-09-20_shared-dataset-contract.md): why one in-memory model, and why the model request has its own field.
