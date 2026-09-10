# Dataclasses

## 1. General Description
The project’s canonical shared in-memory contract lives in `cvsuite.common.core`: leaf value types and the cvsuite runtime dataclasses in `src/cvsuite/common/core/types.py`, containers in `src/cvsuite/common/core/records.py`. The canonical containers are `VisionDataset`, `VisionRecord`, and `ImageInfo`; the cvsuite names `Record` and `ImageRecord` are aliases for `VisionRecord` and `ImageInfo`.

The contract has two parts:
- portable dataset/container fields (used by `[[common/io]]` adapters)
- cvsuite runtime dataclasses (`FMRequest`, `Embedding`, `LabelMeShape`) and the runtime overlay fields on the containers

`[[common/records_dataset]]` explains why these types exist. This document inventories the concrete field sets.

## 2. Inputs and Outputs (I/O)
Inputs:
- JSON produced by `VisionDataset.to_json()`
- dataclass instances produced by branch ingestors and provider bases
- runtime metadata attached during FM execution

Outputs:
- canonical JSON round-trips through `VisionDataset.to_json()` and `VisionDataset.from_json()`
- in-memory records consumed by branch transforms and output writers

## 3. Interfaces
Shared companion enums/constants:
- `Task` enum (`cvsuite.common.core.enums`)
  - public/legacy spellings include `det`, `seg`, `pose`, `cls`, `multi_cls`
  - `Task.seg.value` is canonical `inst-seg`
  - `Task.from_raw(...)` accepts legacy strings and maps them onto canonical values
- `IMG_EXTS`
  - shared allowed image suffix set used by multiple branches

Canonical value types (`cvsuite.common.core.types`):
- `ImageInfo`
  - fields: `path`, `source`, `width`, `height` (dimensions read lazily from `source`)
- `BBox`
  - required: `cx`, `cy`, `w`, `h`, `cls`
  - optional: `label`, `score`, `id`, `group_id`, `attributes`
  - cvsuite overlay fields: `kind` (default `"det"`), `prompt`, `model`, `text`
- `Polygon`
  - required: `points`, `cls`
  - optional: `label`, `score`, `id`, `group_id`, `attributes`
  - cvsuite overlay fields: `prompt`, `model`
- `Keypoints`
  - required: `points`, `cls`
  - optional: `label`, `score`, `id`, `group_id`, `attributes`
  - cvsuite overlay field: `kind` (default `"pose"`)
- `Classification`
  - fields: `label`, `score`, `probs`, `meta`
- `VQA`
  - fields: `question`, `answer`, `score`, `model`, `meta`
- `SemanticMaskRef`
  - fields: `path`, `encoding`, `ignore_index`, `attributes`
- `VisionRecord`
  - fields:
    - `sample_id`
    - `image`
    - `split`
    - `task`
    - `boxes`
    - `polys`
    - `kpts`
    - `classification`
    - `vqas`
    - `semantic_mask`
    - `rel_image_path`
    - `rel_label_path`
    - `source_format`
    - `round_trip`
    - `attributes`
    - cvsuite overlay: `embeddings` (`list[Embedding]`), `labelme_shapes` (`list[LabelMeShape]`)
- `VisionDataset`
  - fields:
    - `records`
    - `classes`
    - `task`
    - `root`
    - `source_format`
    - `round_trip`
    - `meta`
    - cvsuite overlay: `fm_request` (`FMRequest | None`)

cvsuite runtime dataclasses (`cvsuite.common.core.types`):
- `Embedding`
  - fields: `vector`, `model`, `task`, `meta`
- `FMRequest`
  - fields: `task`, `provider`, `prompt`, `config_path`, `device`, `precision`, `max_gpu_memory`, `batch_size`, `weights_dir`, `model_args`, `meta`
  - property alias: `model` reads and writes `provider`
- `LabelMeShape`
  - fields: `label`, `shape_type`, `points`, `flags`, `group_id`, `description`, `other_data`

Aliases and container helpers (`cvsuite.common.core`):
- aliases
  - `ImageRecord = ImageInfo`
  - `Record = VisionRecord`
- `VisionDataset` methods: `to_json(path)`, `from_json(path)` (classmethod), `by_split()`, `resized(size)`
- free functions: `normalize_dataset(dataset)`, `copy_dataset(dataset, **changes)`, `copy_record(record, **changes)`

## 4. Business Decisions and Strict Policies
- Canonical dataset/container ownership belongs to `cvsuite.common.core`; no branch may define a second competing dataset model.
- Geometry is stored normalized in the shared contract. Writers convert it to format-specific absolute or normalized layouts later.
- `kind`, `prompt`, `model`, and `text` are real fields on `BBox` (and `prompt`/`model` on `Polygon`, `kind` on `Keypoints`), not derived from `attributes`. `attributes` stays for open-ended source data.
- `Keypoints.points` is the storage field for keypoint triples; there is no `kpts` alias on the annotation.
- `FMRequest.provider` is the storage field; `model` is a property alias.
- `resized()` is metadata-only for annotations. It does not reproject boxes, polygons, or keypoints because those are already normalized.
- `record.labelme_shapes` intentionally coexists with normalized boxes/polygons/keypoints so LabelMe round-trips can preserve source shape metadata and later overlay predicted shapes.
- `record.attributes`, `record.round_trip`, and `dataset.meta` are intentionally open-ended and are heavily used for branch-specific provenance.

## 5. Implementation Details
JSON encoding and decoding behavior comes from `src/cvsuite/common/core/utils.py`:
- dataclasses become dicts recursively (including the cvsuite overlay fields `fm_request`, `embeddings`, `labelme_shapes`)
- `ImageInfo` serializes its raw slots (`path`, `source`, `width`, `height`) with no lazy dimension read
- `Path` values serialize as strings
- enums serialize by value
- tuples and lists serialize as arrays
- optional nested dataclasses round-trip through type hints; keys absent from an older payload fall back to field defaults
- `VisionDataset.from_json(...)` runs `normalize_dataset(...)` to coerce task hints onto the `Task` vocabulary

The types relate directly to the rest of the spec:
- `[[common/records_dataset]]` explains the contract boundary
- `[[common/io]]` explains the adapters that read and write these types
- `[[common/fm_runtime]]` and `[[common/fm_provider]]` explain how `FMRequest` is consumed
- `[[label/output]]`, `[[vlm/output]]`, `[[gen/output]]`, and `[[classify/output]]` explain how branch writers interpret these fields
