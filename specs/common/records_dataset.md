# Shared Dataset Contract

## 1. General Description
The project’s shared in-memory dataset contract is owned by `cvsuite.common.core`. The central types are `VisionDataset`, `VisionRecord`, and `ImageInfo`. `Record` (= `VisionRecord`) and `ImageRecord` (= `ImageInfo`) are aliases, not a separate data model. There is no `RecordsDataset` alias; the dataset type is `VisionDataset` directly.

The purpose of this contract is not to define a new user-facing storage format. The purpose is to let every branch ingest source layouts into a common image-centric record list, run deterministic transforms or FM-backed enrichments there, and then export to a destination layout without forcing branches to share filesystem conventions.

The portable schema and the cvsuite runtime overlay now live together in `cvsuite.common.core` (`types.py` for value types and runtime dataclasses, `records.py` for the containers). Format reading and writing lives in `[[common/io]]`.

## 2. Inputs and Outputs (I/O)
Inputs normalized into the shared contract today include:
- raw images
- class-organized images
- annotated YOLO, LabelMe, and COCO datasets
- VLM flat JSON pairs
- VLM shard archives
- VLM VQA manifests and raw VQA corpora
- intermediate generation prompt expansions
- temporary orientation-classification batches

Outputs derived from the shared contract today include:
- class folders
- YOLO, LabelMe, COCO, audit, preview, and results exports
- VLM JSON, shard, and VQA-style exports
- generated image outputs
- FM subprocess payloads serialized to `in.json` and `out.json`

## 3. Interfaces
Contract (`cvsuite.common.core`):
- `VisionDataset`, `VisionRecord`, `ImageInfo`
- `Record = VisionRecord`, `ImageRecord = ImageInfo`
- `VisionDataset` methods: `to_json(...)`, `from_json(...)`, `by_split(...)`, `resized(...)`
- free functions: `normalize_dataset(...)`, `copy_dataset(...)`, `copy_record(...)`

Format I/O (`cvsuite.common.io`):
- `read_dataset(...)`, `write_dataset(...)` — see `[[common/io]]`

Record-level conventions:
- one `VisionRecord` represents one image-centric sample
- `record.image.path` may be absolute, dataset-root-relative, or temporarily rewritten relative to an extracted temp root
- `record.rel_image_path` preserves source-relative identity when writers need to reconstruct a stable filename or logical path
- `record.rel_label_path` preserves label-side identity when a writer or stats builder needs it
- `record.attributes` stores branch-specific temporary and durable metadata
- `record.round_trip` stores source-format preservation data such as LabelMe shapes

Dataset-level conventions:
- `classes` stores canonical class names in ID order when class IDs are relevant
- `task` stores a coarse dataset task hint using canonical task ids such as `det`, `inst-seg`, `pose`, `cls`, and `multi-cls`
- `root` stores the filesystem anchor used to resolve relative record paths
- `source_format` stores the canonical source format id when known
- `meta` stores dataset-wide provenance and branch state

cvsuite runtime overlay (real fields on the containers):
- `dataset.fm_request` (`FMRequest | None`)
- `record.embeddings` (`list[Embedding]`)
- `record.labelme_shapes` (`list[LabelMeShape]`), populated by the LabelMe reader for round-trip preservation
- annotation fields: `BBox.kind` / `.prompt` / `.model` / `.text`, `Polygon.prompt` / `.model`, `Keypoints.kind`

All of these serialize through `to_json` / `from_json` like any other field; there are no `_cvsuite_*` storage keys.

## 4. Business Decisions and Strict Policies
- The dataset contract is owned once, in `cvsuite.common.core`. No branch may reintroduce a competing dataset container hierarchy.
- No branch is forced to expose the shared contract as a user-visible format.
- The contract is intentionally richer than any single export format:
  - geometry is normalized into boxes, polygons, and keypoints
  - classification and VQA payloads can coexist on the same record
  - round-trip preservation data can be carried beside normalized annotations
- `VisionDataset.resized(...)` changes image dimensions and writes `resize_to` metadata on both records and dataset meta, but intentionally leaves normalized annotation values untouched.
- Providers are only allowed to read and write through this contract; they do not receive direct access to branch-specific format adapters.
- `meta` is intentionally not the canonical home for primary FM inputs. Provider name, prompt, batch size, device, precision, config path, checkpoint overrides, and related runtime instructions belong in `dataset.fm_request`.
- Legacy cvsuite spellings are preserved only at compatibility or CLI boundaries:
  - `Task.seg` maps to canonical task id `inst-seg`
  - `images` / `unstructured` label inputs map to canonical `flat`
  - VLM `json`, `shards`, and `vqa-style` map to canonical `flat_vlm_json`, `shards_vlm`, and `vqa_style`

## 5. Implementation Details
- `fm_request`, `embeddings`, and `labelme_shapes` are plain dataclass fields; readers and transforms set them directly and `to_json` / `from_json` carry them.
- `normalize_dataset(...)` is called by the branch routers and by `from_json`; it coerces `dataset.task` and each `record.task` onto the `Task` vocabulary and otherwise leaves the dataset alone.
- `copy_dataset(...)` / `copy_record(...)` wrap `dataclasses.replace` and re-normalize task hints.

The shared contract is therefore the common denominator for:
- source ingestion
- branch transforms
- FM execution
- output writing
- stats reporting

Related docs:
- `[[dataclasses]]`
- `[[common/io]]`
- `[[common/fm_runtime]]`
- `[[common/fm_provider]]`
- `[[common/fm_utils]]`
