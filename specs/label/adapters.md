# Label Adapters

## 1. General Description
The label adapter layer is where cvsuite-owned source detection and public spellings are translated into the shared dataset contract. Canonical parsing and writing live in `cvsuite.common.io`; cvsuite keeps:
- source-format detection heuristics
- public CLI spelling compatibility
- multi-source merge semantics
- helper `data.yaml` emission
- branch-level task alias handling where still user-visible

## 2. Inputs and Outputs (I/O)
Input formats handled today:
- YOLO datasets rooted by `data.yaml`-style configs
- COCO JSON datasets and COCO-via-YAML configs
- LabelMe JSON trees
- class-directory classification datasets
- semantic-segmentation mask manifests
- unstructured/image-only trees used as label-branch image sources

Outputs produced by the active adapter layer:
- normalized shared datasets with `image`, `boxes`, `polys`, `kpts`, `classification`, `semantic_mask`, `classes`, `task`, and branch metadata
- helper YAML schema objects via `DataYaml`
- canonical on-disk formats written later by `cvsuite.common.io.write_dataset(...)`

## 3. Interfaces
Router and schema interfaces:
- `label.core.router.detect_format(...)`
- `label.core.router.ingest(...)`
- `label.core.router.ingest_many(...)`
- `label.core.schema.DataYaml`
- `label.core.schema.read_data_yaml(...)`
- `label.core.schema.write_data_yaml(...)`
- `label.core.schema.resolve_data_yaml(...)`

Canonical delegated formats:
- `yolo`
- `coco`
- `labelme`
- `class-dir`
- `semseg_mask`
- `flat` for image-only/unstructured inputs

## 4. Business Decisions and Strict Policies
- Format detection is heuristic but branch-owned:
  - YAML with `format: coco` => COCO
  - YAML with `format: labelme` => LabelMe
  - YAML with `format: semseg_mask` or `format: semseg-mask` => semantic-segmentation mask ingest
  - YAML with split mappings like `{train: {images: ..., masks: ...}}` => semantic-segmentation mask ingest
  - other YAML => YOLO
  - JSON with `images` and `annotations` => COCO
  - JSON with `shapes` => LabelMe
  - directories with split `images/` plus `labels*` => YOLO
  - image-only paths fall back to image/unstructured ingest
- Multi-source merges are label-centric, not numeric-class-centric. Class ids are remapped by resolved class name or label text.
- Public `semseg-mask` is translated to canonical format id `semseg_mask`.
- Branch-level task aliases still exist where older outputs expose them (e.g. `--task seg`).

## 5. Implementation Details
Active ingestion path:
1. Detect the public source spelling in the label router.
2. Map the spelling to the canonical format id.
3. Call `cvsuite.common.io.read_dataset(...)`.
4. Run `normalize_dataset(...)` to coerce task hints.
5. For multi-source ingest, deep-copy records, absolutize image paths against each source root, and remap class ids by label.

Cross-links:
- `[[label/cli]]`
- `[[label/output]]`
- `[[common/io]]`
- `[[common/records_dataset]]`
