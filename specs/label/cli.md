# Label CLI

## 1. General Description
`cvsuite label` ingests one or more annotation-bearing or image-only sources, optionally applies exactly one transform, and then runs one output command. Shared structured-format parsing and writing is delegated to `cvsuite.common.io`; cvsuite keeps the public CLI spellings, source detection policy, multi-source merge behavior, split helpers, and label-specific outputs.

## 2. Inputs and Outputs (I/O)
Canonical usage:
- `cvsuite label <src> [<src> ...] [ingest opts] [<transform> <transform args>] <command> [args]`

Base inputs:
- one or more positional `src` values
- `--from {auto,yolo,labelme,coco,images,class-dir,semseg-mask}`
- `--keypoints-file <path>` for LabelMe ingest only
- `--max-depth <n>` limits recursion for the flat/class-dir readers (`-1` = unlimited)
- `-h/--help` — routed to the specific transform/command parser named on the line
  (e.g. `cvsuite label ./ds to-coco --help` prints the `to-coco` help, not the
  branch usage)

Visible transform commands:
- `filter`
- `ground`
- `ocr`
- `ops`
- `sample`

Visible output commands:
- `to-yolo`
- `to-labelme`
- `to-coco`
- `to-results`
- `audit`
- `inspect`

## 3. Interfaces
Source ingest interface:
- multiple sources are ingested through `router.ingest_many(...)`
- datasets are merged by deep-copying records, normalizing image paths against each source root, and remapping class ids by label text
- cvsuite maps public source spellings to canonical format ids:
  - `images` and detected unstructured trees -> `flat`
  - `yolo` -> `yolo`
  - `labelme` -> `labelme`
  - `coco` -> `coco`
  - `class-dir` -> `class-dir`
  - `semseg-mask` -> `semseg_mask`

Transform discovery interface:
- transform CLI names are synthesized from `label/transform/<proj>/commands/*.py`
- if the module name is `run`, the CLI name is the project name
- otherwise the CLI name is `<proj>-<module>`

Output discovery interface:
- outputs come from `label/output/commands/*.py`

## 4. Business Decisions and Strict Policies
- Only one transform stage is supported before the output command.
- Some outputs preserve legacy branch spellings where that remains user-visible, such as `to-yolo --task seg`.
- YAML semantic-segmentation manifests are detected before generic YOLO fallback when they contain split-level `images` and `masks` mappings.
- Multi-source merging is label-centric rather than numeric-class-centric, so class ids are reallocated by class name across sources.
- The merged dataset task is preserved only when all non-null source dataset tasks agree; otherwise it is downgraded.

## 5. Implementation Details
- The base parser uses `allow_abbrev=False`, so a transform/command flag like
  `--max` is never swallowed as a prefix of a base flag (`--max-depth`).
- `--help` is detected from the raw argv before parsing and dispatched to the
  most specific transform/command parser mentioned before it.

Auto-detection summary:
- YAML file or directory with `data.yaml`
  - `format: coco` => COCO
  - `format: labelme` => LabelMe
  - semantic-segmentation split manifests => `semseg-mask`
  - other YAML => YOLO (split-first or Ultralytics layout; ingest raises if the
    resolved images match zero label files)
- JSON file
  - `images + annotations` => COCO
  - `shapes` => LabelMe
  - generic JSON falls back to COCO
- image file => image-only ingest
- directory
  - prefers LabelMe/COCO JSON sniffing
  - then semantic-segmentation manifest detection
  - then YOLO split-structure heuristics
  - then image-only/unstructured fallback if any image exists

Multi-source merge metadata:
- `meta["multisrc"] = True`
- `meta["sources"] = [<string paths>]`
- `meta["source_formats"] = ...`
- `root` becomes the common path across sources when possible

Related docs:
- `[[label/adapters]]`
- `[[label/output]]`
- `[[common/records_dataset]]`
