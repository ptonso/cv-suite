# Shared Dataset I/O

## 1. General Description

`cvsuite.common.io` owns dataset reading and writing: a small router plus one
adapter per on-disk format. It turns a filesystem source into the canonical
`VisionDataset` / `VisionRecord` contract (`[[common/records_dataset]]`,
`[[dataclasses]]`) and writes that contract back out to a named format.

This layer was previously the external `visionbase` package. It is now folded into
`cvsuite.common.io`; the router API and adapter contracts are unchanged. The
deprecated `report` component was dropped in the merge and is not part of
`cvsuite`.

Branch adapter layers (`[[label/adapters]]`, `[[vlm/adapters]]`, `[[classify/cli]]`)
own public CLI spellings and on-disk source detection, then delegate the actual
parsing/writing here.

## 2. Inputs and Outputs (I/O)

Inputs: a filesystem path (file or directory tree) plus a format id (or `"auto"`),
an optional coarse `task` hint, an optional `splits` filter, and per-adapter
keyword options.

Outputs: an in-memory `VisionDataset` on read; an on-disk format tree on write.
`read_dataset` stamps `dataset.source_format` and each `record.source_format` with
the resolved builtin format id.

## 3. Interfaces

`cvsuite.common.io` (implemented in `router.py`):

- `read_dataset(src, *, format="auto", task="auto", splits=None, **adapter_options) -> VisionDataset`
- `write_dataset(dataset, dst, *, format, splits=None, **writer_options) -> None`
- `register_reader(name, adapter, *, priority=100) -> None`
- `register_writer(name, adapter) -> None`
- `list_readers() -> list[str]`
- `list_writers() -> list[str]`

Import `read_dataset` / `write_dataset` from `cvsuite.common.io`, never from
`cvsuite.common.core` — `common.io` imports `common.core`, so re-exporting the
router from `common.core` would create an import cycle.

Each adapter class exposes:

- `matches(src: Path) -> bool` — cheap on-disk shape check used for `format="auto"`
- `read(path, *, task, splits, **opts) -> VisionDataset`
- `write(dataset, dst, *, splits, **opts) -> None` (writers only)

Builtin adapters are registered lazily on first router call from the `_BUILTINS`
table in `router.py`.

## 4. Business Decisions and Strict Policies

- Builtin format ids and `format="auto"` detection priority (lower wins):

  | id | module | read | write | auto priority |
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

- `format="auto"` picks the lowest-priority-number adapter whose `matches()` is
  true; a tie at the same priority raises. An unknown explicit `format` raises
  `KeyError`; a missing `src` raises `FileNotFoundError`.
- `write_dataset` with an unregistered or reader-only `format` raises `KeyError`.
- Geometry is stored normalized (`BBox` centre/size, `Polygon`/`Keypoints` points in
  `[0, 1]`). Adapters convert to and from format-specific absolute or normalized
  layouts.
- Adapters only read and write through the canonical contract. They never receive
  branch-specific format adapters or CLI state.
- `io/common.py` holds shared adapter mechanics (YAML/JSON parsing, lazy image
  metadata, path resolution, sample keys, `copy_or_link`) and shared constants,
  notably `CLASS_DIR_MULTI_LABELS_ATTR` (class-dir multi-label attribute, consumed
  by `[[classify/output]]`) and `LABELME_META_KEY = "cvsuite"` (LabelMe JSON
  namespace for round-trip metadata).

## 5. Implementation Details

Per-adapter contracts (layouts documented alongside the code):

- `flat` — loose image file or tree into image-only records. Option: `max_depth`
  (`-1` unlimited).
- `class-dir` — class-folder classification, single-label (one class dir per image)
  and multi-label duplicate-tree layouts. Options: `class_name_mode`
  (`basename` | `full_path`), `max_depth`. Split-prefixed trees and threshold-based
  winner selection are handled internally.
- `yolo` — YOLO split trees / `data.yaml` for detection, instance segmentation, and
  pose, plus optional `labels_vqa/` sidecars. Both the split-first
  (`<root>/<split>/images` + `<root>/<split>/labels*`) and Ultralytics
  (`<root>/images/<split>` + `<root>/labels*/<split>`) layouts are read.
  Reading a YOLO source that resolves images but matches zero label files
  raises (image-only trees are ingested via `flat`, not `yolo`). Box `kind`
  (`"det"` / `"seg"` / `"pose"`) is carried on the `BBox.kind` field. Write
  option: `hardlink`.
- `coco` — COCO JSON or YAML-driven COCO layouts (`images` / `annotations` /
  `categories`); detection + instance-segmentation geometry normalized; per-image
  VQA extras. The writer always targets a directory. Write options:
  `hardlink`, `coco_style`, `no_images`, `overwrite`.
- `labelme` — LabelMe image + JSON sidecars. Boxes / polygons / keypoints,
  base64-embedded images, shape flags into `attributes["flags"]`, promoted fields
  (`prompt` / `model` / `text`) onto the annotation's real fields, and full source
  shapes preserved on `record.labelme_shapes` (plus `record.round_trip["labelme"]`
  for `imageData`). Read option: `keypoints_file`. Write options: `embed_image`,
  `hardlink`, `split_layout`.
- `semseg_mask` — semantic-segmentation image/mask manifests (`data.yaml` with
  per-split `images`/`masks` dirs) into records carrying `semantic_mask`
  (`SemanticMaskRef`). Read-only.
- `vqa_style` — normalized VQA manifests (`items` + image paths) and classic raw
  VQA JSON families keyed by `image_id` / `question_id`. Payload centred on `image`
  + `vqas`.
- `shards_vlm` — `.tar` shards of same-stem image + JSON members. Writer transcodes
  images to JPEG and reports a cleanup dir in metadata.
- `flat_vlm_json` — image + same-stem JSON sidecar with VQA-style fields, recursive.
  A sidecar's image is resolved in its own directory first, then tree-wide, then
  by the parallel `labels*/ <-> images/` mirror path; only a genuine
  same-directory `stem.jpg` + `stem.png` clash raises. This keeps a
  bucket-partitioned export (`map-answers ... to-json`) re-ingestable.

Related docs:

- `[[common/records_dataset]]`
- `[[dataclasses]]`
- `[[label/adapters]]`
- `[[vlm/adapters]]`
- `[[architecture]]`
