# Label Transforms

## 1. General Description
The label transform layer mixes deterministic geometry/file transforms with FM-backed enrichment. The branch currently exposes sampling, grounding, OCR, grouped ops, and filtering commands publicly. These transforms run over normalized label datasets produced by `[[label/adapters]]` and preserve or extend annotations before output writers run.

## 2. Inputs and Outputs (I/O)
Public transform families:
- `sample`
- `ground`
- `ocr`
- `ops`
- `filter`

Concrete public command shapes:
- `sample`
  - `--count` or `--frac`
  - `--hardlink`
  - `--seed`
- `ground`
  - `--provider {sam3,gsam,gdino,llmdet,locate_anything,rex_omni,yolo_e}`
  - `--model-id` (`llmdet` and `locate_anything` only)
  - `--prompt` (mutually exclusive with `--reference-folder`)
  - `--reference-folder` (`yolo_e` only; ingested via adapters into a one-image reference dataset)
  - `--device`
  - `--batch`
  - `--precision`
  - `--config`
  - `--no-resume`
  - `--threshold`
  - `--iou-threshold`
- `ocr`
  - `--provider/--model`
  - `--device`
  - `--batch`
  - `--precision`
  - `--config`
  - `--no-resume`
- `ops`
  - public surface is project-level, but routes to internal subcommands such as `rebox`, `make-negatives`, `crop-dets`, and `inverse-crop-dets`
- `filter`
  - public surface routes to `train-logistic` and `apply`

Outputs:
- mutated dataset
- temporary auxiliary directories for some ops transforms
- FM metadata for grounding and OCR runs

## 3. Interfaces
`sample` interface:
- global sampling over records, preserving annotations and existing splits
- writes `dataset.meta["sample"]`

`ground` interface:
- runs in one of two mutually exclusive modes: text-prompt or reference-image
- text-prompt mode accepts either one raw string prompt or a YAML/JSON prompt spec
  - raw strings are treated as one prompt item and are not split on punctuation
  - YAML/JSON may be a flat list of prompts or a label-to-prompts mapping
  - every prompt is forwarded separately per image and emitted back as the exact label/prompt string
- reference-image mode (`yolo_e` only) ingests `--reference-folder` via `[[label/adapters]]` into a
  reference dataset whose labels are generalized onto the target dataset; the reference's class
  names are emitted as the output labels
- calls shared FM runtime with task family `ground`
- post-filters new annotations only

`ocr` interface:
- calls shared FM runtime with task family `ocr`
- appends OCR boxes to records

`ops` interfaces:
- `rebox`
  - recompute one detection box from visible keypoints
- `make-negatives`
  - generate random square negative crops under a temp root
- `crop-dets`
  - crop detections or polygons into a new temp dataset
- `inverse-crop-dets`
  - takes a `modified_crops` dataset (a LabelMe crop dataset produced from `crop-dets` output) and pastes its annotations back onto the original source images, reversing `crop-dets`

## 4. Business Decisions and Strict Policies
Sampling:
- global sample only; no class-balancing logic here
- preserves annotation payloads instead of creating image-only outputs
- `--count` must be non-negative and cannot exceed total record count
- `--frac` must be between 0 and 1, and a positive fraction guarantees at least one sample when the dataset is non-empty
- this is the correct sampler for YOLO, COCO, and LabelMe inputs

Grounding:
- duplicate prompts assigned to different labels are fatal
- exactly one of `--prompt` or `--reference-folder` must be supplied; both-set and neither-set are fatal
- `--reference-folder` is only valid for `yolo_e`; other providers reject it
- `--model-id` is only valid for `llmdet` and `locate_anything`; unsupported checkpoint ids are fatal
- `yolo_e` accepts text prompts XOR a reference image, never both, and re-validates this in the provider
- `yolo_e` reference mode requires the reference folder to resolve to exactly one annotated image; more than one is fatal
- `sam3` rejects CPU execution
- `rex_omni` box scores are coordinate-token likelihoods (`exp(mean log p)` over the generated
  coordinate tokens), so `--threshold` filters them; degenerate outputs that cannot be aligned to
  the generated tokens fall back to score 1.0 with a warning
- post-filtering compares before/after record annotation counts so only newly added annotations are eligible for thresholding or NMS
- score thresholding is group-based through `group_id`, so linked box/polygon outputs rise or fall together
- IoU suppression is also group-based and happens per output label after prompt-to-label remapping
- overlapping groups from distinct labels are preserved
- prompt labels extend the dataset class list as needed
- temporary grounding prompt attributes are removed in `finally`

OCR:
- provider input is normalized so `paddleOCR` aliases to `paddleocr`
- OCR boxes are appended without removing existing annotations
- OCR detections use `BBox.kind == "ocr"` and recognized text is stored in `BBox.text`

Ops:
- `rebox`
  - `expand` and `adaptive` are currently implemented identically
  - `edge_tol` is currently unused
  - replaces record boxes with a single box using the first keypoint class
- `make-negatives`
  - `min-gap` and `iou-thresh` are declared but currently reserved and unenforced
  - writes new image files under a temp negatives root
  - appends empty-annotation records with `negative=True`
- `crop-dets`
  - mutually exclusive `imgsz` vs `long_size`
  - supports padding modes `background`, `black`, `gray`, `white`
  - keeps only the triggering object in each crop
  - preserves a matching segmentation polygon when one can be associated with that object
  - rewrites crop-local box/polygon geometry for the final crop canvas
  - assigns classification label from the source label on output records

## 5. Implementation Details
Grounding implementation path:
1. Parse prompt spec into flat prompts plus prompt-to-label mapping.
2. Attach temporary per-record prompt attributes.
3. Call FM runtime/provider.
4. Compare old vs new annotation sets.
5. Filter low-score or overlapping new groups.
6. Remap prompt labels, update class IDs, normalize dataset task.

OCR implementation path:
- build `FMRequest(task="ocr", provider="paddleocr", ...)`
- provider appends OCR boxes with `kind="ocr"` and `text`

Ops temp datasets:
- `make-negatives` temp root prefix: `vislabel_neg_*`
- `crop-dets` temp root prefix: `vislabel_crop_*`
- `inverse-crop-dets` temp root prefix: `vislabel_inverse_crop_*`
- `crop-dets` outputs can be sent to normal writers such as `to-yolo`, `to-labelme`, or `to-coco`
- `inverse-crop-dets` ingests its `modified_crops` argument through the LabelMe adapter and re-projects crop annotations onto the source images using the `crop_id` recorded by `crop-dets`

Related docs:
- `[[label/cli]]`
- `[[label/adapters]]`
- `[[label/output]]`
- `[[label/filter]]`
- `[[common/fm_runtime]]`
