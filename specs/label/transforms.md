# Label Transforms

Related: **Read first** [label cli](cli.md). **See also** [label filter](filter.md), [label adapters](adapters.md), [label output](output.md), [fm_runtime](../common/fm_runtime.md).

## Overview

Label transforms change an annotated dataset before an output writes it. Some are deterministic (sampling, geometry operations, filtering), and some use foundation models to add annotations (grounding, OCR). Filtering is large enough for its own spec: [label filter](filter.md).

## Language

- **Grounding**: finding boxes or masks in an image that match text prompts (or a reference image).
- **Group id**: links a box and its polygon so they are scored and suppressed together.
- **Prompt label**: the label a prompt is remapped to on output.

## Map

- `label/transform/sample/`, `ground/`, `ocr/`, `ops/`, `filter/`, each with `commands/`.
- `label/core/crops.py`, `annotate/`: crop and annotation helpers.

## Contracts

### sample

`--count` or `--frac`, `--hardlink`, `--seed`. Global sampling over records that keeps annotations and existing splits. Writes `dataset.meta["sample"]`.

### ground

Arguments: `--provider {sam3,gsam,gdino,llmdet,locate_anything,rex_omni,yolo_e}`; `--model-id` (`llmdet`, `locate_anything` only); `--prompt` (exclusive with `--reference-folder`); `--reference-folder` (`yolo_e` only, ingested via [adapters](adapters.md) into a one-image reference dataset); `--device`, `--batch`, `--precision`, `--config`, `--no-resume`, `--threshold`, `--iou-threshold`.

Two exclusive modes:

- Text-prompt: one raw string (one prompt item, never split on punctuation) or a YAML/JSON spec (a flat list, or a label-to-prompts mapping). Each prompt is sent per image and emitted back as the exact label or prompt string.
- Reference-image (`yolo_e` only): the reference dataset's class names become the output labels.

Runs the runtime with task family `ground`; only newly added annotations are post-filtered.

Flow: parse the prompt spec into flat prompts plus a prompt-to-label mapping; attach the temporary record attribute `ground_prompts`; call the runtime; compare old and new annotation sets; filter low-score or overlapping new groups; remap prompt labels, update class ids, normalize the dataset task.

### ocr

`--provider/--model`, `--device`, `--batch`, `--precision`, `--config`, `--no-resume`. Builds `FMRequest(task="ocr", provider="paddleocr", ...)` and appends OCR boxes with `BBox.kind == "ocr"` and the text in `BBox.text`. Existing annotations stay.

### ops

Project-level surface routing to `rebox`, `make-negatives`, `crop-dets` and `inverse-crop-dets`.

- `rebox`: recompute one detection box from visible keypoints, replacing record boxes with a single box using the first keypoint class. `expand` and `adaptive` modes behave identically, and `edge_tol` is unused.
- `make-negatives`: random square negative crops under a temporary root (`vislabel_neg_*`), appended as empty-annotation records with `negative=True`. `min-gap` and `iou-thresh` are declared but not enforced.
- `crop-dets`: crops detections or polygons into a new temporary dataset (`vislabel_crop_*`). `imgsz` and `long_size` are exclusive; padding modes `background`, `black`, `gray`, `white`. Each crop keeps only its triggering object, with a matching polygon when one is associated; geometry is rewritten to the crop canvas; the classification label comes from the source label. Output can go to normal writers.
- `inverse-crop-dets`: takes a `modified_crops` dataset (a LabelMe crop dataset produced from `crop-dets`), ingests it through the LabelMe adapter, and pastes its annotations back onto the original images using the recorded `crop_id` (temporary root `vislabel_inverse_crop_*`).

### filter

Routes to `train-logistic` and `apply`; see [label filter](filter.md).

## Constraints

- `sample` MUST be a global sampler with no class balancing, MUST keep annotation payloads, and is the right sampler for YOLO, COCO and LabelMe inputs. `--count` MUST be non-negative and at most the record count; `--frac` MUST be in `[0, 1]`, and a positive fraction MUST yield at least one sample for a non-empty dataset.
- `sample --hardlink` MUST only be paired with outputs that expose `hardlink` ([pipeline](../pipeline.md#hardlink-hint)).
- Duplicate prompts assigned to different labels MUST be fatal.
- Exactly one of `--prompt` and `--reference-folder` MUST be given; both or neither MUST be fatal.
- `--reference-folder` MUST be accepted only for `yolo_e`, and MUST resolve to exactly one annotated image. `--model-id` MUST be accepted only for `llmdet` and `locate_anything`, and an unsupported checkpoint id MUST be fatal. `yolo_e` MUST re-validate text-XOR-reference in the provider.
- `sam3` MUST reject CPU execution.
- `rex_omni` box scores are coordinate-token likelihoods (`exp(mean log p)` over generated coordinate tokens), so `--threshold` filters them. Degenerate outputs that cannot be aligned to tokens fall back to score `1.0` with a warning.
- Post-filtering MUST compare annotation counts before and after so only new annotations are thresholded or suppressed. Score thresholding and IoU suppression MUST be group-based through `group_id`, and IoU suppression MUST run per output label after prompt-to-label remapping. Overlapping groups of different labels MUST be kept.
- Prompt labels MUST extend the dataset class list as needed. Temporary `ground_prompts` attributes MUST be removed in `finally`.
- OCR provider input MUST normalize `paddleOCR` to `paddleocr`.
