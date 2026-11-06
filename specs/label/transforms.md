# Label Transforms

Related: **Read first** [label cli](cli.md). **See also** [label filter](filter.md), [label adapters](adapters.md), [label output](output.md), [split](../common/split.md), [select and remap](../common/select_remap.md), [sampling](../common/sampling.md), [fm_runtime](../common/fm_runtime.md).

## Overview

Label transforms change an annotated dataset before an output writes it, and they chain left to right ([pipeline](../pipeline.md#chaining)). Some are deterministic (sampling, selecting, remapping classes, splitting, geometry operations, filtering), and some use foundation models to add annotations (grounding, OCR). Filtering is large enough for its own spec: [label filter](filter.md). `split`, `select` and `remap-classes` are shared with the `class` branch and described in [split](../common/split.md) and [select and remap](../common/select_remap.md).

## Language

- **Grounding**: finding boxes or masks in an image that match text prompts (or a reference image).
- **Group id**: links a box and its polygon so they are scored and suppressed together.
- **Prompt label**: the label a prompt is remapped to on output.

## Map

- `label/transform/sample/`, `select/`, `remap_classes/`, `split/`, `ground/`, `ocr/`, `ops/`, `filter/`, each with `commands/`. The command modules of `select`, `remap-classes` and `split` call `common/transforms/` and `common/splitting.py`.
- `label/core/crops.py`, `annotate/`: crop and annotation helpers.

## Contracts

### sample

`--count` or `--frac`, `--stratify-by KEY[,KEY...]`, `--seed`. Sampling over records that keeps annotations and existing splits, following [nested sampling](../common/sampling.md), globally or inside each [stratum](../common/sampling.md#strata) (`--stratify-by split` keeps the fraction in every split): records are ranked by their layout path, so a smaller sample is a subset of a larger one for the same seed. The sample size and seed are reported through the step's `info`.

### split, select, remap-classes

Contracts are in [split](../common/split.md#command) and [select and remap](../common/select_remap.md). The command names are `split`, `select` and `remap-classes`.

### ground

Arguments: `--provider {sam3,gsam,gdino,llmdet,locate_anything,rex_omni,yolo_e}`; `--model-id` (`llmdet`, `locate_anything` only); `--prompt` (exclusive with `--reference-folder`); `--reference-folder` (`yolo_e` only, ingested via [adapters](adapters.md) into a one-image reference dataset); `--device`, `--batch`, `--precision`, `--config`, `--no-resume`, `--threshold`, `--iou-threshold`. The module declares `FM = True`.

Two exclusive modes:

- Text-prompt: one raw string (one prompt item, never split on punctuation) or a YAML/JSON spec (a flat list, or a label-to-prompts mapping). Each prompt is sent per image and emitted back as the exact label or prompt string.
- Reference-image (`yolo_e` only): the reference dataset's class names become the output labels.

Runs the runtime with task family `ground`; only newly added annotations are post-filtered.

Flow: parse the prompt spec into flat prompts plus a prompt-to-label mapping; attach the temporary record attribute `ground_prompts`; call the runtime; compare old and new annotation sets; filter low-score or overlapping new groups; remap prompt labels, update class ids, normalize the dataset task.

### ocr

`--provider/--model`, `--device`, `--batch`, `--precision`, `--config`, `--no-resume`. The module declares `FM = True`. Builds `FMRequest(task="ocr", provider="paddleocr", ...)` and appends OCR boxes with `BBox.kind == "ocr"` and the text in `BBox.text`. Existing annotations stay.

### ops

Project-level surface routing to `rebox`, `make-negatives`, `crop-dets` and `inverse-crop-dets`.

- `rebox`: recompute one detection box from visible keypoints, replacing record boxes with a single box using the first keypoint class. `expand` and `adaptive` modes behave identically, and `edge_tol` is unused.
- `make-negatives`: `--splits`, `--imgsz` (square window), `--per-image K` or `--ratio R` (negatives per annotated instance of the image, exclusive with `--per-image`), `--size-from {fixed,boxes}` (`boxes` sizes each window like a randomly chosen box of the same image), `--emit {image,box}` (default `image`), `--label NAME`, `--within {image,annotations}` (default `image`), `--on-shortfall {error,warn}` (default `error`), `--seed`, `--preview`. A window is accepted only when it intersects no box, polygon or keypoint rectangle of its image, and it is never clipped, so an image smaller than the window yields none. Each negative gets up to 100 placement tries. A shortfall lists the images and counts that could not be filled.
  - `--emit image` writes the windows under a temporary root (`vislabel_neg_*`) with names built from the layout path and an index, and appends them as empty-annotation records with `negative=True`.
  - `--emit box` writes no image. It adds one box per window to the source record, with the class `--label` (required, appended to the dataset's classes when new) and `attributes["negative"] = True`, so `crop-dets`, `select`, `split` and the outputs treat it like any annotation. A box also avoids the boxes placed before it on the same image. `--preview` is rejected, and no temporary root is created.
  - `--within annotations` restricts the windows to the extent of the image's boxes, polygons and keypoints, and an image without annotations is a shortfall. `image` allows the whole image.
- `crop-dets`: crops detections or polygons into a new temporary dataset (`vislabel_crop_*`). `imgsz` and `long_size` are exclusive; padding modes `background`, `black`, `gray`, `white`, and `shift`. `--margin-frac` adds a fraction of the box size on each side and `--margin-px` adds source pixels on each side, on top of it. `shift` requires `--imgsz`: it grows the expanded box to a square from real surrounding pixels, slides the square inside the image, never pads, and uses a side of at most the shorter image side. Each crop keeps only its triggering object, with a matching polygon when one is associated; geometry is rewritten to the crop canvas; the classification label comes from the source label. Each crop record carries its provenance in `attributes["crop"]`: `crop_id`, `source_image`, `source_box_px` (`x, y, w, h`), `crop_rect_px` (`x1, y1, x2, y2`), `margin_frac`, `margin_px`, `pad_type`. `--manifest FILE.json` writes the list of those entries, each with the crop's `file`, `split` and `label`. Output can go to normal writers.
- `inverse-crop-dets`: takes a `modified_crops` dataset (a LabelMe crop dataset produced from `crop-dets`), ingests it through the LabelMe adapter, and pastes its annotations back onto the original images using the recorded `crop_id` (temporary root `vislabel_inverse_crop_*`).

`make-negatives --emit image`, `crop-dets` and `inverse-crop-dets` list their temporary roots in `meta["cleanup_dirs"]`.

### filter

Routes to `train-logistic` and `apply`; see [label filter](filter.md).

## Constraints

- `sample` MUST keep annotation payloads and MUST NOT balance classes: without `--stratify-by` it is a global sampler. `--count` MUST be non-negative and at most the record count; `--frac` MUST be in `[0, 1]`, and a positive fraction MUST yield at least one sample for a non-empty dataset.
- Duplicate prompts assigned to different labels MUST be fatal.
- Exactly one of `--prompt` and `--reference-folder` MUST be given; both or neither MUST be fatal.
- `--reference-folder` MUST be accepted only for `yolo_e`, and MUST resolve to exactly one annotated image. `--model-id` MUST be accepted only for `llmdet` and `locate_anything`, and an unsupported checkpoint id MUST be fatal. `yolo_e` MUST re-validate text-XOR-reference in the provider.
- `sam3` MUST reject CPU execution.
- `rex_omni` box scores are coordinate-token likelihoods (`exp(mean log p)` over generated coordinate tokens), so `--threshold` filters them. Degenerate outputs that cannot be aligned to tokens fall back to score `1.0` with a warning.
- Post-filtering MUST compare annotation counts before and after so only new annotations are thresholded or suppressed. Score thresholding and IoU suppression MUST be group-based through `group_id`, and IoU suppression MUST run per output label after prompt-to-label remapping. Overlapping groups of different labels MUST be kept.
- Prompt labels MUST extend the dataset class list as needed. Temporary `ground_prompts` attributes MUST be removed in `finally`.
- OCR provider input MUST normalize `paddleOCR` to `paddleocr`.
- A negative from `make-negatives` MUST NOT intersect any annotation of its source image, and a shortfall MUST fail unless `--on-shortfall warn` is given.
- `make-negatives --emit box` MUST require `--label`, and `--label` MUST be rejected with `--emit image`. Two boxes it adds to one image MUST NOT intersect.
- `sample --stratify-by class` MUST raise.
- A `shift` crop MUST NOT contain padding, and `--pad-type shift` without `--imgsz` MUST raise.
- `crop-dets` provenance MUST be enough to cut the same window again from the source image.

## History

- [What belongs in cvsuite](../logs/2026-11-05_what-belongs-in-cvsuite.md): why selection, class remapping, constrained negatives and context crops belong here.
- [Chained transforms and split as a transform](../logs/2026-11-05_chained-transforms-and-split.md): why transforms chain and `sample` lost `--hardlink`.
- [Gaps found by migrating the projects](../logs/2026-11-06_migration-gaps.md): why `sample` takes strata and `make-negatives` can add boxes.
