# Cookbook

Copy-paste recipes for the common workflows. Every command follows `cvsuite <branch> <src> [transform] <output>`; see [Getting started](getting-started.md) for the shape and [`specs/`](../specs/) for every flag.

## Convert between annotation formats

```bash
cvsuite label ./dataset-yolo    to-coco    ./dataset-coco
cvsuite label ./dataset-coco    to-labelme ./dataset-labelme
cvsuite label ./dataset-labelme to-yolo    ./dataset-yolo
```

Every output is a directory. `to-coco` writes a COCO dataset folder
(`<split>/_annotations.coco.json` + images + a helper `data.yaml`); add
`--no-images` to write just the annotations.

Ingest is auto-detected. Force it with `--from {yolo,coco,labelme,images,class-dir,semseg-mask}` if a folder is ambiguous.

Merge several sources into one export (classes are matched by name, not id):

```bash
cvsuite label ./train-a ./train-b ./train-c to-coco ./merged
```

## Sample a subset

```bash
# annotated datasets: keep annotations, keep splits
cvsuite label ./dataset sample --count 500 to-yolo ./subset
cvsuite label ./dataset sample --frac 0.1  to-coco ./subset

# hard-link instead of copying bytes (fast, same filesystem)
cvsuite label ./dataset sample --count 500 --hardlink to-yolo ./subset
```

For flat image folders use `cvsuite prep sample`; for class folders use `cvsuite class ... sample`.

## Auto-label with a grounding model

```bash
cvsuite label ./images ground --provider gsam --prompt "fire, smoke, person" to-coco ./grounded
```

- The prompt can be one string, or a YAML/JSON file that is either a list of phrases or a `label -> phrases` map.
- Filter weak boxes with `--threshold`, deduplicate with `--iou-threshold`.
- Providers: `sam3`, `gsam`, `gdino`, `llmdet`, `locate_anything`, `rex_omni`, `yolo_e`. `sam3` requires a GPU.

## Add OCR boxes

```bash
cvsuite label ./receipts ocr --provider paddleocr to-labelme ./ocr-out
```

OCR boxes are appended to existing annotations with `kind="ocr"` and the recognized string in `.text`.

## Classify an image folder

```bash
cvsuite class ./images infer --provider clip --prompt "good,bad" to-class-dir ./classified
```

- `--prompt` is a comma-separated label list, a YAML/JSON list, or a `label -> prompt(s)` map.
- With a list, `--template-prompt "a photo of a <class>"` wraps each label.
- Route low-confidence images aside with `to-class-dir --threshold 0.6` (they land in an `unlabeled/` folder).
- Add train/val/test folders on the way out: `to-class-dir --val-frac 0.1 --test-frac 0.1`.

Class-aware resampling of an already-labeled tree:

```bash
cvsuite class ./classified sample --mode balance  --max-n-per-class 200 to-class-dir ./balanced
cvsuite class ./classified sample --mode preserve --max-frac-per-class 0.5 to-class-dir ./smaller
```

## Caption or answer questions with a VLM

```bash
# one caption per image
cvsuite vlm ./images caption --provider qwen to-json ./captions

# free-form question over every image
cvsuite vlm ./images ask --prompt "What safety equipment is visible?" --provider qwen to-json ./answers

# answer questions already attached to a VQA dataset
cvsuite vlm ./vqa.json vqa --provider qwen to-json ./predictions --skip-images

# normalize free-text answers to a fixed label set
cvsuite vlm ./predictions map-answers yes no unsure to-json ./mapped
```

Export to shards or a VQA-style manifest instead:

```bash
cvsuite vlm ./vqa.json vqa --provider qwen to-shards ./exports
cvsuite vlm ./predictions to-vqa-style ./manifest
```

## Generate and edit images

```bash
# text to image
cvsuite gen create --provider flux --prompt "a red fox in the snow" to-dst ./fox.png

# many prompts, several images each, into a directory
cvsuite gen create --provider flux --prompt prompts.yaml --num-images 4 to-dst ./out-dir

# edit existing images (reuses label-style ingest for annotated sources)
cvsuite gen edit ./photos --prompt "make it night, keep the layout" to-dst ./edited
```

Provider-specific knobs go through repeatable `--model-arg key=value`; `--width` / `--height` override any size passed that way.

## Clean up raw image folders (`prep`)

```bash
# unzip, flatten, dedupe, sequential rename
cvsuite prep arrange ./dump ./clean --unzip --flatten plain --exact-dedup --rename-seq

# normalize pixels: cap the long side, convert to JPEG
cvsuite prep process ./clean ./normalized --size 1024 --resize-mode cap-long --to-format jpg

# fix rotation with a small orientation model (runs on CPU)
cvsuite prep orient ./normalized ./oriented --batch 16

# pool several folders and sample a flat set
cvsuite prep sample ./a ./b ./c ./sampled --count 300 --hardlink
```

`arrange` -> `process` -> `sample` chains directly: `sample` skips the
`_non_images/` folder and `manifest.yaml` that the earlier steps leave
behind. Add `--dry-run` to any prep command to see the plan without
touching the filesystem.

## Emit a stats sidecar

Outputs that support it take `--with-stats`, writing `stats.yaml` beside the result:

```bash
cvsuite label ./dataset to-coco ./out --with-stats
cvsuite class ./images infer --provider clip --prompt "a,b" to-class-dir ./out --with-stats
```
