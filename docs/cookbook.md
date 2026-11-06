# Cookbook

Copy-paste recipes for the common workflows. Every command follows `cvsuite <branch> <src> [transform] <output>`; see [Getting started](getting-started.md) for the shape and [`specs/`](../specs/) for every flag.

## Clean up raw image folders (`prep`)

```bash
# unzip, flatten, dedupe, sequential rename
cvsuite prep arrange ./dump \
    --unzip \
    --flatten plain \
    --exact-dedup \
    --rename-seq \
    ./clean

# normalize pixels: cap the long side, convert to JPEG
cvsuite prep process ./clean \
    --size 1024 \
    --resize-mode cap-long \
    --to-format jpg \
    ./normalized

# pool several folders and sample a flat set
cvsuite prep sample ./a ./b ./c \
    --count 300 \
    --link-mode hard \
    ./sampled
```

`arrange` -> `process` -> `sample` chains directly: `sample` skips the
`_non_images/` folder and `manifest.yaml` that the earlier steps leave
behind. Add `--dry-run` to any prep command to see the plan without
touching the filesystem.

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

# hard-link or symlink instead of copying bytes (a failed link raises)
cvsuite label ./dataset sample --count 500 to-yolo ./subset --link-mode hard
```

`--stratify-by` samples inside each group of records that share a key, so a fraction holds in every split (or every folder, or every attribute value). A smaller fraction stays a subset of a larger one.

```bash
# 20% of train, 20% of val and 20% of test
cvsuite label ./dataset sample --frac 0.2 --stratify-by split to-yolo ./subset

# 20% of every class inside every split
cvsuite class ./dataset sample --mode preserve --max-frac-per-class 0.2 --stratify-by split to-class-dir ./subset
```

For flat image folders use `cvsuite prep sample`; for class folders use `cvsuite class ... sample`.

## Auto-label with a grounding model

```bash
# inline: one literal prompt, used as a single class
cvsuite label ./images ground \
    --provider gsam \
    --prompt "person in a safety vest" \
    to-coco ./grounded

# yaml: several classes, each with synonym phrases
cvsuite label ./images ground \
    --provider gsam \
    --prompt prompts.yaml \
    to-coco ./grounded
```

```yaml
# prompts.yaml — label -> phrase(s); every phrase is forwarded to the model
# and detections come back under the one label they're grouped under
fire:
  - fire
  - flames
smoke:
  - smoke
person:
  - person
  - human
```

- An inline `--prompt` is **one literal string** — commas are kept as-is, not split into classes.
- A YAML file can also just be a flat list (`- fire`, `- smoke`, `- person`) when you don't need synonyms.
- Filter weak boxes with `--threshold`, deduplicate with `--iou-threshold`.
- Providers: `sam3`, `gsam`, `gdino`, `llmdet`, `locate_anything`, `rex_omni`, `yolo_e`. `sam3` requires a GPU.

## Add OCR boxes

```bash
cvsuite label ./receipts ocr --provider paddleocr to-labelme ./ocr-out
```

OCR boxes are appended to existing annotations with `kind="ocr"` and the recognized string in `.text`.

## Classify an image folder

```bash
cvsuite class ./images infer \
    --provider clip \
    --prompt "good,bad" \
    to-class-dir ./classified
```

`--prompt` here is always inline, never a file path. A comma list is wrapped with `--template-prompt`; for explicit phrases per class, pass a JSON map instead:

```bash
cvsuite class ./images infer \
    --provider clip \
    --prompt '{"good": ["a clean part", "no visible defects"], "bad": ["a damaged part", "scratched"]}' \
    to-class-dir ./classified
```

- Route low-confidence images aside with `to-class-dir --threshold 0.6` (they land in an `unlabeled/` folder).
- Add train/val/test folders: `split --val-frac 0.1 --test-frac 0.1` before `to-class-dir`.

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

# several questions per image, from a YAML file (a flat list, or label -> question(s))
cvsuite vlm ./images ask --prompt questions.yaml --provider qwen to-json ./answers

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

# many prompts, several images each, into a directory (prompts.yaml: a flat list, or id -> prompt)
cvsuite gen create --provider flux --prompt prompts.yaml --num-images 4 to-dst ./out-dir

# edit existing images (reuses label-style ingest for annotated sources)
cvsuite gen edit ./photos --prompt "make it night, keep the layout" to-dst ./edited
```

Provider-specific knobs go through repeatable `--model-arg key=value`; `--width` / `--height` override any size passed that way.

## Emit a stats sidecar

Outputs that support it take `--with-stats`, writing `stats.yaml` beside the result:

```bash
cvsuite label ./dataset to-coco ./out --with-stats
cvsuite class ./images infer --provider clip --prompt "a,b" to-class-dir ./out --with-stats
```


## Split without leaking

`split` keeps every group on one side and balances class counts across splits. A group is anything a key spec names: a regex on the path, a folder level, an attribute, a CSV column or perceptually similar images.

```bash
# groups from the file name, class-balanced, 20% val and 20% test
cvsuite label ./boards split --group-by 'regex:^(\w+?)(?:_top|_bottom)?\.' --val-frac 0.2 --test-frac 0.2 to-yolo ./out

# near-duplicate photos stay together too
cvsuite label ./photos split --group-by 'dir:0' --group-by 'similar:6' --val-frac 0.2 to-coco ./out

# five cross-validation folds: ./out/fold0 ... ./out/fold4, each with its own data.yaml
cvsuite label ./boards split --group-by 'regex:^(\w+?)_' --folds 5 --val-frac 0.15 to-yolo ./out

# only fold 2, identical to the fold2 of the full run
cvsuite label ./boards split --group-by 'regex:^(\w+?)_' --folds 5 --val-frac 0.15 --only-fold 2 to-yolo ./out
```

The command prints the target and achieved share of each class per split and fails when a class is further than `--tolerance` (default 0.05) from its target. `--keep test` pins the records already in `test` and splits the rest.

A group with a kept record is kept whole, so the kept split can grow. `--on-siblings drop` removes the other members of those groups instead, and the kept split stays exactly as it was:

```bash
cvsuite label ./boards split --keep test --group-by 'regex:^(\w+?)_' --on-siblings drop --val-frac 0.2 to-yolo ./out
```

`--stratify-by class-size:20` balances each class twice, for boxes under 20 px and boxes from 20 px up (`sqrt(w * h)` in source pixels). More thresholds make more buckets: `class-size:20,64`.

## Narrow the question, then split

`select` removes things by property and `remap-classes` changes the vocabulary. Chain them before `split` so every later step sees the narrowed dataset.

```bash
cat > classes.yaml <<'YAML'
merge:
  light: [canopy_light, led_highbay_light, tube_light]
rename:
  gauge_face: analog_gauge
drop: [Background]
YAML

cvsuite label ./raw \
    select --min-box-px 20 --exclude-name 'rec1' --empty drop \
    remap-classes --map classes.yaml --others other \
    split --group-by 'dir:0' --val-frac 0.2 \
    to-yolo ./out
```

`--empty` is required whenever a filter can leave an image without annotations: `drop` removes the image, `keep` keeps it as a negative. Samples are nested: for one `--seed`, `sample --frac 0.1` is a subset of `sample --frac 0.25`.

## COCO with background images

The COCO reader keeps every listed image, so images without annotations stay in the dataset as negatives. An annotation that names an unknown image or category raises instead of landing in class 0.

```bash
cvsuite label ./coco-with-backgrounds split --val-frac 0.2 to-yolo ./yolo-with-backgrounds
```

The COCO reader looks for each listed image next to the JSON, in a sibling `images/` folder and under the dataset root. When the images live elsewhere, name the folders (repeat the flag for several):

```bash
cvsuite label ./annotations --from coco --images-dir ./photos/loosen --images-dir ./photos/missing to-yolo ./out
```

A listed image that is found nowhere raises and names the first few.

## Polygons as detection boxes

A segmentation dataset can train a detector. `--boxes-from-polygons` writes one detection box per polygon, taken from the polygon's extent, next to the segmentation labels.

```bash
cvsuite label ./seg-dataset to-yolo ./det-view --task det --boxes-from-polygons
```

## Negatives and context crops

`make-negatives` adds background windows that touch no box, polygon or keypoint. It fails when an image cannot hold them, and lists the images that fell short.

```bash
# two backgrounds per annotated object, each sized like one of the image's boxes
cvsuite label ./dataset ops make-negatives --ratio 2 --size-from boxes --on-shortfall warn to-yolo ./with-negatives
```

`--emit box --label NAME` adds each background window as a box of that class on its own image, with no new image written. `crop-dets` and every output then treat it like any annotation, and `--within annotations` keeps the boxes inside the area the real boxes span.

```bash
cvsuite label ./dataset ops make-negatives --emit box --label background --size-from boxes --ratio 0.07 --within annotations ops crop-dets --imgsz 224 --pad-type shift to-labelme ./crops
```

`crop-dets --pad-type shift` cuts a square of real surrounding pixels around each box, slid inside the image, with no padding. `--margin-px` adds pixels on each side and `--manifest` records where every crop came from.

```bash
cvsuite label ./dataset ops crop-dets --imgsz 224 --pad-type shift --margin-px 8 --manifest crops.json to-class-dir ./crops
```
