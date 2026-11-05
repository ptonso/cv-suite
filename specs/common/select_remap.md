# Select and Remap Classes

Related: **Read first** [records_dataset](records_dataset.md). **See also** [label transforms](../label/transforms.md), [classify transforms](../classify/transforms.md), [pipeline](../pipeline.md#chaining), [eda cli](../eda/cli.md).

## Overview

Two transforms narrow a dataset before it is split, analyzed or written. `select` removes things by their properties: small annotations, classes, images by name. `remap-classes` changes the vocabulary: it merges, renames and drops classes by a YAML map. Both are shared by the `label` and `class` branches and by `eda`, and both are plain flags. Neither has an expression language.

They also answer the "which question am I asking" part of an analysis. To study defects against normal parts in a dataset with a thousand classes, `remap-classes --map defects.yaml --others normal` leaves two classes, and every later step sees two.

Example: `cvsuite label ./raw select --min-box-px 20 --exclude-name 'pcb98rec1' --empty drop remap-classes --map classes.yaml --others other split --val-frac 0.2 to-yolo ./out`.

## Language

- **Annotation filter**: a `select` flag that removes boxes, polygons or keypoints.
- **Emptied record**: a record with annotations before a transform and none after it.
- **Others**: the class that receives every class the map does not mention.

## Map

- `common/transforms/select.py` and `common/transforms/remap.py`: the shared logic.
- `label/transform/select/commands/run.py`, `label/transform/remap_classes/commands/run.py` and the same names under `classify/transform/commands/`: thin command modules.

## Contracts

### select

```
select [--min-box-px N] [--min-side-px N] [--min-area-frac F]
       [--keep-class A,B | --drop-class A,B]
       [--exclude-name REGEX]... [--empty {drop,keep}]
```

- `--min-box-px N` removes annotations whose `sqrt(width * height)` is below N source pixels. `--min-side-px N` uses the longer side. `--min-area-frac F` uses the share of the image area. Pixel sizes use the image's width and height.
- `--keep-class` keeps only the listed classes. `--drop-class` removes them. The two are exclusive. The class list is renumbered to the remaining classes in their original order, and class ids on boxes, polygons and keypoints follow.
- `--exclude-name REGEX` removes records whose [layout path](keys.md#layout-path) matches `re.search`. It is repeatable.
- `--empty` says what happens to an emptied record: `drop` removes it, `keep` keeps it as an image without annotations. It is required whenever an annotation filter or a class flag runs on a dataset with geometry. Keeping silently turns images with unlabeled small objects into negatives.
- Pixel and area flags raise on a dataset with no boxes or polygons.
- On a classification dataset, `--keep-class` and `--drop-class` act on labels. A record whose labels are all removed is dropped, and `--empty` does not apply.

### remap-classes

```
remap-classes --map FILE.yaml [--others NAME] [--empty {drop,keep}]
```

```yaml
merge:
  light: [canopy_light, led_highbay_light, led_linear_light, tube_light]
rename:
  gauge_face: analog_gauge
drop: [Background]
```

- `merge` maps a new name to the classes it replaces. `rename` maps one class to a new name. `drop` removes classes with their annotations.
- The new class list holds the `merge` names in file order, then the `rename` targets in file order, then the unmapped classes in their original order, then `--others` if given. `--others NAME` collects every class that no section mentions.
- Without `--others`, unmapped classes keep their names.
- Everything that names a class follows: `cls` and `label` on boxes, polygons and keypoints, `dataset.classes`, classification `label` and `probs` (probabilities of merged classes add up), the multi-label attributes of class datasets, `labelme_shapes` and `round_trip["labelme"]` labels, and `round_trip["coco"]["categories"]`, which is rebuilt from the new list.
- `--empty` applies when a drop or a merge into `drop` empties records, with the same rule as `select`.

## Constraints

- A class named in `--keep-class`, `--drop-class` or the map that does not exist in the dataset MUST raise, naming it.
- A class MUST NOT appear in two sections of the map or twice in one list. A new name that equals an existing class MUST be listed among its own sources, otherwise the command raises, so a merge cannot capture a class by accident.
- `--others` MUST NOT name a class that the map also creates.
- A dataset with semantic masks MUST be rejected by `remap-classes` and by the class flags of `select`.
- After either transform, every class id MUST index `dataset.classes`, and no class MAY have a name that two ids share. A test checks this on boxes, polygons, keypoints and classification data.
- A dataset left with no records MUST raise.

## History

- [What belongs in cvsuite](../logs/2026-11-05_what-belongs-in-cvsuite.md): the filters and vocabulary merges that projects wrote by hand.
- [The eda branch](../logs/2026-11-05_eda-branch.md): why a question is narrowed with these transforms.
