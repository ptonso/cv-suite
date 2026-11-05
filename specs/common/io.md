# Shared Dataset I/O

Related: **Read first** [records_dataset](records_dataset.md). **See also** [label adapters](../label/adapters.md), [vlm adapters](../vlm/adapters.md), [classify output](../classify/output.md).

## Overview

`cvsuite.common.io` reads a folder or file into the shared dataset and writes the dataset back out in a named on-disk format. It is a small router plus one adapter per format. New formats are added as new adapters without touching the rest of the system.

Branches keep their own public spellings and source-detection rules ([label adapters](../label/adapters.md), [vlm adapters](../vlm/adapters.md), [classify cli](../classify/cli.md)), then delegate parsing and writing here.

## Language

- **Adapter**: a class that reads and/or writes one format.
- **Format id**: the adapter's registered name, listed below.
- **Auto detection**: choosing a reader by asking each adapter's `matches()`, in priority order.
- **Claim**: a writer's declaration, before its first write, that it owns a directory or file for this run.
- **Dirty path**: a claimed path that already holds content, a file or a non-empty directory.

## Map

- `io/router.py`: registration, `read_dataset`, `write_dataset`, the built-in table.
- `io/<format>.py`: one adapter per format.
- `io/common.py`: shared mechanics and constants.

## Contracts

### Router (`io/router.py`)

```python
read_dataset(src, *, format="auto", task="auto", splits=None, **adapter_options) -> VisionDataset
write_dataset(dataset, dst, *, format, splits=None, overwrite=False, **writer_options) -> None
register_reader(name, adapter, *, priority=100) -> None
register_writer(name, adapter) -> None
list_readers() -> list[str]
list_writers() -> list[str]
```

Import the router from `cvsuite.common.io`, never from `cvsuite.common.core`: `io` imports `core`, so re-exporting would create an import cycle.

`read_dataset` stamps `dataset.source_format` and each `record.source_format` with the resolved format id. Built-in adapters register lazily on the first router call from the `_BUILTINS` table.

### Adapter interface

- `matches(src: Path) -> bool`: cheap on-disk shape check used by auto detection.
- `read(path, *, task, splits, **opts) -> VisionDataset`
- `write(dataset, dst, *, splits, claims, **opts) -> None` (writers only). `claims` is the run's `OutputClaims`. Writers that refuse existing files on their own accept it through `**opts` and ignore it.

### Output claims

`write_dataset` builds one `OutputClaims` per call from `overwrite` and the folders of the dataset's source images, and hands it to the writer. Before its first write into a directory or file, a writer calls `claims.claim(path)`:

- A path that is absent or empty is claimed.
- A dirty path raises `FileExistsError` naming the path and `--overwrite`.
- A dirty path with `overwrite=True` is removed, and only that path, then claimed.
- A path claimed earlier in the same run is a no-op, so several passes over one tree claim it once.

After the claim the path holds only this run's files. Records of one run whose names collide get `_1`, `_2`, and so on within it.

Claimed paths per writer:

| id | claimed paths |
|---|---|
| `yolo` | for each split written: `<split>/images`, each `<split>/labels*`, `<split>/labels_vqa` |
| `labelme` | each split directory written, or `dst` itself in the flat layout |
| `class-dir` | each class folder written: `<label>`, or `<split>/<label>` |
| `coco` | default layout: each `<split>` folder. `coco_style`: the shared `images/` folder, claimed whole, and each `annotations/annotations_<split>.json`. `no_images`: only the annotation files |

Content outside the claimed paths is left alone: a split tree or class folder that an earlier run wrote and this run does not write stays on disk. `vqa_style`, `flat_vlm_json`, `shards_vlm` refuse existing files themselves and claim nothing.

Folds are written by one `write_dataset` call per fold, each into its own `fold<i>` folder with its own claims ([split](split.md#writing-folds)).

### Linking images

Writers that place source images take the writer option `link_mode` (`copy`, `hard` or `sym`), set from `--link-mode` ([pipeline](../pipeline.md#link-mode)). They call `copy_or_link(src, dst, *, link_mode)` in `io/common.py`. It creates the parent folder, raises `FileExistsError` when `dst` exists, and links or copies as asked. A link that fails raises `OSError` naming both paths. It never falls back to a copy.

### Built-in formats

Auto detection picks the lowest priority number whose `matches()` is true.

| id | module | read | write | priority |
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

### Format details

- `flat`: loose image or tree into image-only records. Option `max_depth` (`-1` unlimited).
- `class-dir`: class-folder classification, single-label and multi-label duplicate-tree layouts. Read options `class_name_mode` (`basename` | `full_path`), `max_depth`. Write option `link_mode`. The writer places each record under its `record.split`: `<split>/<label>` when any record is not `train`, and `<label>` otherwise. It does not assign splits. Split-prefixed trees and threshold-based winner selection are handled inside.
- `yolo`: split trees and `data.yaml` for detection, instance segmentation and pose, plus optional `labels_vqa/` sidecars. `names` in `data.yaml` is a list or an index-to-name mapping, and both read as class names. Reads both the split-first layout (`<root>/<split>/images` + `<root>/<split>/labels*`) and the Ultralytics layout (`<root>/images/<split>` + `<root>/labels*/<split>`). `BBox.kind` (`det`, `seg`, `pose`) carries the box kind. Write options `link_mode` and `boxes_from_polygons`. The `det` label tree holds boxes of kind `det`, and with `boxes_from_polygons` also one box per polygon, taken from the polygon's extent, so a polygon-only dataset can be written as detection. Every task pass over a record uses the image name and label stem that the record received in the first pass, so the det, seg and pose label trees pair by stem.
- `coco`: COCO JSON or YAML-driven layouts (`images`, `annotations`, `categories`); detection and instance-segmentation geometry normalized; per-image VQA extras. The reader makes one record for every listed image, so images without annotations stay in the dataset as images without annotations. An annotation whose image or category is not listed raises. The writer always targets a directory. Write options `link_mode`, `coco_style`, `no_images`.
- `labelme`: image plus JSON sidecars. Boxes, polygons, keypoints, base64 images, shape flags in `attributes["flags"]`, promoted `prompt` / `model` / `text` fields, full source shapes on `record.labelme_shapes` and `imageData` under `record.round_trip["labelme"]`. Read option `keypoints_file`. Write options `embed_image`, `link_mode`, `split_layout`.
- `semseg_mask`: semantic-segmentation manifests (`data.yaml` with per-split `images` / `masks` dirs) into records carrying `semantic_mask`. Read-only.
- `vqa_style`: normalized VQA manifests (`items` plus image paths) and classic raw VQA JSON keyed by `image_id` / `question_id`. Write option `link_mode`. `flat_vlm_json` takes `link_mode` as well.
- `shards_vlm`: `.tar` shards of same-stem image and JSON members. The writer transcodes images to JPEG and reports a cleanup dir in metadata.
- `flat_vlm_json`: image plus same-stem JSON sidecar, recursive. A sidecar's image is resolved in its own directory first, then tree-wide, then by the parallel `labels*/` to `images/` mirror path. This keeps a bucket-partitioned export (`map-answers ... to-json`) re-ingestable.

### Shared constants (`io/common.py`)

`io/common.py` holds shared adapter mechanics (YAML/JSON parsing, lazy image metadata, path resolution, sample keys, `copy_or_link`) and constants, notably `CLASS_DIR_MULTI_LABELS_ATTR` (consumed by [classify output](../classify/output.md)) and `LABELME_META_KEY = "cvsuite"` (LabelMe JSON namespace for round-trip metadata).

## Constraints

- `format="auto"` MUST pick the adapter with the lowest priority number whose `matches()` is true; a tie at the same priority MUST raise.
- An unknown explicit `format` MUST raise `KeyError`. A missing `src` MUST raise `FileNotFoundError`. `write_dataset` with an unregistered or reader-only format MUST raise `KeyError`.
- Reading a YOLO source that resolves images but matches no label files MUST raise; image-only trees go through `flat`.
- In `flat_vlm_json`, only a same-directory `stem.jpg` plus `stem.png` clash MAY raise.
- Adapters MUST read and write only through the shared dataset and MUST NOT receive branch-specific adapters or CLI state.
- Geometry MUST be normalized in memory; adapters convert to and from format layouts.
- `coco` writing MUST target a directory.
- A writer that places files in a destination MUST claim every directory or file it writes before the first write into it, except the writers listed above as refusing existing files themselves.
- A dirty claimed path MUST raise `FileExistsError` unless `overwrite=True`, and `overwrite=True` MUST remove only that path.
- A claim with `overwrite=True` MUST raise when the path is, or contains, a folder of a source image of the dataset being written, and when the path is a symbolic link.
- Within one run, a record MUST keep one image name and one label stem across every task pass of the `yolo` writer.
- Two `write_dataset` calls into one destination MAY write different splits. A `coco_style` destination MUST be written by one call that carries every split.
- A coco round trip MUST keep images without annotations.
- A `data.yaml` whose `names` is a mapping MUST read the names, not the indices, as classes.
- `copy_or_link` MUST NOT fall back to a copy when a link fails, and `sym` MUST write an absolute link to the resolved source.
- Multi-source COCO merging MUST match classes by name and images by file, never by the numeric ids of one file.

## History

- [Shared dataset contract](../logs/2026-09-20_shared-dataset-contract.md): why format handling is decoupled from the rest of the system.
- [Idempotent outputs](../logs/2026-11-05_idempotent-outputs.md): why writers claim what they write, and why a rerun raises unless `--overwrite`.
- [Chained transforms and split as a transform](../logs/2026-11-05_chained-transforms-and-split.md): why links are controlled by `link_mode` and folds are written one call each.
