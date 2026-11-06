# What belongs in cvsuite

Affects: [architecture](../architecture.md), [keys](../common/keys.md), [split](../common/split.md), [select and remap](../common/select_remap.md), [sampling](../common/sampling.md), [similarity](../common/similarity.md), [prep transforms](../prep/transforms.md), [label transforms](../label/transforms.md), [classify transforms](../classify/transforms.md), [eda cli](../eda/cli.md).

Recorded from a survey of the maintainer's sibling projects and the maintainer's account, 2026-11-05.

## Situation

cvsuite began as the boilerplate every vision project needs: converting formats, resizing labels, organizing class folders, hardlinking so datasets do not explode. Foundation-model tools were merged in later. Projects built on it (pcb-mc-ml, synth-label, petro-cv, teia-lab) still hand-write deterministic dataset code that cvsuite does not offer:

- Grouped and stratified splitting, cross-validation folds, pinning an official test split. pcb-mc-ml (`prep/cls_exp/transform.py`, `nodes.py`) and synth-label (`tools/prep_binary.py`, `prep_cv.py`, `sdnet_split.py`) each wrote their own.
- Cross-split near-duplicate detection, in pcb-mc-ml as a dHash drop and in synth-label as SSIM "twin" analysis in about eight scripts.
- Dropping boxes by size, excluding boards or classes, merging class vocabularies.
- Constrained negative sampling and square context crops (pcb-mc-ml `square_context_box`, `add_background_nodes`).
- Nested fractions of a dataset (synth-label fraction sweeps, teia-lab `subsample.py`).
- Per-dataset exploratory analysis: structure, twins and metadata leaks, with the same plotting and `guess_accuracy` code copied across five synth-label datasets.
- Two input gaps: the COCO reader drops images that carry no annotations, and the YOLO det writer skips boxes derived from polygons. `synth-label/tools/stage_det.py` exists to work around both.

## Alternatives considered

- **Add every recurring routine as a command.** The person-polygon analysis in synth-label shows where this ends: the general interface would be larger than the code it replaces, and cvsuite would absorb one project's nuances.
- **Leave everything in the projects.** The boilerplate stays, and projects keep staging data by hand to fit around cvsuite's gaps.
- **An expression language for filtering** (`--where "area>20 and class in (...)"`). It invites every project's logic into the CLI.
- **Python API only.** `read_dataset` and `write_dataset` exist, but a library call cannot be chained, previewed or reproduced from a shell script the way a command can.

## Decision

A routine belongs in cvsuite when it passes four checks:

1. It recurs in unrelated projects, in about the same shape.
2. The project-specific part can enter as data: a regex, a column name, a mapping file, a number.
3. It works on what the [shared dataset](../common/records_dataset.md) already models.
4. Checking its result is cheaper than rewriting it.

What passes: [splitting](../common/split.md) with group and stratum keys, folds and pinning; [perceptual similarity](../common/similarity.md) as a group key and as an audit; [select and remap-classes](../common/select_remap.md); [nested sampling](../common/sampling.md); the I/O gaps above; constrained negatives and context crops; and the [eda branch](../eda/cli.md).

What stays in projects: person-polygon analysis, kNN graph construction, augmentation, dataset fetching, logistic background probes, and anything whose general interface would be larger than the code.

Two escape hatches keep nuance out of the tool. The `file:` key reads a group per image from a CSV that a project writes with a short script. The shared dataset and `read_dataset` / `write_dataset` stay usable from Python.

## What tipped it

The splitting code in the sibling projects differs only in how a group is defined: two regexes on the file name in pcb-mc-ml, `frame_id // bin_size` in synth-label, EXIF capture times and SSIM components for SDNET. After the group is defined, the algorithm is the same: assign whole groups to splits or folds so that per-class counts approach target shares. The group definition is data, and the algorithm is general.

## Trade-offs accepted

- Some recurring code stays in projects, and exotic grouping needs a CSV.
- pcb-mc-ml's "Missing X becomes X with a visibility flag" moves information from a class into an attribute, and `remap-classes` does not express it.
- Project owners migrate their own scripts. This change does not touch pcb-mc-ml, synth-label or petro-cv.
