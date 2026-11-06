# Idempotent output writers

Affects: [io](../common/io.md), [label output](../label/output.md), [classify output](../classify/output.md), [architecture](../architecture.md).

Recorded from an audit of the output writers and the maintainer's approval of the plan, 2026-11-05.

## Situation

The YOLO, LabelMe and class-dir writers, and the default COCO image writer, resolve a file name that already exists by adding `_1`, `_2`, and so on. The suffix was written in the initial release for collisions between records of one run: several sources merged into one output, `--preserve-splits` or `--val-frac` placing records of different source folders in one split tree, and several records of one image. No spec, test or commit message covers a rerun into a destination that an earlier run filled.

A downstream project that reran one script three times got three copies of its evaluation view and its training view, with non-numeric stems that broke COCO image ids. Nothing failed at write time. The architecture spec and the guide claimed that a re-run reproduces the same output, which this behavior contradicts. The other writers (`to-vqa-style`, `to-json`, `to-shards`, `gen to-dst`) already refuse an existing file, and `prep arrange` merges by documented contract.

## Alternatives considered

- **Overwrite by default.** A rerun would silently delete or replace output, including output of a different dataset written to the same folder. The writers would also need a rule for files the new run does not produce.
- **Skip existing files.** A rerun after a changed sample or recipe would mix old and new files, and the manifests could not say which is which.
- **Refuse a non-empty destination.** It matches the fail-fast writers, and it breaks the working pattern of writing `train` and `val` in separate invocations into one root.
- **Per-run manifest that records what each run wrote.** It allows suffixes for same-run collisions only, at the price of a new persisted file and its migration. The claim rule below gives the same result without a manifest.
- **Leave the behavior and document it.** The caller would keep carrying an external `rm -rf` that every frozen script has to remember.

## Decision

Writers claim each directory or file they write, before the first write. A claimed path that already holds content raises, and `--overwrite` replaces exactly that path. After the claim the path holds only this run's files, so the existing suffix rule keeps resolving same-run collisions unchanged. The contract is in [io](../common/io.md#output-claims).

A record keeps one image name and one label stem for the whole run, so the det, seg and pose label trees pair by stem. The helper `data.yaml` of `to-yolo` and `to-coco` lists the split trees present on disk, so it stays true across invocations.

## What tipped it

The train-then-val pattern. Claims are per split tree, so it keeps working, and the same rule covers the failure that cost the downstream project a day.

## Trade-offs accepted

- Callers that relied on silent accumulation now get `FileExistsError` and must pass `--overwrite`.
- `--overwrite` clears only the trees the run writes. A split tree or class folder from an earlier run that this run does not write stays on disk.
- COCO with `--coco-style` shares one `images/` folder across splits, so it claims that folder whole and takes every split in one invocation. The default COCO layout is per split and supports split-by-split runs.
- A claimed path that contains a source image of the dataset refuses `--overwrite`, because replacing it would delete the input.
