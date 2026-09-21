# Stats

Related: **Read first** [records_dataset](records_dataset.md). **See also** [classify output](../classify/output.md), [label output](../label/output.md), [vlm output](../vlm/output.md), [gen output](../gen/output.md).

## Overview

Output commands that opt in with `--with-stats` write a `stats.yaml` describing what they wrote. Stats are built at writer time from the final dataset, the destination and branch run metadata, so each branch keeps ownership while the document shape, file placement, score summaries and split summaries are shared.

## Language

- **Export context**: branch name, command name, destination, run metadata, primary artifact paths and writer details, built by `make_export_context`.

## Map

- `common/stats/utils.py`: shared helpers.
- `common/stats/builders.py`: one builder per branch.
- `common/stats/attach.py`: wiring for outputs.

## Contracts

Helpers (`utils.py`): `resolve_stats_path(dst)` (file target gives sibling `stats.yaml`, directory target gives `dst/stats.yaml`), `make_export_context(...)`, `emit_stats_yaml(document, ctx)`, `score_summary(scores)`, `base_stats_document(dataset, ctx)`.

Builders (`builders.py`): `build_classify_stats`, `build_label_stats`, `build_vlm_stats`, `build_gen_stats`.

Document base: `schema_version: 1`, `branch`, `command`, `output`, `dataset.records`, `dataset.splits`, then branch fields:

- classify: assignments by output label, unlabeled count, confidence summary, multi-class flags, thresholds.
- label: prompt counts and OCR text-box counts.
- vlm: unique images, QA counts, answered and unanswered, question-label counts, answer buckets, partitions and shards written.
- gen: mode, prompt count, fanout mode, image size, source summary, primary artifact paths.

Users: [classify output](../classify/output.md) (`to-class-dir`), [label output](../label/output.md) (`to-results` and any output with `--with-stats`), [vlm output](../vlm/output.md), [gen output](../gen/output.md).

## Constraints

- Stats MUST exist only on outputs that opt into the shared builder path.
- Empty or null-ish sections MUST be pruned before emission.
- An existing `stats.yaml` MUST be overwritten.
- Split order MUST put `train`, `val`, `test` first, then any other labels.
- `score_summary([])` MUST return `{"available": 0}`, never invented min, mean or max.
- Label names MUST prefer explicit annotation label text, then class-list lookup by `cls`, then the raw value.
- Builders MUST report branch semantics, not only raw field counts.
