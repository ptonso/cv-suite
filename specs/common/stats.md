# Stats

## 1. General Description
The shared stats subsystem provides optional `stats.yaml` emission for output commands that explicitly opt into it. The system is intentionally output-centric rather than pipeline-global: stats are built at writer time using the final `VisionDataset`, the destination path, and branch-specific run metadata. This keeps branch ownership intact while centralizing common document shape, file placement, score summaries, and split summaries.

## 2. Inputs and Outputs (I/O)
Inputs:
- Final `VisionDataset`.
- Export context built from branch name, command name, destination path, run metadata, primary artifact paths, and writer details.
- Optional `--with-stats` flags on supported outputs.

Outputs:
- `stats.yaml` adjacent to or inside the output destination.
- A normalized document with `schema_version`, branch metadata, output metadata, dataset counts, and branch-specific summary fields.

## 3. Interfaces
Shared helpers in `common/stats/utils.py`:
- `resolve_stats_path(dst)`
  - file target => sibling `stats.yaml`
  - directory target => `dst/stats.yaml`
- `make_export_context(...)`
- `emit_stats_yaml(document, ctx)`
- `score_summary(scores)`
- `base_stats_document(dataset, ctx)`

Branch-specific builders in `common/stats/builders.py`:
- `build_classify_stats(...)`
- `build_label_stats(...)`
- `build_vlm_stats(...)`
- `build_gen_stats(...)`

## 4. Business Decisions and Strict Policies
- Empty or null-ish sections are pruned before YAML emission.
- Existing `stats.yaml` is overwritten intentionally.
- Split ordering prefers `train`, `val`, `test` before any extra split labels.
- `score_summary([])` returns `{"available": 0}` rather than fake min/mean/max values.
- Label naming prefers explicit annotation label text, then class-list lookup by `cls`, then raw numeric/string fallback.
- Stats builders report branch semantics rather than only raw field counts. For example:
  - classify reports output assignment buckets and threshold application,
  - label reports prompt counts and OCR text-box counts,
  - VLM reports answered vs unanswered QA pairs and answer-bucket partitions,
  - gen reports prompt fanout mode and source-summary metadata.

## 5. Implementation Details
Document shape always starts from a shared base:
- `schema_version: 1`
- `branch`
- `command`
- `output`
- `dataset.records`
- `dataset.splits`

Writer-specific behaviors currently implemented:
- `[[classify/output]]` uses stats for `to-class-dir`.
- `[[label/output]]` uses stats for `to-results` and any other output that wires in `--with-stats`.
- `[[vlm/output]]` stats include shard counts or partition counts when relevant.
- `[[gen/output]]` stats include primary artifact paths and generation metadata.

Related docs:
- `[[classify/output]]`
- `[[label/output]]`
- `[[vlm/output]]`
- `[[gen/output]]`
