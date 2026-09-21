# VLM Adapters

Related: **Read first** [vlm cli](cli.md). **See also** [io](../common/io.md), [records_dataset](../common/records_dataset.md), [vlm transforms](transforms.md), [vlm output](output.md).

## Overview

The VLM adapter layer decides how VLM source formats are detected and turns them into the shared dataset plus `record.vqas`. Parsing and writing live in [io](../common/io.md); the branch owns public spellings, detection precedence ([cli](cli.md#format-detection-precedence)), prompt and question helpers, temporary-resource cleanup and answer partitioning.

Every format follows one rule: each logical sample becomes one record, and each question-answer item becomes one `VQA` on `record.vqas`.

## Language

- **VQA**: one question-answer item. Ground-truth answers sit in `VQA.meta["ground_truth"]` and predicted answers in `VQA.answer`.
- **Prompt spec**: what `ask` accepts as questions.

## Map

- `vlm/core/router.py`, `vlm/core/io.py`, `vlm/core/adapters/`.

## Contracts

Inputs: image-only trees; flat image-plus-JSON directories; TAR shards of paired image and JSON members; normalized VQA-style manifests; raw VQA question, annotation and prediction JSON corpora.

Outputs: shared datasets with `record.vqas`; record attributes `vlm_sample_key`, `vlm_image_id`, `vlm_predominant_label`, item metadata; cleanup metadata for temporary extraction directories.

Interfaces: `vlm.core.router.ingest(...)`, `vlm.core.io.qa_from_payload(...)`, `qa_to_payload(...)`, `load_prompt_spec(...)`, `cleanup_dataset_resources(...)`.

Canonical formats delegated to: `flat`, `flat_vlm_json`, `shards_vlm`, `vqa_style`.

Format specifics:

- Flat JSON: matches sibling image and `.json` by stem; accepts `qa_pairs` or `qas`; keeps split strings exactly as written; stores promoted fields `image_id`, `predominant_label`, `original_filename`.
- Shards: reads `.tar` members and requires both `<sample>.jpg|png...` and `<sample>.json`; extracts to a temporary root and makes `record.image.path` relative to it; records shard provenance in item metadata; keeps richer split strings.
- VQA-style manifests: require `items` plus `images_root`; dataset-level manifest metadata survives into record attributes and QA metadata.
- VQA-style raw corpora: join question, annotation and prediction JSONs by `question_id` and `image_id`; find images by trailing numeric ids in filenames; keep `data_subtype`-style splits such as `train2014`; keep question-side metadata, expected answers and predicted answers, models and scores.
- Prompt specs used by `ask`: raw strings, JSON/YAML files, JSON strings, top-level lists, or label-to-questions mappings.

## Constraints

- All formats MUST normalize into `record.vqas`.
- Flat JSON and shards MUST require complete image and JSON pairs and MUST fail on ambiguous or incomplete pairs.
- Shard ingest MUST extract to a temporary directory and record it in dataset metadata for cleanup.
- Normalized manifests and raw VQA question files MUST NOT coexist in one ingest. Ambiguous image-id matches MUST fail instead of guessing.
- Older source spellings MUST be kept inside QA metadata where they are user-visible.
