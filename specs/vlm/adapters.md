# VLM Adapters

## 1. General Description
The VLM adapter layer owns how VLM-oriented source formats are detected on disk and normalized into the shared dataset contract plus `record.vqas`. Active canonical parsing and writing now lives in `cvsuite.common.io`; the VLM branch keeps ownership of:
- public CLI spellings such as `json`, `shards`, and `vqa-style`
- format-detection precedence
- prompt/question helpers
- temp-resource cleanup
- answer partitioning semantics

All current VLM ingest paths converge on the same internal rule: each logical sample becomes one record, and each question-answer item becomes one `VQA` attached to `record.vqas`.

## 2. Inputs and Outputs (I/O)
Input formats handled today:
- image-only trees
- flat image-plus-JSON sidecar directories
- TAR shard archives containing paired image and JSON members
- normalized VQA-style manifests
- raw VQA-style question/annotation/prediction JSON corpora

Outputs produced by the active adapter layer:
- shared datasets with normalized `record.vqas`
- record attributes such as `vlm_sample_key`, `vlm_image_id`, `vlm_predominant_label`, and item metadata
- cleanup metadata for temp extraction directories

## 3. Interfaces
Router and shared IO interfaces:
- `vlm.core.router.ingest(...)`
- `vlm.core.io.qa_from_payload(...)`
- `vlm.core.io.qa_to_payload(...)`
- `vlm.core.io.load_prompt_spec(...)`
- `vlm.core.io.cleanup_dataset_resources(...)`

Canonical delegated formats:
- `flat`
- `flat_vlm_json`
- `shards_vlm`
- `vqa_style`

## 4. Business Decisions and Strict Policies
- All formats normalize into `record.vqas`.
- Ground-truth answers stay in `VQA.meta["ground_truth"]`.
- Predicted answers stay in `VQA.answer`.
- Flat JSON and shard formats require complete image/JSON pairing and fail on ambiguous or incomplete pairs.
- Shard ingest extracts to a temp directory and records that temp root in dataset metadata for mandatory cleanup.
- VQA-style ingest is intentionally strict about ambiguity:
  - normalized manifests and raw VQA question files may not coexist for one ingest
  - ambiguous image-id matches fail instead of guessing
- Prompt specs used by `ask` may be raw strings, JSON/YAML files, JSON strings, top-level lists, or label-to-questions mappings.
- cvsuite preserves legacy metadata source spellings such as `json`, `shards`, and `vqa-style` inside QA metadata when those spellings are part of user-visible behavior.

## 5. Implementation Details
Adapter-specific behaviors:
- flat JSON
  - matches sibling image and `.json` files by stem
  - accepts `qa_pairs` or legacy `qas`
  - preserves split strings exactly as written, rather than canonicalizing them down to `train/val/test`
  - stores promoted fields such as `image_id`, `predominant_label`, and `original_filename`
- shards
  - reads `.tar` members and requires both `<sample>.jpg|png...` and `<sample>.json`
  - writes extracted images to a temp root and makes `record.image.path` relative to that temp dataset root
  - records shard provenance in item metadata
  - preserves richer split strings on read
- VQA-style normalized manifests
  - require `items` plus an `images_root`
  - allow dataset-level manifest metadata to survive into record attributes and QA metadata
- VQA-style raw corpora
  - join question JSONs, annotation JSONs, and prediction JSONs by `question_id` and `image_id`
  - discover images by trailing numeric image ids in filenames
  - preserve `data_subtype`-style split strings such as `train2014`
  - preserve question-side metadata, annotation-side expected answers, and prediction-side answers/models/scores

Cross-links:
- `[[vlm/cli]]`
- `[[vlm/transforms]]`
- `[[vlm/output]]`
- `[[common/io]]`
- `[[common/records_dataset]]`
