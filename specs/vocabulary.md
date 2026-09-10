# Vocabulary

## 1. General Description
This document defines the ubiquitous language used across branches. Many names are branch-specific, but several shared terms are foundational to understanding the system described in `[[overview]]`, `[[architecture]]`, and `[[common/records_dataset]]`.

## 2. Inputs and Outputs (I/O)
The “inputs” to this vocabulary are code-level identifiers, CLI terms, metadata keys, and schema names. The “outputs” are normalized domain definitions that the rest of the spec reuses.

## 3. Interfaces
Core shared terms:
- `VisionDataset`: the canonical top-level in-memory dataset object (`cvsuite.common.core`).
- `VisionRecord`: one image-centric sample inside a dataset.
- `ImageInfo`: image path and image dimensions for a record.
- `Record`, `ImageRecord`: aliases for `VisionRecord` and `ImageInfo`.
- `FMRequest`: the runtime request attached to a dataset before FM execution (`cvsuite.common.core`).
- `Task`: enum over canonical task ids such as `det`, `inst-seg`, `pose`, `cls`, and `multi-cls`.
- `Split`: logical dataset split such as `train`, `val`, or `test`; some VLM paths also preserve richer values such as `train2014`.

Branch vocabulary:
- `ingest`: convert a source filesystem layout into the shared dataset contract.
- `transform`: mutate a dataset before an output writer runs.
- `output command`: materialize a dataset into a branch-owned on-disk form.
- `unstructured`: image-first folder ingest that infers labels or splits heuristically from directory structure.
- `multi-class ingest`: classification ingest mode where one image basename may appear under multiple class folders and therefore represents multiple labels.
- `flat`: canonical image-only dataset format used behind cvsuite spellings such as `images` or label-branch image-only ingest.
- `flat JSON`: cvsuite VLM spelling for canonical format `flat_vlm_json`.
- `VQA-style`: cvsuite VLM spelling for canonical format `vqa_style`.
- `shards`: cvsuite VLM spelling for canonical format `shards_vlm`.

FM vocabulary:
- `provider`: the model wrapper selected by the branch, such as `clip`, `qwen`, `sam3`, or `openrouter`.
- `family`: one of the FM provider families: `classify`, `create`, `edit`, `ground`, `ocr`, `vlm`.
- `local provider`: provider executed in a managed local virtualenv.
- `API provider`: provider executed with the current interpreter and remote API access.
- `managed auto device placement`: the VLM path where `--device auto` creates Hugging Face device-map and memory-budget settings automatically.
- `stage dir`: transient working directory under a provider cache used for downloads, offload folders, or temporary files.

## 4. Business Decisions and Strict Policies
- `class` is the public branch name; `classify` is internal only. `cvsuite classify` is rejected as an unknown branch.
- Branch names use hyphenated public CLI names even when internal package names contain underscores.
- Specs plus code/tests are the documentation source of truth. Branch-local nested READMEs are intentionally being removed.
- The dataset contract is owned once, in `cvsuite.common.core`; `Record` / `ImageRecord` are aliases, not a second schema.
- “Prompt” is overloaded intentionally:
  - classification prompt means class vocabulary or class-to-prompts mapping
  - grounding prompt means grounding phrases or label-to-prompts mapping
  - VLM prompt means questions or question specs
  - generation prompt means text prompt(s) for create or edit
- `with-stats` only exists on outputs that explicitly opt into the shared stats builder path.

## 5. Implementation Details
Important metadata keys and attributes that recur throughout the code:
- `_cvsuite_fm_request`: dataset meta key used to store the runtime `FMRequest`
- `classify_ingest_mode`: dataset meta key used to preserve how class ingest occurred
- `classify_multi_class_labels`: record attribute storing labels assigned during cvsuite multi-class ingest
- `class_dir_multi_labels`: the class-dir multi-label attribute (`cvsuite.common.io.common.CLASS_DIR_MULTI_LABELS_ATTR`) used by cvsuite outputs
- `ground_prompts`: temporary record attribute used by grounding providers
- `gen_mode`: dataset meta key distinguishing `create` vs `edit`
- `gen_prompt`, `gen_prompt_index`, `gen_sample_index`, `gen_output_key`: generation record attributes
- `gen_source_image`, `gen_source_key`, `gen_source_record_idx`: edit-generation provenance attributes
- `vlm_sample_key`, `vlm_image_id`, `vlm_predominant_label`, `vlm_answer_bucket`: VLM-specific record attributes
- `fm_tasks`: record attribute appended by provider bases to show which FM tasks touched a record

Related documents:
- `[[dataclasses]]`
- `[[common/records_dataset]]`
- `[[common/fm_provider]]`
- `[[common/fm_runtime]]`
