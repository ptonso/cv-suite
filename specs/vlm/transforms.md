# VLM Transforms

## 1. General Description
The VLM branch exposes four transforms with distinct purposes:
- `caption`: seed an image-only dataset with a default caption question
- `ask`: seed one or more unanswered questions from a prompt spec
- `vqa`: run a VLM over already-seeded unanswered questions
- `map-answers`: partition answered QA records into canonical answer buckets for downstream export

These transforms are responsible for question seeding, provider runtime invocation, and answer normalization.

## 2. Inputs and Outputs (I/O)
`caption`:
- provider/runtime args
- no explicit prompt argument; uses the built-in caption prompt

`ask`:
- `--prompt` required
- provider/runtime args

`vqa`:
- provider/runtime args

`map-answers`:
- positional canonical labels
- optional `--labels-file`

Shared runtime args for FM-backed transforms:
- `--provider/--model`
- `--model-id`
- `--device`
- `--batch`
- `--max-gpu-memory`
- `--precision`
- `--config`
- `--weights`
- `--no-resume`
- repeatable `--model-arg key=value`

Outputs:
- updated `record.vqas`
- dataset answer partitions for `map-answers`
- FM metadata under `dataset.meta["fm"]`

## 3. Interfaces
Question-seeding interfaces:
- `caption` seeds the constant prompt `Describe this image in one concise sentence.`
- `ask` loads prompt specs from:
  - raw string
  - existing YAML/JSON file path
  - JSON string
  - list
  - label-to-questions mapping
- `vqa` only runs inference against pending unanswered QA items

Answer-mapping interfaces:
- `load_label_registry(labels, labels_file=...)`
- `match_answer_text(answer, registry)`
- `apply_answer_mapping(dataset, registry)`

Record attribute and metadata interfaces:
- `vlm_sample_key`
- `vlm_image_id`
- `vlm_predominant_label`
- `vlm_answer_bucket`
- dataset meta `vlm_answer_buckets`

## 4. Business Decisions and Strict Policies
- `caption` requires an image-only dataset with no existing questions.
- `ask` appends only missing unanswered questions rather than duplicating existing QA pairs.
- `has_pending_vqas()` is true only when a question lacks both answer and model.
- `--max-gpu-memory` requires `--device auto`; other device modes reject it.
- Provider aliases and model namespaces are normalized:
  - `minicpm-v` -> `minicpm_v`
  - bare `Qwen2.5-VL-*` and `Qwen3-VL-*` ids are rewritten to `Qwen/...`
- Answer mapping always includes fallback label `unclear` and forces it to the end of the registry.
- Label registry rules:
  - labels may come from CLI and/or labels-file
  - labels-file may be a list or a label-to-synonyms mapping
  - labels must be safe relative path segments
  - normalized synonym collisions are fatal

Answer-matching order is strict:
1. full answer
2. first/last non-empty line
3. lead-in spans after phrases like `answer:` or `final answer:`
4. trimmed spans after stripping rationale tails such as `because` and `since`
5. symbolic token matching such as `A`, `(B)`, `option C`
6. fallback to `unclear`

Ambiguity handling:
- hedged or multi-label answers such as `yes/no` or `A or B` map to `unclear`
- ambiguous matches are marked with rule suffixes such as `:ambiguous`

## 5. Implementation Details
FM-backed VLM execution:
- transform runtime builds `FMRequest(task="vlm", provider=<normalized provider>, prompt=<prompt payload>)`
- provider-side bases update existing QA entries or append new ones
- VLM FM metadata includes managed auto-placement information when relevant

`map-answers` dataset rewrite:
- expands each record into one record per QA pair
- rewrites sample keys with question-specific suffixes
- stores mapped label, matching rule, and alias in QA metadata
- writes `vlm_answer_bucket` on each output record
- output commands that support partitioning materialize one destination subtree per answer bucket; the `flat_vlm_json` reader resolves each sidecar's image within its own bucket dir, so the partitioned tree re-ingests cleanly

Related docs:
- `[[vlm/cli]]`
- `[[vlm/output]]`
- `[[common/fm_provider]]`
- `[[common/fm_runtime]]`
