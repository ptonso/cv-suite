# VLM Transforms

Related: **Read first** [vlm cli](cli.md). **See also** [vlm adapters](adapters.md), [vlm output](output.md), [fm_provider](../common/fm_provider.md), [fm_runtime](../common/fm_runtime.md).

## Overview

Four transforms, each with one job:

- `caption` seeds an image-only dataset with a default caption question.
- `ask` seeds one or more unanswered questions from a prompt spec.
- `vqa` runs a model over the pending unanswered questions.
- `map-answers` sorts answered questions into fixed answer buckets for export.

Example: `cvsuite vlm ./photos ask --prompt "Is there a dog?" to-json ./out`, then `vqa` fills answers, then `map-answers yes no` partitions them.

## Language

- **Pending VQA**: a question with neither answer nor model (`has_pending_vqas()`).
- **Answer bucket**: a canonical label an answer maps to; `unclear` is always present and last.
- **Label registry**: the labels and their synonyms.

## Map

- `vlm/transform/commands/`: `caption.py`, `ask.py`, `vqa.py`, `map_answers.py`.
- `vlm/transform/core/`: question seeding and answer mapping.

## Contracts

### Arguments

- `caption`: runtime args; uses the built-in prompt `Describe this image in one concise sentence.`
- `ask`: `--prompt` (required), runtime args.
- `vqa`: runtime args.
- `map-answers`: positional canonical labels, optional `--labels-file`.
- Runtime args for model-backed transforms: `--provider/--model`, `--model-id`, `--device`, `--batch`, `--max-gpu-memory`, `--precision`, `--config`, `--weights`, `--no-resume`, repeatable `--model-arg key=value`.

### Outputs

Updated `record.vqas`; answer partitions for `map-answers`; model metadata under `dataset.meta["fm"]`. Attributes and metadata: `vlm_sample_key`, `vlm_image_id`, `vlm_predominant_label`, `vlm_answer_bucket`, dataset meta `vlm_answer_buckets`.

### Model runs

Builds `FMRequest(task="vlm", provider=<normalized provider>, prompt=<prompt payload>)`. The provider updates existing QA entries or appends new ones. Metadata includes managed auto-placement information when relevant.

### Answer mapping

Interfaces: `load_label_registry(labels, labels_file=...)`, `match_answer_text(answer, registry)`, `apply_answer_mapping(dataset, registry)`.

Matching order:

1. full answer
2. first or last non-empty line
3. lead-in spans after phrases such as `answer:` or `final answer:`
4. spans trimmed of rationale tails such as `because` and `since`
5. symbolic tokens such as `A`, `(B)`, `option C`
6. fallback to `unclear`

Hedged or multi-label answers such as `yes/no` or `A or B` map to `unclear`. Ambiguous matches carry rule suffixes such as `:ambiguous`.

Dataset rewrite: each record expands into one record per QA pair, sample keys get question-specific suffixes, the mapped label, matching rule and alias go into QA metadata, and `vlm_answer_bucket` is written on each record. Outputs that support partitioning write one subtree per bucket; the `flat_vlm_json` reader resolves each sidecar's image inside its own bucket directory, so a partitioned tree re-ingests.

## Constraints

- `caption` MUST require an image-only dataset with no existing questions.
- `ask` MUST append only missing unanswered questions and MUST NOT duplicate existing QA pairs.
- `--max-gpu-memory` MUST require `--device auto`; other device modes MUST reject it.
- Provider aliases and namespaces MUST be normalized: `minicpm-v` to `minicpm_v`; bare `Qwen2.5-VL-*` and `Qwen3-VL-*` ids to `Qwen/...`.
- The label registry MUST always include `unclear` and force it to the end. Labels come from the CLI and/or the labels file (a list or a label-to-synonyms mapping). Labels MUST be safe relative path segments. Normalized synonym collisions MUST be fatal.
