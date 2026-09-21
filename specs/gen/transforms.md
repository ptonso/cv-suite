# Gen Transforms

Related: **Read first** [gen cli](cli.md). **See also** [gen output](output.md), [fm_provider](../common/fm_provider.md), [fm_runtime](../common/fm_runtime.md).

## Overview

Generation actions are the first stage of `cvsuite gen`. `create` builds placeholder records from prompts. `edit` reads source images or datasets and expands every source record across all prompts and sample counts. Both then call the shared model runtime with task `gen`; the dataset metadata decides whether the provider family is `create` or `edit`.

## Language

- **Fanout mode**: how prompts multiply into records. `create` uses `prompt-list` (one record per prompt and sample index); `edit` uses `cartesian` (source record by prompt by sample index).
- **Output key** (`gen_output_key`): stable key naming a generated image.

## Map

- `gen/transform/commands/create.py`, `edit.py`.
- `gen/core/batch.py`: prompt expansion. `gen/core/router.py`: edit-source ingest, which delegates structured cases to label ingest ([label adapters](../label/adapters.md)).

## Contracts

### create

Arguments: repeatable `--prompt` (required), `--num-images >= 1`, and the runtime arguments below.

### edit

Arguments: positional `src`; `--from {auto,yolo,labelme,coco,images,unstructured}`; repeatable `--prompt` (required); `--num-images >= 1`; the same runtime arguments.

### Runtime arguments

`--provider/--model`, `--model-id`, `--width`, `--height`, `--device`, `--batch`, `--max-gpu-memory`, `--precision`, `--config`, `--no-resume`, repeatable `--model-arg key=value`.

### Prompt files

Recognized extensions `.yaml`, `.yml`, `.json`. Payloads: a list of strings, a mapping `id -> prompt`, or a list of single-entry mappings.

### Datasets produced

Both set `dataset.meta["gen_mode"]` to `create` or `edit`, plus `gen_prompt_count`, `gen_num_images_per_prompt`, `gen_fanout_mode`.

- create: one record per prompt and sample index, with image path `pending/<output_key>.png` and attributes `gen_prompt`, `gen_prompt_index`, `gen_sample_index`, `gen_output_key`.
- edit: source records expanded by prompt and sample index, keeping source annotations and record fields, with attributes `gen_source_record_idx`, `gen_source_image`, `gen_source_key` plus the prompt and output-key fields. Metadata adds source format, source record count and source annotation totals for boxes, polygons and keypoints.

### Request

`FMRequest(task="gen", provider=<provider>, prompt=<JSON-encoded list of prompts>, ...)`. `model_id`, `width`, `height` go through `model_args`.

## Constraints

- Prompts from repeated `--prompt` flags MUST keep order and duplicates.
- Prompt ids from files MUST be unique after filename sanitization; a collision MUST be fatal.
- An existing path that is not YAML or JSON MUST be treated as a literal prompt string.
- `--width` and `--height` MUST override same-name `--model-arg` values and MUST be positive integers.
- OpenRouter runs MUST reject non-default device or precision, `--config` and `--max-gpu-memory`, and MUST accept only model ids from the provider's allowed list ([fm_provider](../common/fm_provider.md)).
- Gated Hugging Face access errors MUST be rewritten with actionable guidance; size-related generation errors MAY be wrapped in a provider-specific size-rejection message.
