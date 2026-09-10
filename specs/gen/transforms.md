# Gen Transforms

## 1. General Description
Generation transforms are the action stage of `cvsuite gen`. `create` constructs placeholder records from prompt expansions. `edit` ingests source images or datasets and then expands them cartesianly across prompts and sample counts. Both actions then delegate inference to the shared FM runtime with task `gen`, while dataset metadata determines whether the provider family is `create` or `edit`.

## 2. Inputs and Outputs (I/O)
`create` inputs:
- repeatable `--prompt` required
- `--num-images >= 1`
- runtime args from shared generation runtime:
  - `--provider/--model`
  - `--model-id`
  - `--width`
  - `--height`
  - `--device`
  - `--batch`
  - `--max-gpu-memory`
  - `--precision`
  - `--config`
  - `--no-resume`
  - repeatable `--model-arg key=value`

`edit` inputs:
- positional `src`
- `--from {auto,yolo,labelme,coco,images,unstructured}`
- repeatable `--prompt` required
- `--num-images >= 1`
- same runtime args as `create`

Outputs:
- `VisionDataset` with generated-result placeholders or FM-mutated outputs
- `dataset.meta["gen_mode"]` set to `create` or `edit`
- prompt-expansion metadata and source-summary metadata

## 3. Interfaces
Prompt-expansion interfaces in `gen/core/batch.py`:
- prompt file extensions recognized: `.yaml`, `.yml`, `.json`
- supported prompt file payloads:
  - list of strings,
  - mapping `id -> prompt`,
  - list of single-entry mappings

Create dataset interface:
- one record per prompt x sample index
- initial image path set to `pending/<output_key>.png`
- record attributes:
  - `gen_prompt`
  - `gen_prompt_index`
  - `gen_sample_index`
  - `gen_output_key`

Edit dataset interface:
- ingest edit source through `gen.core.router`, which delegates structured cases to label ingest
- expand source records x prompts x sample index
- preserve source annotations and copy source record fields
- add attributes:
  - `gen_source_record_idx`
  - `gen_source_image`
  - `gen_source_key`
  - prompt/output-key fields

## 4. Business Decisions and Strict Policies
- Prompt strings from repeated `--prompt` flags preserve order and duplicates.
- Prompt IDs coming from files must be unique after filename sanitization; collisions are fatal.
- Non-YAML/JSON existing paths are treated as literal prompt strings rather than prompt files.
- `--width` and `--height` override any same-name values passed through `--model-arg`.
- Width and height, when provided, must be positive integers.
- OpenRouter restrictions are stricter than local providers:
  - non-default device/precision choices are rejected,
  - config and max-gpu-memory are rejected,
  - model IDs must come from the provider’s allowed-model list.
- Runtime error rewriting is user-facing:
  - gated Hugging Face access issues are rewritten with actionable guidance,
  - size-related generation errors can be wrapped in provider-specific size-rejection messages.

Fanout rules:
- `create` uses `gen_fanout_mode="prompt-list"`
- `edit` uses `gen_fanout_mode="cartesian"`

## 5. Implementation Details
Generation request construction:
- branches attach `FMRequest(task="gen", provider=<provider>, prompt=<json payload>, ...)`
- prompt payload is JSON-encoded list of prompt strings
- `model_id`, `width`, and `height` are forwarded through `model_args`

Create metadata:
- `gen_prompt_count`
- `gen_num_images_per_prompt`
- `gen_fanout_mode`

Edit metadata additionally includes:
- source format
- source record count
- source annotation totals for boxes, polygons, and keypoints

Related docs:
- `[[gen/cli]]`
- `[[gen/output]]`
- `[[common/fm_provider]]`
- `[[common/fm_runtime]]`
