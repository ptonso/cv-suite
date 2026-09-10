# FM Providers

## 1. General Description
The provider layer defines what model families exist, how provider names resolve to modules, and what contract each provider subprocess must implement. Providers are split into local providers discovered from package structure and API providers registered explicitly in code. The runtime described in `[[common/fm_runtime]]` treats providers as subprocess modules; the provider layer describes how those modules are found and what they do once launched.

## 2. Inputs and Outputs (I/O)
Inputs:
- Provider names such as `clip`, `siglip2`, `sam3`, `qwen`, `flux`, `qwen_image_edit`, `openrouter`.
- Provider family context such as `classify`, `ground`, `vlm`, `create`, `edit`, `ocr`.
- `VisionDataset` JSON payloads plus `FMRequest`.

Outputs:
- `ProviderSpec` metadata used by branch help and runtime resolution.
- Provider subprocess modules that mutate a dataset and write `out.json`.
- Provider-generated metadata stored under `dataset.meta["fm"]` and record-level fields.

## 3. Interfaces
Registry layer in `common/fm/providers/registry.py`:
- local families: `classify`, `create`, `edit`, `ground`, `ocr`, `vlm`
- `iter_providers_for_family(family, include_descriptions=False)`
- `resolve_provider_spec(provider_name, family=...)`
- `resolve_provider_module(provider_name, family=...)`
- `provider_choices_for_family(family)`
- `default_model_id_for_provider(provider_name, family=...)`
- `allowed_model_ids_for_provider(provider_name, family=...)`
- `format_available_providers(...)`

Current local providers covered by the registry and tests include (the family directories under `common/fm/providers/<family>/` are authoritative; the lists below are representative):
- classify: `clip`, `deep_orientation`, `siglip2`
- create: `flux`, `qwen_image`, `sana`, `stable_diffusion`
- edit: `dim_edit`, `flux2_klein`, `instruct_pix2pix`, `ovis_u1_3b`, `qwen_image_edit`, `step1x_edit`
- ground: `gdino`, `gsam`, `llmdet`, `locate_anything`, `rex_omni`, `sam3`, `yolo_e`
- ocr: `paddleocr`
- vlm: `blip`, `cogvlm`, `internvl`, `llava`, `minicpm_v`, `paligemma`, `qwen`

Current API providers:
- `openrouter` for `create`
- `openrouter` for `edit`
- `openrouter` for `vlm`

Provider base contracts in `common/fm/providers/base.py` and `common/fm/providers/bases/*`:
- `BaseFMModel.build_cli_parser()` exposes `--input`, `--output`, and `--no-resume`
- `BaseFMModel.run_from_paths(ProcessArgs)` loads `VisionDataset`, validates `FMRequest`, restores checkpoints, processes jobs, updates FM metadata, and writes output JSON
- resumable job orchestration with `manifest.json`, `state.json`, `records.jsonl`, and `completed_jobs.jsonl`
- dataclass-based `model_args` validation
- prompt/config normalization
- provider-specific job builders and batch processors

Module metadata interfaces:
- local provider help metadata is read from source with `ast`, not by importing heavy model modules
- concrete provider modules may expose:
  - `DEFAULT_MODEL_ID`
  - `PARAMS`
  - `MODEL_ALIASES`
  - provider class attribute `description`

## 4. Business Decisions and Strict Policies
- Provider names are resolved per family. A provider can exist in one family and not another.
- Alias resolution is explicit. Example: `minicpm-v` resolves to `minicpm_v`.
- API providers are not auto-discovered from the filesystem; they are hard-coded `ProviderSpec` entries.
- Wrapper identity and backend checkpoint identity are intentionally different:
  - wrapper identity is `provider_name` / `model_name` and is what branch CLIs pass through `--model`
  - backend checkpoint/runtime identity is `model_id` and belongs inside `FMRequest.model_args`
- OpenRouter providers have strict allowed-model lists and restrictive runtime semantics compared with local providers.
- Unknown `--model-arg` keys are fatal when the provider options class is a dataclass. Error messages enumerate supported keys and the options class name.
- Prompt normalization differs by family but is always done provider-side before job execution.
- Providers are resumable by default and only ignore prior run state when `--no-resume` is forwarded.
- Help metadata is deterministic and offline. Source constants and aliases are parsed without loading the heavy provider runtime.

## 5. Implementation Details
Provider-side behaviors shared across bases:
- classification base
  - writes `record.classification`
  - can null out the winning label when thresholding fails while still preserving probabilities and score
  - appends `fm_tasks=["classify"]`
- ground base
  - reads grounding prompts from record attributes
  - appends boxes/polygons with `model`, `prompt`, and `group_id`
  - finalizes dataset task to `seg` if any polygons remain
- OCR base
  - ensures a `text` class exists
  - writes OCR boxes with `kind="ocr"` and attached text
- VLM base
  - answers existing unanswered QAs or creates prompt-derived QA jobs
  - prepends `<image>\n` when required by the model wrapper
  - manages auto-placement memory budgeting when `--device auto`
- generation bases
  - create jobs are built from prompt-expanded records
  - edit jobs prefer `gen_source_image` when present
  - generated images are materialized under provider work directories before the branch output writer moves them

Source-metadata extraction details:
- `_source_local_provider_info(...)` reads `DEFAULT_MODEL_ID`, `PARAMS`, aliases, and class `description` from the module source AST
- the registry skips packages and any module whose filename starts with `_`
- compatibility aliases such as `resolve_model_module(...)` and `format_available_models(...)` still exist during the rename from “models” terminology to “providers”

Related docs:
- `[[common/fm_runtime]]`
- `[[common/fm_utils]]`
- `[[common/fm_cache]]`
- `[[classify/transforms]]`
- `[[label/transforms]]`
- `[[vlm/transforms]]`
- `[[gen/transforms]]`
