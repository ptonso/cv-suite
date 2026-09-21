# FM Providers

Related: **Read first** [fm_runtime](fm_runtime.md). **See also** [fm_utils](fm_utils.md), [fm_cache](fm_cache.md), [records_dataset](records_dataset.md).

## Overview

A provider is the wrapper around one foundation model. This spec covers which providers exist, how a name resolves to a module, and what a provider process must do once the [runtime](fm_runtime.md) launches it.

Providers come in two kinds. Local providers are discovered from the package structure and run in a managed virtual environment. API providers are registered by hand in code and run in the current interpreter. Providers are grouped into families by what they do: `classify`, `create`, `edit`, `ground`, `ocr`, `vlm`.

A provider process loads a dataset JSON, validates the `FMRequest`, processes the records in resumable jobs, and writes an output JSON. Branch transformations ([classify](../classify/transforms.md), [label](../label/transforms.md), [vlm](../vlm/transforms.md), [gen](../gen/transforms.md)) build the request; providers never decide output layout.

## Language

- **Provider name**: what branches pass through `--model`, for example `clip`, `qwen`, `sam3`, `openrouter`.
- **`model_id`**: the backend checkpoint identity, inside `FMRequest.model_args`. Different from the provider name.
- **Provider spec** (`ProviderSpec`): metadata used by help and resolution.
- **Base**: a family's shared provider class (`BaseFMModel` and `common/fm/providers/bases/*`).
- **Managed auto placement**: the VLM path where `--device auto` builds a Hugging Face device map and memory budget.
- **`fm_tasks`**: record attribute appended by the bases to list which model tasks touched the record.

## Map

- `common/fm/providers/registry.py`: name resolution and help metadata.
- `common/fm/providers/base.py`, `bases/*`: provider base classes.
- `common/fm/providers/<family>/<provider>.py`: one module per local provider. The family directories are authoritative for what exists.
- `common/fm/providers/setup_venv/`: environment setup scripts ([fm_cache](fm_cache.md)).
- `common/fm/providers/_template.py`: template for a new provider.

Providers at time of writing (representative):

- classify: `clip`, `deep_orientation`, `siglip2`
- create: `flux`, `qwen_image`, `sana`, `stable_diffusion`
- edit: `dim_edit`, `flux2_klein`, `instruct_pix2pix`, `ovis_u1_3b`, `qwen_image_edit`, `step1x_edit`
- ground: `gdino`, `gsam`, `llmdet`, `locate_anything`, `rex_omni`, `sam3`, `yolo_e`
- ocr: `paddleocr`
- vlm: `blip`, `cogvlm`, `internvl`, `llava`, `minicpm_v`, `paligemma`, `qwen`
- API: `openrouter` for `create`, `edit` and `vlm`

## Contracts

### Registry (`registry.py`)

`iter_providers_for_family(family, include_descriptions=False)`, `resolve_provider_spec(provider_name, family=...)`, `resolve_provider_module(provider_name, family=...)`, `provider_choices_for_family(family)`, `default_model_id_for_provider(provider_name, family=...)`, `allowed_model_ids_for_provider(provider_name, family=...)`, `format_available_providers(...)`.

Module metadata a provider module may expose: `DEFAULT_MODEL_ID`, `PARAMS`, `MODEL_ALIASES`, and a class attribute `description`. Help metadata is read from module source with `ast`, never by importing heavy model modules. The registry skips packages and any module whose filename starts with `_`.

### Provider process

- `BaseFMModel.build_cli_parser()` exposes `--input`, `--output`, `--no-resume`.
- `BaseFMModel.run_from_paths(ProcessArgs)` loads the dataset, validates `FMRequest`, restores checkpoints, processes jobs, updates model metadata, and writes the output JSON. Metadata lands in `dataset.meta["fm"]` and on records.
- Resumable state uses `manifest.json`, `state.json`, `records.jsonl`, `completed_jobs.jsonl`.
- `model_args` are validated against a dataclass options class.

### Family behavior

- classify: writes `record.classification`; may null the winning label when thresholding fails while keeping probabilities and score; appends `fm_tasks=["classify"]`.
- ground: reads prompts from record attributes (`ground_prompts`); appends boxes and polygons with `model`, `prompt`, `group_id`; sets the dataset task to `seg` if polygons remain.
- ocr: ensures a `text` class; writes boxes with `kind="ocr"` and text.
- vlm: answers existing unanswered questions or creates prompt-derived ones; prepends `<image>\n` when the wrapper needs it; manages memory budgeting when `--device auto`.
- create / edit: create jobs come from prompt-expanded records; edit jobs prefer `gen_source_image` when present; generated images are written under provider work directories before the branch output moves them.

## Constraints

- Provider names MUST be resolved per family; a provider may exist in one family and not another.
- Aliases MUST be explicit (`minicpm-v` resolves to `minicpm_v`).
- API providers MUST be hard-coded `ProviderSpec` entries and MUST NOT be discovered from the filesystem.
- Wrapper identity (`provider_name` / `model_name`, passed as `--model`) MUST stay separate from checkpoint identity (`model_id`, inside `FMRequest.model_args`).
- OpenRouter providers MUST use strict allowed-model lists and MUST reject device, precision, config and `--max-gpu-memory` options that local providers accept.
- An unknown `--model-arg` key MUST be fatal when the options class is a dataclass, and the message MUST list the supported keys and the options class name.
- Prompt normalization MUST happen provider-side, before job execution, and differs per family.
- Providers MUST be resumable by default and ignore prior state only when `--no-resume` is forwarded.
- Help metadata MUST be deterministic and offline.
- Tests: `common/tests/test_base_model.py`, `test_fm_model_bases.py`, `test_gen_model.py`, `test_grounding_wrappers.py`.

## Extending

To add a provider: copy `common/fm/providers/_template.py` into the right family directory, set `DEFAULT_MODEL_ID`, `PARAMS`, `MODEL_ALIASES` and `description`, subclass the family base, and add `setup_venv/<provider>.sh` for a local provider. Reuse the helpers in [fm_utils](fm_utils.md) for paths, precision and Hugging Face sources. An API provider is added as a `ProviderSpec` entry in the registry.

## History

- [Isolated provider environments](../logs/2026-09-20_isolated-provider-environments.md): why local providers are separate processes with their own environments.
