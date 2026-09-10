# Overview

## 1. General Description
`cvsuite` is a CLI-first Python package exposed as `cvsuite` through the console entry point defined in `pyproject.toml`. The system exists to perform reproducible filesystem-to-filesystem computer-vision workflows where users generally provide a source location, optionally run one branch-specific transform, and then materialize a destination layout. The codebase is organized around branch CLIs rather than one monolithic pipeline: `[[cli]]` dispatches into `class`, `label`, `vlm`, `gen`, `prep`, and `common`, and each branch owns its own ingest heuristics, transform semantics, and export layout rules.

The central architectural decision is that no on-disk dataset format is treated as universal. Branches normalize inputs into one shared in-memory dataset contract when that abstraction is useful. That contract, its format adapters, and the FM runtime dataclasses all live in `cvsuite.common` (`core`, `io`, `fm`). `label`, `classify`, `vlm`, and `gen` depend heavily on that contract. `prep` stays intentionally file-first and only constructs temporary datasets for localized FM-backed steps such as orientation scoring. Foundation-model execution is deliberately isolated in the shared FM runtime described in `[[common/fm_runtime]]`, while output layout policy remains branch-owned.

These specs are the technical source of truth for the current system: behavior is documented here once, in one place, with code and tests as the implementation reference. Pedagogical material — install steps, the command model, worked examples — lives in [`../docs/`](../docs/), not here. Branch-level nested `README.md` files are intentionally being removed.

## 2. Inputs and Outputs (I/O)
Inputs handled by the system today include:
- raw image files and folders
- class-organized image trees for `cvsuite class`
- annotated datasets in YOLO, LabelMe, COCO, and image-only/unstructured forms for `cvsuite label`
- VLM-oriented flat JSON, shards, VQA-style manifests, raw VQA corpora, and raw image sets for `cvsuite vlm`
- prompt strings, prompt files, and prompt mappings for generation, grounding, classification, and VLM prompting
- FM provider identifiers, runtime hints, and model-specific `--model-arg key=value` overrides

Outputs produced by the system today include:
- class-directory exports for classification datasets
- YOLO, LabelMe, COCO, audit YAML, preview windows, and annotated results trees for labeling workflows
- flat JSON pairs, sharded TAR datasets, and VQA-style manifests for VLM workflows
- final generated image files or directories for generation workflows
- rearranged, processed, sampled, or orientation-normalized raw-image folders for prep workflows
- optional `stats.yaml` sidecars for branches and outputs that explicitly support `--with-stats`; see `[[common/stats]]`

## 3. Interfaces
The main public interface is the top-level `cvsuite` CLI:
- `cvsuite <branch> [args]`
- `cvsuite class ...`
- `cvsuite label ...`
- `cvsuite vlm ...`
- `cvsuite gen ...`
- `cvsuite prep ...`
- `cvsuite common ...`

The main internal interfaces are:
- `[[common/records_dataset]]` and `[[dataclasses]]` for the shared dataset contract
- `[[common/io]]` for the dataset format router and adapters
- `[[common/fm_runtime]]` for subprocess-based FM execution
- `[[common/fm_provider]]` for provider registration, provider families, and provider-side contracts
- `[[common/fm_utils]]` for shared path, precision, prompt, cache, and geometry helpers used inside providers
- branch routers and adapters that detect source formats before transforms run

## 4. Business Decisions and Strict Policies
- The CLI is intentionally branch-oriented. There is no universal “convert anything to anything” command.
- Branches own source detection, output layout, and public CLI spellings. `common` does not choose folder layouts or dataset schemas.
- The shared dataset contract is owned once, in `cvsuite.common.core` (`VisionDataset`, `VisionRecord`, `ImageInfo`, geometry types, and the runtime dataclasses `FMRequest` / `Embedding` / `LabelMeShape`). `Record` and `ImageRecord` are aliases. Format adapters live in `cvsuite.common.io`.
- The system prefers deterministic behavior. Seeds default to fixed values in sampling and split assignment paths, and many outputs use collision suffixing rather than overwrite-by-default.
- Generation, VLM, OCR, grounding, and classification FM tasks all use the same shared runtime pattern: branch builds a shared dataset, populates `dataset.fm_request`, shared runtime launches a provider subprocess, provider mutates records, branch finalizes outputs.
- `meta` is not the place for the primary user request. Provider name, prompt, config path, batch size, device, precision, checkpoint overrides, and related runtime inputs belong in `dataset.fm_request`.
- Historical packages that are not part of the active CLI architecture, such as `src/nn/vision`, are intentionally not treated as source of behavior for `cvsuite`.
- Behavioral guardrails are explicit. Examples include:
  - `cvsuite class sample` refuses annotated datasets and active multi-class ingest datasets
  - `cvsuite prep sample` refuses annotated datasets and nested directory trees
  - `cvsuite label sample --hardlink` may only be paired with outputs that expose a `hardlink` argument
  - `sam3` grounding rejects CPU execution
  - `--max-gpu-memory` is only valid for managed auto-placement VLM paths

## 5. Implementation Details
Repository-level structure is described in `[[architecture]]`, but the high-level flow is:
1. `cvsuite` dispatches into a branch CLI.
2. The branch CLI discovers commands from package structure or explicit registries.
3. The branch ingests filesystem inputs into the shared dataset contract via `[[common/io]]`, or performs a file-first prep workflow.
4. The branch optionally applies one transform or action stage.
5. The branch writes a destination layout and may emit `stats.yaml`.

The concrete branch responsibilities are:
- `[[classify/cli]]`, `[[classify/transforms]]`, `[[classify/output]]`
- `[[label/cli]]`, `[[label/transforms]]`, `[[label/output]]`, `[[label/filter]]`, `[[label/adapters]]`
- `[[vlm/cli]]`, `[[vlm/transforms]]`, `[[vlm/output]]`, `[[vlm/adapters]]`
- `[[gen/cli]]`, `[[gen/transforms]]`, `[[gen/output]]`
- `[[prep/cli]]`, `[[prep/transforms]]`, `[[prep/output]]`
- `[[common/cli]]`, `[[common/records_dataset]]`, `[[common/io]]`, `[[common/fm_runtime]]`, `[[common/fm_provider]]`, `[[common/fm_utils]]`, `[[common/fm_cache]]`, `[[common/stats]]`
