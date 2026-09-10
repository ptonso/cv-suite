# Classify Transforms

## 1. General Description
The classify branch currently exposes two transforms: `infer` and `sample`. `infer` is the FM-backed classification path. `sample` is the deterministic class-aware sampling path for already labeled class datasets. Both transforms operate over the shared dataset objects used by cvsuite and preserve branch-owned class-directory semantics documented in `[[classify/output]]`.

## 2. Inputs and Outputs (I/O)
`infer` inputs:
- `--provider/--model`
- `--prompt`
- `--template-prompt`
- `--device {auto,cpu,gpu}`
- `--batch`
- `--precision`
- `--config`
- `--no-resume`

`infer` outputs:
- same dataset object, mutated through `FMRunner`
- `dataset.fm_request` populated for task `classify`

`sample` inputs:
- `--mode {preserve,balance}`
- exactly one of `--max-n-per-class` or `--max-frac-per-class`
- `--with-replacement`
- `--hardlink`
- `--seed`

`sample` outputs:
- a filtered dataset
- sampling metadata under `dataset.meta["sample"]`

## 3. Interfaces
`infer` interface:
- builds an `FMRequest` with `task="classify"`
- provider family resolves to `classify`
- only forwards `template_prompt` through `model_args`
- invokes `FMRunner.from_dataset(dataset).run(dataset, no_resume=...)`

Classification prompt interfaces:
- `--prompt` may be:
  - a comma-separated label string
  - a YAML/JSON label list
  - a YAML/JSON mapping from label to one or more prompt strings
- when `--prompt` is list-like, `--template-prompt` is applied by replacing `<class>` with each label
- explicit prompt maps bypass template injection and use the provided prompt strings verbatim
- default template is `a photo of a <class>`

`sample` interface:
- consumes class labels from `record.classification`
- writes sampling bookkeeping under `dataset.meta["sample"]`:
  - `mode`
  - `seed`
  - `with_replacement`
  - `hardlink`
  - `size_knob` (`max_n_per_class` or `max_frac_per_class`)
  - `size_value` (the chosen knob's value)
  - `targets`
  - `class_sizes`
  - `sample_size`

## 4. Business Decisions and Strict Policies
`infer` policies:
- if classify ingest mode meta indicates active multi-class ingest, the transform clears the special ingest-mode behavior before FM inference so the dataset becomes a standard inferable classify dataset
- classification prompting uses the shared class-prompt utilities, including the `<class>` template token requirement for list-like prompts

`sample` policies:
- rejects annotated datasets and tells the user to use `cvsuite label sample`
- rejects active multi-class ingest datasets because sampling is currently single-label only
- rejects unlabeled datasets and points the user toward `cvsuite prep sample`
- rejects datasets that already carry annotations
- `preserve` mode preserves relative class proportions under a cap
- `balance` mode forces equal per-class targets under the chosen cap
- sampling without replacement errors if any target exceeds class size
- fraction-based targets floor counts, but non-zero target logic still guarantees at least 1 when appropriate in branch code paths that produce positive targets

Exact sampling heuristics:
- `preserve` + `max_n_per_class`
  - target for class = `floor(class_size * max_n_per_class / max_class_size)`, minimum 1
- `preserve` + `max_frac_per_class`
  - target for class = `floor(class_size * frac)`
- `balance` + `max_n_per_class`
  - target for every class = exact cap
- `balance` + `max_frac_per_class`
  - target for every class = `floor(min_class_size * frac)`, minimum 1

## 5. Implementation Details
Ingest-side label semantics that transforms depend on:
- regular unstructured ingest may infer label from the parent folder or from the folder before `images`
- multi-class ingest groups identical basenames across class folders and stores the full label list in both cvsuite metadata and the canonical class-dir multi-label attribute
- duplicate basename images in multi-class ingest must be byte-identical and size-identical

Threshold-related logic used later by outputs but rooted in classify semantics:
- effective label selection prefers `classification.probs` when available
- global threshold, per-class thresholds, and threshold metadata all participate in output-time label routing
- multi-class outputs emit all labels meeting threshold rather than only the argmax label

Related docs:
- `[[classify/cli]]`
- `[[classify/output]]`
- `[[common/fm_runtime]]`
