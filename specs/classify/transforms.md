# Classify Transforms

Related: **Read first** [classify cli](cli.md). **See also** [classify output](output.md), [fm_runtime](../common/fm_runtime.md).

## Overview

The `class` branch has two transforms. `infer` classifies each image with a foundation model. `sample` picks a subset of an already labeled class dataset while keeping class balance rules. Both keep the class-folder semantics that [output](output.md) writes.

## Language

- **Preserve** sampling: keep relative class proportions under a cap. **Balance** sampling: force equal per-class targets.
- **Class template**: a prompt containing `<class>` that is filled with each label.

## Map

- `classify/transform/commands/infer.py`, `sample.py`.
- Prompt helpers: [fm_utils](../common/fm_utils.md).

## Contracts

### infer

Arguments: `--provider/--model`, `--prompt`, `--template-prompt`, `--device {auto,cpu,gpu}`, `--batch`, `--precision`, `--config`, `--no-resume`.

Builds an `FMRequest(task="classify")`, forwards only `template_prompt` through `model_args`, and calls `FMRunner.from_dataset(dataset).run(dataset, no_resume=...)`. Provider family is `classify`. The dataset is mutated through the runner.

`--prompt` is one of: a comma-separated label string; a YAML/JSON label list; a YAML/JSON mapping from label to one or more prompt strings. For list-like prompts, `--template-prompt` replaces `<class>` with each label (default `a photo of a <class>`). Explicit prompt maps are used verbatim.

### sample

Arguments: `--mode {preserve,balance}`; exactly one of `--max-n-per-class` or `--max-frac-per-class`; `--with-replacement`; `--hardlink`; `--seed`.

Reads labels from `record.classification` and writes `dataset.meta["sample"]` with `mode`, `seed`, `with_replacement`, `hardlink`, `size_knob` (`max_n_per_class` or `max_frac_per_class`), `size_value`, `targets`, `class_sizes`, `sample_size`.

Targets:

| mode | knob | target per class |
|---|---|---|
| preserve | `max_n_per_class` | `floor(class_size * max_n / max_class_size)`, minimum 1 |
| preserve | `max_frac_per_class` | `floor(class_size * frac)` |
| balance | `max_n_per_class` | the cap |
| balance | `max_frac_per_class` | `floor(min_class_size * frac)`, minimum 1 |

### Threshold semantics used at output

Effective label selection prefers `classification.probs`. Global, per-class and metadata thresholds decide routing; multi-class outputs emit every label meeting its threshold, not only the argmax. See [output](output.md).

## Constraints

- `infer` MUST clear the multi-class ingest mode before inference, so the dataset becomes a standard inferable class dataset.
- List-like `--prompt` MUST contain the `<class>` template token through `--template-prompt`.
- `sample` MUST reject annotated datasets (use `cvsuite label sample`), datasets with active multi-class ingest (sampling is single-label only), unlabeled datasets (use `cvsuite prep sample`), and datasets that already carry annotations.
- `sample` without replacement MUST error if any target exceeds its class size.
- Fraction targets floor their counts; the formulas above give the minimums.
