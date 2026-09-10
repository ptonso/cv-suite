# specs

These documents are the technical source of truth for cvsuite: architecture, the
shared dataset contract, per-branch behavior, business decisions, and the
guardrails that must not break. Code and tests are the implementation reference;
these specs describe the contracts around them.

For a friendly introduction — install, the command model, worked examples — see
[`../docs/`](../docs/) instead. Guides teach; specs constrain.

## Layout

- [`overview.md`](overview.md) — what the system is and the shape of every workflow
- [`architecture.md`](architecture.md) — repository structure and layer ownership
- [`vocabulary.md`](vocabulary.md) — the shared terms used everywhere
- [`cli.md`](cli.md) — top-level `cvsuite <branch>` dispatch
- [`dataclasses.md`](dataclasses.md) — the canonical in-memory types

Per branch: [`label/`](label/), [`classify/`](classify/), [`vlm/`](vlm/),
[`gen/`](gen/), [`prep/`](prep/).

Shared infrastructure: [`common/`](common/) — the dataset contract
(`records_dataset.md`), format I/O (`io.md`), the FM runtime, provider and cache
contracts, and stats.

Cross-references between specs use `[[path]]` link syntax, relative to `specs/`.

## Contributing without breaking contracts

Until a dedicated contribution guide exists, the rules to respect when extending
the system are stated inline in the relevant spec:

- adding a branch → [`architecture.md`](architecture.md) (only packages with
  `cli.py` are public branches)
- adding a transform or output command → the branch's `cli.md` (discovery rules)
  and `transforms.md` / `output.md`
- adding an FM provider → [`common/fm_provider.md`](common/fm_provider.md) and the
  template at `src/cvsuite/common/fm/providers/_template.py`
- touching the dataset contract → [`common/records_dataset.md`](common/records_dataset.md)
  (it is owned once, in `cvsuite.common.core`)
