# cvsuite specs

Related: **Read first** [architecture](architecture.md). **See also** [../docs/](../docs/).

cvsuite is a command-line tool for computer-vision dataset work: preparing raw files, labeling, classifying, question-answering and generating images, with foundation models running in isolated environments. These specs describe the contracts a contributor must respect and the vocabulary the code uses. Code and tests are the implementation reference. For install steps and worked examples read the guides in [`../docs/`](../docs/): guides teach, specs constrain.

Every spec has the same shape: Overview, Language, Map, Contracts, Constraints, and History when a [log note](logs/index.md) explains it. Specs use relative markdown links only.

## Reading order

Start here:

- [architecture](architecture.md): the branches, the three layers, shared terms, top-level dispatch.
- [pipeline](pipeline.md): the source, transform, output shape shared by `class`, `label`, `vlm`, `gen`, and the `branch_cli` helpers.

Shared infrastructure, in dependency order:

- [common/records_dataset](common/records_dataset.md): the shared in-memory dataset and its types.
- [common/io](common/io.md): format router and adapters.
- [common/fm_runtime](common/fm_runtime.md): how a foundation model is run in its own process.
- [common/fm_provider](common/fm_provider.md): providers, families, registry, provider process contract.
- [common/fm_utils](common/fm_utils.md): shared provider helpers.
- [common/fm_cache](common/fm_cache.md): model cache layout, Hugging Face sharing, purge.
- [common/stats](common/stats.md): `stats.yaml`.
- [common/cli](common/cli.md): `cvsuite common` cache commands.

Branches:

- `prep`: [cli](prep/cli.md), [transforms](prep/transforms.md), [output](prep/output.md).
- `label`: [cli](label/cli.md), [adapters](label/adapters.md), [transforms](label/transforms.md), [filter](label/filter.md), [output](label/output.md).
- `class`: [cli](classify/cli.md), [transforms](classify/transforms.md), [output](classify/output.md).
- `vlm`: [cli](vlm/cli.md), [adapters](vlm/adapters.md), [transforms](vlm/transforms.md), [output](vlm/output.md).
- `gen`: [cli](gen/cli.md), [transforms](gen/transforms.md), [output](gen/output.md).

History: [logs/index.md](logs/index.md).

## Contributing without breaking contracts

- Adding a branch: [architecture](architecture.md#contracts). Only packages with `cli.py` are public.
- Adding a transform or output command: the branch's `cli.md`, `transforms.md` and `output.md`, and [pipeline](pipeline.md).
- Adding a provider: [fm_provider](common/fm_provider.md#extending).
- Adding a dataset format: [io](common/io.md).
- Touching the dataset: [records_dataset](common/records_dataset.md). It is owned once, in `cvsuite.common.core`.
