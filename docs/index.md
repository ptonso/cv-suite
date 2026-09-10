# cvsuite guides

These pages are the friendly introduction. For exact behavior and the contracts that must not break, see [`specs/`](../specs/).

- [Getting started](getting-started.md) — install, run your first command, understand FM backends.
- [Cookbook](cookbook.md) — copy-paste recipes for the common workflows.

## The mental model

cvsuite does one shape of work: **take a source on disk, optionally run one transform, write a destination on disk.** Every pipeline branch reads the same way:

```
cvsuite <branch> <src> [<transform> <transform args>] <output-command> [output args]
```

- `<branch>` picks the domain: `label`, `class`, `vlm`, `gen`, `prep`, `common`.
- `<src>` is a file or folder. The branch detects its format unless you pass `--from`.
- `<transform>` is optional and there is at most one. It mutates the dataset in memory.
- `<output-command>` is required. It writes the result. Output commands are spelled `to-<something>` (`to-coco`, `to-class-dir`, `to-json`, `to-dst`, …).

`cvsuite prep` is the exception: it is plain subcommands (`cvsuite prep arrange`, `cvsuite prep process`, …) that work directly on folders, with no transform/output chaining.

## Why branches instead of one converter

There is deliberately no universal "convert anything to anything" command. A YOLO dataset and a VQA corpus need different ingest heuristics, different transforms, and different export rules, so each branch owns those decisions. What they share is the in-memory representation:

- `VisionDataset` holds records; `VisionRecord` is one image and its annotations; `ImageInfo` is the image path and size.
- Reading and writing on-disk formats is centralized in `cvsuite.common.io`.
- Model inference is centralized in `cvsuite.common.fm`.

You rarely touch these directly from the CLI, but knowing they exist explains why, for example, `cvsuite label` and `cvsuite gen edit` can both ingest a YOLO folder.

## Foundation models

Transforms like `ground`, `ocr`, `infer`, `caption`, `vqa`, and the `gen` actions run a model. cvsuite does **not** pull heavy ML dependencies into the base install. Instead, the first time you use a provider, cvsuite builds a dedicated virtualenv for it and runs the model in a subprocess. Weights and virtualenvs go to one cache outside the package (`~/.cache/cvsuite` by default), and models already in your Hugging Face cache are reused rather than re-downloaded. See [Getting started](getting-started.md#foundation-model-backends) for details and cache management.

## Determinism

Sampling and split assignment use fixed default seeds. Most writers add a collision suffix rather than overwriting. Re-running the same command generally reproduces the same output.
