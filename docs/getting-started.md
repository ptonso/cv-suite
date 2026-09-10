# Getting started

## Install

cvsuite needs Python 3.10 or newer.

```bash
pip install cv-suite
```

From source, for development:

```bash
git clone https://github.com/ptonso/cv-suite
cd cv-suite
pip install -e ".[dev]"
```

This installs the `cvsuite` command, the dataset I/O layer, and the pure-Python transforms. It does **not** install PyTorch or any model libraries — those are handled per-provider (see below).

Check it works:

```bash
cvsuite --help
cvsuite label --help
```

## The command shape

Every pipeline branch reads left to right:

```
cvsuite <branch> <src> [<transform> <transform args>] <output-command> [output args]
```

A minimal run has no transform:

```bash
cvsuite label ./my-yolo-dataset to-coco ./my-dataset
```

Here `cvsuite label` ingested a YOLO folder (auto-detected), built the in-memory dataset, and `to-coco` wrote it back out as a COCO dataset directory (`<split>/_annotations.coco.json` + images + a helper `data.yaml`). Every output command writes a directory; pass `--no-images` to `to-coco` to write only the annotations.

Add one transform in the middle to change the data before it is written:

```bash
cvsuite label ./my-yolo-dataset sample --count 200 to-coco ./subset
```

`cvsuite prep` is different — it is plain subcommands over folders, with no chaining:

```bash
cvsuite prep process ./raw ./normalized --size 1024 --resize-mode cap-long
```

## Picking the source format

Branches auto-detect the source. Override with `--from` when detection is ambiguous or you want to be explicit:

```bash
cvsuite label ./folder --from images to-yolo ./out
cvsuite vlm ./folder --from shards to-json ./out
```

If a directory matches more than one structured format, ingest fails rather than guessing — pass `--from` to resolve it.

## Foundation-model backends

Transforms that run a model — `label ground`, `label ocr`, `class infer`, `vlm caption` / `ask` / `vqa`, and both `gen` actions — select a **provider** (`clip`, `gsam`, `qwen`, `paddleocr`, `flux`, …).

cvsuite keeps these out of your main environment. The first time you use a local provider:

1. cvsuite runs the provider's setup script (`src/cvsuite/common/fm/providers/setup_venv/<provider>.sh`).
2. That builds a dedicated virtualenv under `$CVSUITE_HOME/providers/<provider>/` and fetches weights into `$CVSUITE_HOME/hub/`.
3. The model runs in a subprocess against that venv; results are merged back into your dataset.

Subsequent runs reuse the venv and weights. API-backed providers (e.g. `openrouter`) skip the venv and just need an API key in the environment or a nearby `.env` file.

### Where things are cached

Everything cvsuite downloads lives in one place, outside the installed package:

```bash
cvsuite common cache-path
```

```
cache root:    ~/.cache/cvsuite        (set CVSUITE_HOME to move it)
hub:           ~/.cache/cvsuite/hub
providers:     ~/.cache/cvsuite/providers
hf cache:      ~/.cache/huggingface/hub (reused by symlink when it already has a model)
```

Provider virtualenvs and model weights are large — tens of gigabytes each for the diffusion and VLM providers. Point `CVSUITE_HOME` at a disk with room:

```bash
export CVSUITE_HOME=/mnt/big/cvsuite
```

`CVSUITE_HOME` is read from the environment or from a `.env` file in your project (or any parent directory), the same way Hugging Face tokens are. If it is unset, cvsuite follows `XDG_CACHE_HOME` and otherwise uses `~/.cache/cvsuite`. To keep weights on a separate disk from the venvs, set `CVSUITE_HUB_DIR` as well.

### Reusing models you already have

If a model is already in your Hugging Face cache — because you pulled it for another project — cvsuite **symlinks** it instead of downloading it again. Nothing is copied, and cvsuite never writes into your Hugging Face cache: if your copy is incomplete or pinned to a different revision, cvsuite fetches its own private copy instead.

`cache-list` marks which is which:

```
hub repos: 2 (1 shared)
  shared  models--facebook--sam3 -> ~/.cache/huggingface/hub/models--facebook--sam3
  owned   models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K  3.9 GB
```

cvsuite finds your cache through the usual variables — `HF_HUB_CACHE`, then `HUGGINGFACE_HUB_CACHE`, then `$HF_HOME/hub`, then `~/.cache/huggingface/hub`.

### Managing the cache

```bash
cvsuite common cache-list                          # what is installed, and how big
cvsuite common cache-list --json
cvsuite common cache-purge clip                    # one provider's venv + weights
cvsuite common cache-purge --all --part venv --dry-run
cvsuite common cache-purge --all --part hub        # weights only, keep the venvs
```

`cache-purge` asks for confirmation unless you pass `--yes`, and `--dry-run` shows the plan without touching anything.

Purging a **shared** model only removes cvsuite's symlink — the model stays in your Hugging Face cache for whatever else uses it.

## Stats sidecars

File-writing outputs that opt in accept `--with-stats`, which drops a `stats.yaml` next to the output describing what was written:

```bash
cvsuite class ./images infer --provider clip --prompt "good,bad" to-class-dir ./out --with-stats
```

## Next

- [Cookbook](cookbook.md) — recipes for each branch.
- [`specs/`](../specs/) — exact behavior, flags, and guardrails.
