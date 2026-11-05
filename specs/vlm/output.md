# VLM Output

Related: **Read first** [vlm cli](cli.md). **See also** [vlm transforms](transforms.md), [io](../common/io.md), [stats](../common/stats.md).

## Overview

The VLM outputs write three persistent formats (flat JSON pairs, shard TAR archives, normalized VQA-style manifests) and offer one interactive preview. The structured writers delegate to `cvsuite.common.io.write_dataset(...)`; the branch keeps split filtering, answer-bucket partitioning, cleanup tracking and stats.

## Language

- **Partitioned dataset**: a dataset after `map-answers`, where each record carries `vlm_answer_bucket`. Outputs write one subdirectory per bucket.
- **Prefix match**: split filtering where `train` matches `train2014`.

## Map

- `vlm/output/commands/`: `to_json.py`, `to_shards.py`, `to_vqa_style.py`, `inspect.py`.
- `vlm/output/common.py`: split filtering and partition helpers.

## Contracts

- `to-json` (canonical `flat_vlm_json`): `dst`, `--link-mode {copy,hard,sym}`, `--skip-images`. Writes one JSON sidecar per record.
- `to-shards` (canonical `shards_vlm`): `dst`, `--target-shard-size-mb`, `--max-samples-per-shard`. Rolls shards by approximate bytes and/or samples, under `dst/shards/`.
- `to-vqa-style` (canonical `vqa_style`): `dst`, `--link-mode {copy,hard,sym}`, `--skip-images`. Writes manifest `vqa.json` with `images_root: "images"`, even when images are skipped.
- `inspect`: `--split`, `--max`.
- Image placement follows [pipeline](../pipeline.md#link-mode). `to-shards` transcodes images and has no link mode.
- Common: split filtering through `--split/--splits` where exposed; `--with-stats` on file-writing outputs.

Helpers (`vlm/output/common.py`): split filtering with prefix matching, `iter_answer_partitions(dataset)` (groups by `vlm_answer_bucket`), `ensure_unpartitioned(dataset)`.

Stats (`build_vlm_stats`): unique images, QA pairs, answered and unanswered counts, question-label counts, mapped answer buckets, partitions and shards written.

## Constraints

- `to-json`, `to-shards` and `to-vqa-style` MUST support partitioned datasets by writing one subdirectory per bucket.
- `inspect` MUST reject partitioned datasets with a clear error.
- Empty partitions MUST still be materialized for answer-mapping outputs, so all canonical buckets, including `unclear`, exist.
- `--skip-images` MUST suppress image export and MUST keep the logical image reference in the payload.
- Flat JSON naming MUST prefer a unique existing relative image stem, else `sample_key`.
- Shard writing MUST require complete image and JSON pairs.
- Richer split strings such as `train2014` MUST be preserved; prefix matching applies only when users filter.

## History

- [Chained transforms and split as a transform](../logs/2026-11-05_chained-transforms-and-split.md): why `--link-mode` replaced `--hardlink` on these outputs.
