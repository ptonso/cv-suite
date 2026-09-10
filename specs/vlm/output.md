# VLM Output

## 1. General Description
The VLM output layer writes VLM datasets into three persistent formats and one inspection path: flat JSON pairs, shard TAR archives, normalized VQA-style manifests, and interactive preview. The active structured writers are delegated to `cvsuite.common.io.write_dataset(...)`, while cvsuite keeps split filtering, answer-bucket partitioning, cleanup tracking, and stats/report semantics.

## 2. Inputs and Outputs (I/O)
Output commands:
- `to-json`
- `to-shards`
- `to-vqa-style`
- `inspect`

Common output args:
- split filtering through `--split/--splits` on commands that expose it via shared helpers
- `--with-stats` on supported file-writing outputs

Format-specific args:
- `to-json`
  - `dst`
  - `--hardlink`
  - `--skip-images`
- `to-shards`
  - `dst`
  - `--target-shard-size-mb`
  - `--max-samples-per-shard`
- `to-vqa-style`
  - `dst`
  - `--hardlink`
  - `--skip-images`
- `inspect`
  - `--split`
  - `--max`

## 3. Interfaces
Shared output helpers in `vlm/output/common.py`:
- split filtering supports prefix matching, so `train` matches `train2014`
- `iter_answer_partitions(dataset)` groups partitioned datasets by `vlm_answer_bucket`
- `ensure_unpartitioned(dataset)` protects outputs that do not support partitioned datasets

Canonical delegated writers:
- `to-json` -> canonical format `flat_vlm_json`
- `to-shards` -> canonical format `shards_vlm`
- `to-vqa-style` -> canonical format `vqa_style`

## 4. Business Decisions and Strict Policies
- `to-json`, `to-shards`, and `to-vqa-style` all support partitioned datasets. When partitioned, they write one subdirectory per answer bucket.
- `inspect` does not support partitioned datasets and exits with a clear error.
- Empty partitions are still materialized for answer mapping outputs so all canonical buckets, including `unclear`, exist.
- `--skip-images` suppresses image export but still preserves the logical image reference in the written payload.
- Flat JSON write naming prefers a unique existing relative image stem when available; otherwise it falls back to `sample_key`.
- Shard ingest and write logic require complete image/JSON pairs; incomplete pairs are treated as errors.
- cvsuite preserves richer split strings such as `train2014` and only uses prefix matching when users explicitly filter them.

## 5. Implementation Details
Writer behaviors:
- `to-json`
  - writes one JSON sidecar per record
  - can skip image export
  - can hardlink exported images
- `to-shards`
  - rolls shards by approximate target bytes and/or max samples
  - writes under `dst/shards/`
- `to-vqa-style`
  - writes normalized manifest `vqa.json`
  - uses `images_root: "images"` even when images are skipped

Stats behavior through `[[common/stats]]`:
- VLM stats include:
  - unique images
  - total QA pairs
  - answered/unanswered counts
  - question-label counts
  - mapped answer buckets
  - partitions written and/or shards written

Related docs:
- `[[vlm/cli]]`
- `[[vlm/transforms]]`
- `[[common/stats]]`
