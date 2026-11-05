# Nested Sampling

Related: **Read first** [records_dataset](records_dataset.md). **See also** [label transforms](../label/transforms.md#sample), [classify transforms](../classify/transforms.md#sample), [prep transforms](../prep/transforms.md#sample), [keys](keys.md).

## Overview

Learning-curve and fraction experiments compare 10%, 25% and 50% of one dataset. They only mean something when the 10% sample is contained in the 25% sample, which is contained in the 50% sample. All three samplers, `label sample`, `class sample` and `prep sample`, share one selection rule that gives this: every item gets a rank from a seeded hash of its stable key, and a sample of size k is the k items of lowest rank.

## Language

- **Stable key**: a string that names an item independently of the dataset's other contents: the [layout path](keys.md#layout-path) for records, the path relative to the source folder for `prep`.
- **Rank**: the SHA-1 hex digest of `"<seed>:<key>"`, compared as a string.

## Map

- `common/sampling.py`: `rank_order`, `take`.

## Contracts

```python
rank_order(keys: Sequence[str], seed: int) -> list[int]    # indices of keys, lowest rank first
take(keys: Sequence[str], seed: int, k: int) -> list[int]  # the first k of rank_order, in ascending index order
```

- `label sample` ranks all records. `class sample` ranks within each class and applies its per-class targets ([classify transforms](../classify/transforms.md#sample)). `prep sample` ranks the pooled files, using the source index and the relative path as the key.
- `--frac F` takes `floor(n * F)` items, at least one for a positive fraction on a non-empty set. `--count N` takes N.
- `class sample --with-replacement` keeps drawing with a seeded `random.Random`, because a with-replacement draw has no rank order.
- Every sampler defaults to seed `0`. `prep sample` with no `--seed` therefore reproduces.

## Constraints

- For one seed and one set of keys, a smaller sample MUST be a subset of every larger sample. This is checked by a test over several sizes, including across the sizes where `random.sample` changes algorithm.
- The selection MUST NOT depend on the order of the input, and adding an item MUST NOT change the relative order of the others.
- Duplicate keys MUST raise.

## History

- [What belongs in cvsuite](../logs/2026-11-05_what-belongs-in-cvsuite.md): fraction sweeps that projects built by hand.
