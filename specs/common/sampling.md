# Nested Sampling

Related: **Read first** [records_dataset](records_dataset.md), [keys](keys.md). **See also** [label transforms](../label/transforms.md#sample), [classify transforms](../classify/transforms.md#sample), [prep transforms](../prep/transforms.md#sample), [keys](keys.md).

## Overview

Learning-curve and fraction experiments compare 10%, 25% and 50% of one dataset. They only mean something when the 10% sample is contained in the 25% sample, which is contained in the 50% sample. All three samplers, `label sample`, `class sample` and `prep sample`, share one selection rule that gives this: every item gets a rank from a seeded hash of its stable key, and a sample of size k is the k items of lowest rank.

## Language

- **Stable key**: a string that names an item independently of the dataset's other contents: the [layout path](keys.md#layout-path) for records, the path relative to the source folder for `prep`.
- **Rank**: the SHA-1 hex digest of `"<seed>:<key>"`, compared as a string.
- **Stratum**: a set of records that share the value of the `--stratify-by` keys. A sample draws from every stratum on its own, as if the stratum were the whole dataset.

## Map

- `common/sampling.py`: `rank_order`, `take`, `strata`.

## Contracts

```python
rank_order(keys: Sequence[str], seed: int) -> list[int]    # indices of keys, lowest rank first
take(keys: Sequence[str], seed: int, k: int) -> list[int]  # the first k of rank_order, in ascending index order
strata(values: Sequence[Sequence[str]]) -> list[list[int]]  # record indices of each stratum, strata in sorted value order
```

### Strata

`label sample` and `class sample` take `--stratify-by KEY[,KEY...]`, where each KEY is a [key spec](keys.md#key-specs) that reads one value per record (`split`, `dir:N`, `regex:`, `attr:` or `file:`). A record's stratum is the tuple of its values. Without the option the dataset is one stratum, which is the unstratified behavior.

- The sample is drawn inside each stratum with the rule above. `label sample` returns the chosen records in dataset order. Ranks use the layout path, so a record's rank does not depend on which strata exist.
- `--frac F` applies to every stratum: `floor(n * F)` of its `n` records, at least one for a positive fraction. `--count N` takes N from every stratum and raises, naming the stratum, when one holds fewer.
- `class sample` computes its per-class targets ([classify transforms](../classify/transforms.md#sample)) inside each stratum, so `--stratify-by split` keeps the class proportions of every split.
- `similar:` and a key that cannot read one value per record in isolation MUST raise. `label sample --stratify-by` rejects the name `class`, because an annotated record holds several classes.

- `label sample` ranks all records. `class sample` ranks within each class and applies its per-class targets ([classify transforms](../classify/transforms.md#sample)). `prep sample` ranks the pooled files, using the source index and the relative path as the key.
- `--frac F` takes `floor(n * F)` items, at least one for a positive fraction on a non-empty set. `--count N` takes N.
- `class sample --with-replacement` keeps drawing with a seeded `random.Random`, because a with-replacement draw has no rank order.
- Every sampler defaults to seed `0`. `prep sample` with no `--seed` therefore reproduces.

## Constraints

- For one seed and one set of keys, a smaller sample MUST be a subset of every larger sample. This is checked by a test over several sizes, including across the sizes where `random.sample` changes algorithm.
- The selection MUST NOT depend on the order of the input, and adding an item MUST NOT change the relative order of the others.
- Duplicate keys MUST raise.
- With strata, a smaller sample MUST stay a subset of every larger sample for one seed and one set of strata, and the strata of the sample MUST be those of the dataset: no stratum with records MAY end with none while `--frac` is positive.

## History

- [What belongs in cvsuite](../logs/2026-11-05_what-belongs-in-cvsuite.md): fraction sweeps that projects built by hand.
- [Gaps found by migrating the projects](../logs/2026-11-06_migration-gaps.md): why sampling takes strata, and why `split` is one of them.
