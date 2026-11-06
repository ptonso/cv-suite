# Perceptual Similarity

Related: **Read first** [records_dataset](records_dataset.md). **See also** [keys](keys.md), [split](split.md), [eda report](../eda/report.md).

## Overview

Near-identical photos are the usual cause of an optimistic test score: a re-export, a burst, a pair of frames a second apart. This module finds them. It hashes each image to 64 bits, compares hashes by Hamming distance at array speed, and links images within a distance into components. `split --group-by similar:BITS` keeps each component on one side of a split, and the `eda` split section measures how close each val or test image is to the train set.

The module also offers a structural similarity (SSIM) on small grayscale thumbnails for analyses that want a finer measure than a hash.

## Language

- **Hash**: the 64-bit perceptual hash of an image (`imagehash.phash`, the function `prep` already uses), stored as 8 packed bytes.
- **Distance**: the number of differing bits between two hashes, from 0 to 64.
- **Twin**: a pair of images whose distance is at most a chosen threshold, or whose SSIM is at least one.

## Map

- `common/similarity.py`. The hash function lives in `prep/core/fs.py` (`perceptual_hash`) and is reused.

## Contracts

```python
hash_bits(paths: Sequence[Path]) -> np.ndarray              # (n, 8) uint8, row i is paths[i]
nearest_distance(query, reference, *, chunk=1024) -> np.ndarray   # (n_query,) smallest distance to any reference row
nearest(query, reference, k=1, *, chunk=1024) -> (distances, indices)   # (n_query, k) each, ordered by distance, then by reference index
pairs_within(bits, max_distance, *, chunk=1024) -> list[tuple[int, int]]   # i < j
components(n, pairs) -> list[int]                           # component id per index, the smallest member index
ssim(a: np.ndarray, b: np.ndarray) -> float                 # gray float arrays of equal shape, Gaussian window 11, sigma 1.5
gray_thumb(path, size=160) -> np.ndarray                    # gray thumbnail used for ssim
```

`nearest` gives the closest reference rows themselves, which the `eda` split section needs for its twin table and for the SSIM shortlist. A caller that already holds the hashes, such as the `eda` image table, passes them as packed rows (`hash_bits` layout) instead of hashing again.

Distances are computed on XORed packed bytes with a 256-entry popcount table, in chunks, so memory stays bounded for tens of thousands of images. `ssim` is the Gaussian-window index and uses OpenCV for the filtering.

A threshold is stated in bits for hashes (`similar:6`, `--twin-threshold 6`) and as an index in `[0, 1]` for SSIM (`--twin-threshold 0.6`). The two scales are not interchangeable.

## Constraints

- An unreadable image MUST raise and name its path. No hash is invented for it.
- Results MUST be deterministic and independent of chunk size.
- `pairs_within`, `nearest_distance` and `nearest` MUST NOT materialize the full n by n distance matrix.
- A threshold MUST be validated: bits in `[0, 64]`, SSIM in `[0, 1]`.
- `prep arrange --perceptual-dedup` keeps its own contract ([prep transforms](../prep/transforms.md#arrange)) and MAY adopt these functions without changing it.

## History

- [What belongs in cvsuite](../logs/2026-11-05_what-belongs-in-cvsuite.md): the cross-split duplicate checks that projects wrote by hand.
- [The eda branch](../logs/2026-11-05_eda-branch.md): why the default measure is a hash and SSIM is optional.
