# Central model cache

Affects: [fm_cache](../common/fm_cache.md), [fm_runtime](../common/fm_runtime.md), [fm_utils](../common/fm_utils.md).

Recorded from the maintainer's account on 2026-09-20.

## Situation

A model may already have been downloaded for another application. Weights also need one known place, so that it is possible to tell which models exist.

## Alternatives considered

Not recorded.

## Decision

Everything cvsuite stores for models lives under one root, and Hugging Face models already present in the user's global Hugging Face cache are reused instead of downloaded again. The details of the layout, the completeness check and the treatment of the old in-package cache are in [fm_cache](../common/fm_cache.md).

## What tipped it

Reuse of existing downloads, and a single place from which to see which models exist.

## Trade-offs accepted

Not recorded. The reasons for the completeness check on shared snapshots and for dropping the old in-package `.cache` without migration were not given.
