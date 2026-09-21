# Shared dataset contract

Affects: [records_dataset](../common/records_dataset.md), [io](../common/io.md), [architecture](../architecture.md), [fm_runtime](../common/fm_runtime.md).

Recorded from the maintainer's account on 2026-09-20.

## Situation

Formats on disk differ per domain, and the model scripts should not care which one they were fed.

## Alternatives considered

Not recorded.

## Decision

Keep one in-memory dataset between input/output and everything else. Format reading and writing is abstracted out into adapters, so new formats are added as new components while every system, especially the foundation-model scripts, keeps operating over the same interface.

The model request travels in a dedicated `fm_request` field and not in the general `meta` dictionary. The maintainer's recollection is that this gives the foundation-model scripts a shared structure; the exact reason was not remembered.

## What tipped it

Decoupling input/output from the rest of the system so it can grow independently.

## Trade-offs accepted

Not recorded.
