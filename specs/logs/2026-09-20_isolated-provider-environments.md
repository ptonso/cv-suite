# Isolated provider environments

Affects: [fm_runtime](../common/fm_runtime.md), [fm_provider](../common/fm_provider.md), [fm_cache](../common/fm_cache.md).

Recorded from the maintainer's account on 2026-09-20.

## Situation

Most models are research projects. Some cannot be imported as Python packages, and many need dependencies that conflict with each other.

## Alternatives considered

A dedicated container per model. It may come in the future; a specialized virtual environment is considered more than enough for now.

## Decision

Each local provider runs as its own process in its own managed virtual environment. Conflicting dependencies are isolated that way, and the rest of the system was split around that boundary: the branch talks to a provider only through files (`in.json`, `out.json`). API providers need no isolation and run in the current interpreter.

## What tipped it

Conflicting dependencies between models.

## Trade-offs accepted

Not recorded.
