# Branch per domain

Affects: [architecture](../architecture.md), [pipeline](../pipeline.md), [prep cli](../prep/cli.md), [classify cli](../classify/cli.md), [label cli](../label/cli.md), [vlm cli](../vlm/cli.md), [gen cli](../gen/cli.md).

Recorded from the maintainer's account on 2026-09-20. The system grew in steps, so the dates of the individual choices are not known.

## Situation

The system started with only the `prep` solutions. Deterministic label solutions came next. Foundation models were added after that, first with label-based ones. Along the way the underlying dataset turned out to differ between kinds of application, and so did the interface each one needs.

## Alternatives considered

A single interface for everything, or a "convert anything to anything" command. Not recorded in more detail.

## Decision

Split the interface by domain: each computer-vision domain gets one large command, and the commands inside it are what can be done in that domain. Adding a new capability then extends one branch and leaves the interfaces of the others stable.

Consequences for the branches:

- The pipeline branches keep their own parsers because their semantics differ. Some transforms need a prompt, others take free-form text, and the formats read and written differ: label work (grounding, semantic segmentation and so on) uses one set of formats, generation another.
- `prep` differs from the rest because it works on a messy folder (zip files, loose images) and organizes it before any dataset format can apply. It therefore works on files, and uses the shared dataset only for orientation scoring.
- `classify` is public only as `class`, for brevity.

## What tipped it

Extensibility: stable interfaces per type of application, as new functionality arrives.

## Trade-offs accepted

Not recorded.
