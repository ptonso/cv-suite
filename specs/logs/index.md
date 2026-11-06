# Log

- [2026-09-20 Branch per domain](2026-09-20_branch-per-domain.md): why the CLI is split by domain, and why `prep`, `class` and the pipeline parsers look the way they do.
- [2026-09-20 Shared dataset contract](2026-09-20_shared-dataset-contract.md): why formats and models meet in one in-memory model.
- [2026-09-20 Isolated provider environments](2026-09-20_isolated-provider-environments.md): why each local model runs in its own virtual environment and process.
- [2026-09-20 Central model cache](2026-09-20_central-model-cache.md): why model weights are centralized and Hugging Face downloads are reused.
- [2026-11-05 Idempotent outputs](2026-11-05_idempotent-outputs.md): why a rerun into a dirty destination raises, how `--overwrite` replaces only what a writer owns, and why same-run collisions keep their suffix.
- [2026-11-05 What belongs in cvsuite](2026-11-05_what-belongs-in-cvsuite.md): the four-question test for adding a routine, what passed it, and what stays in projects.
- [2026-11-05 Chained transforms and split as a transform](2026-11-05_chained-transforms-and-split.md): why a run takes several transforms, why splitting leaves the outputs, and the clean breaks that follow.
- [2026-11-05 The eda branch](2026-11-05_eda-branch.md): why dataset analysis is a branch, its sections, and how it scales to 1000 classes.
- [2026-11-06 Gaps found by migrating the projects](2026-11-06_migration-gaps.md): what the four projects still did by hand after the first migration, and the options that close it.
