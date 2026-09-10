"""
Template for new FM model wrappers.

The fm runner executes a model entrypoint inside a managed venv with the following contract:
- CLI args: only --input / --output / --no-resume; runtime options come from dataset.fm_request.
- I/O: read a VisionDataset JSON from --input, mutate it with predictions/metadata,
  and write the updated dataset to --output.
- Checkpoints: BaseFMModel writes per-batch JSONL checkpoints under FM_MODEL_CACHE/runs/<run-id>.
- Env: FM_CALLER_CWD contains the working directory of the caller; use it to resolve
  relative paths if needed.
- Caches: two different roots, do not mix them.
  - ctx.hub_dir (FM_HUB_DIR) is the shared Hugging Face repo cache. Pass it to
    utils.materialize_hf_model_source(hub_dir=...) so a model the user already has is
    symlinked from their own HF cache instead of downloaded again.
  - ctx.weights_dir (FM_WEIGHTS_DIR) is this provider's own writable directory, for
    artifacts that are not HF repos (direct URL downloads, framework-managed checkpoints).
- Upstream git checkouts belong in <provider>/pkgs, cloned by the setup script as
  PKG_ROOT="${cache_root}/pkgs" and read back via load_repo_checkout(ctx, "<Repo>").
  Never clone inside the venv: the runtime deletes it before every rebuild.
- Registration: add the new wrapper to `cvsuite.common.fm.providers.registry.MODEL_MODULES`.
"""

from __future__ import annotations

from dataclasses import dataclass

from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.providers.base import BatchResult, RuntimeContext
from cvsuite.common.fm.providers.bases import BaseRecordImageModel, RecordImageJob


@dataclass(frozen=True)
class TemplateOptions:
    threshold: float | None = None


TemplateJob = RecordImageJob


class TemplateModel(BaseRecordImageModel[TemplateOptions, None]):
    model_name = "template"
    description = "Template FM wrapper."
    options_cls = TemplateOptions

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[TemplateOptions]) -> None:
        # Resolve resources relative to the caller cwd if you accept relative paths:
        #   caller_cwd = ctx.caller_cwd
        # Use ctx.weights_dir as a writable cache directory for HF or custom checkpoints.
        return None

    def process_batch(self, dataset: VisionDataset, batch, runtime, ctx: RuntimeContext[TemplateOptions]) -> BatchResult:
        modified = []
        for job in batch:
            rec = dataset.records[job.record_idx]
            rec.attributes.setdefault("fm_tasks", []).append("my-task")
            modified.append(job.record_idx)
        return BatchResult(modified_record_indices=modified)

    def build_fm_meta(self, dataset: VisionDataset, runtime, ctx: RuntimeContext[TemplateOptions]) -> dict[str, object]:
        return {
            "task": "template",
            "model": self.model_name,
            "device": ctx.request.device,
            "precision": ctx.request.precision,
            "batch_size": ctx.batch_size,
            "weights": str(ctx.weights_dir) if ctx.weights_dir else None,
            "config": str(ctx.config_path) if ctx.config_path else None,
        }


MODEL = TemplateModel()


def main(argv=None) -> int:
    return MODEL.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
