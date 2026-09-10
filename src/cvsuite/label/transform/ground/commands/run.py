"""Run grounding models over label datasets using prompt-per-forward-pass semantics."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvsuite.common.fm.providers.registry import format_available_providers

DEFAULT_THRESHOLD = 0.3
DEFAULT_IOU_THRESHOLD = 0.4

SUPPORTED_PROVIDERS = ("sam3", "gsam", "gdino", "llmdet", "locate_anything", "rex_omni", "yolo_e")
REFERENCE_PROVIDERS = ("yolo_e",)
MODEL_ID_PROVIDERS = ("llmdet", "locate_anything")


def attach(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", "--model", dest="provider", choices=SUPPORTED_PROVIDERS, required=True, help="Grounding provider id.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single grounding prompt string, YAML list of prompts, or YAML label-to-prompts mapping.",
    )
    parser.add_argument(
        "--reference-folder",
        dest="reference_folder",
        type=Path,
        default=None,
        help=(
            "Folder of reference images+labels (auto-detected format). YOLO-E generalizes these "
            "labels onto the target dataset. Mutually exclusive with --prompt; yolo_e only."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu", "cuda"],
        default="auto",
        help="Execution device preference. sam3 requires CUDA.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size for model inference.")
    parser.add_argument("--precision", default="fp32", help="Numerical precision hint (fp32/fp16/bf16).")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional backend checkpoint id. Supported by llmdet and locate_anything.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional model-specific YAML config.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any resumable FM run state and force a fresh model pass.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Global confidence threshold applied by the label ground transform after FM inference.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU NMS threshold applied per output label after prompt-to-label remapping.",
    )
    parser.epilog = format_available_providers("ground")


def run(dataset, args: argparse.Namespace):
    from .. import core

    has_prompt = bool(getattr(args, "prompt", None) and str(args.prompt).strip())
    reference_folder = getattr(args, "reference_folder", None)
    if has_prompt == bool(reference_folder):
        raise ValueError("ground requires exactly one of --prompt or --reference-folder.")
    model_id = str(getattr(args, "model_id", "") or "").strip()
    if model_id and args.provider not in MODEL_ID_PROVIDERS:
        raise ValueError(f"--model-id is only supported by provider(s): {', '.join(MODEL_ID_PROVIDERS)}.")
    if reference_folder is not None and args.provider not in REFERENCE_PROVIDERS:
        raise ValueError(f"--reference-folder is only supported by provider(s): {', '.join(REFERENCE_PROVIDERS)}.")

    base_classes = list(dataset.classes)
    before_counts = core.snapshot_annotation_counts(dataset)

    if reference_folder is not None:
        prompt_to_label: dict[str, str] = {}
        grounded = None
        try:
            grounded = core.run_ground_reference(
                dataset,
                model=args.provider,
                reference_folder=reference_folder,
                threshold=float(args.threshold),
                device=args.device,
                precision=args.precision,
                batch_size=args.batch,
                config_path=args.config,
                no_resume=bool(args.no_resume),
            )
            core.filter_new_annotations(
                grounded,
                before_counts,
                threshold=float(args.threshold),
                iou_threshold=float(getattr(args, "iou_threshold", core.DEFAULT_IOU_THRESHOLD)),
                prompt_to_label=prompt_to_label,
                base_classes=base_classes,
            )
            return grounded
        finally:
            if grounded is not None:
                core.clear_ground_prompts(grounded)
            core.clear_ground_prompts(dataset)

    prompt_spec = core.parse_prompt_spec(args.prompt)
    grounded = None
    try:
        grounded = core.run_ground(
            dataset,
            model=args.provider,
            prompts=prompt_spec.prompts,
            threshold=float(args.threshold),
            device=args.device,
            precision=args.precision,
            batch_size=args.batch,
            config_path=args.config,
            model_id=model_id,
            no_resume=bool(args.no_resume),
        )
        core.filter_new_annotations(
            grounded,
            before_counts,
            threshold=float(args.threshold),
            iou_threshold=float(getattr(args, "iou_threshold", core.DEFAULT_IOU_THRESHOLD)),
            prompt_to_label=prompt_spec.prompt_to_label,
            base_classes=base_classes,
        )
        return grounded
    finally:
        if grounded is not None:
            core.clear_ground_prompts(grounded)
        core.clear_ground_prompts(dataset)
