from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Mapping, Tuple

import numpy as np
import yaml

from cvsuite.common.core import BBox, Polygon
from cvsuite.common.core.enums import Task
from cvsuite.common.core import FMRequest
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.core import FMRunner
from cvsuite.common.fm.core import utils as fm_utils
from cvsuite.common.fm.providers.registry import allowed_model_ids_for_provider

GROUND_PROMPTS_KEY = "ground_prompts"
DEFAULT_THRESHOLD = 0.3
DEFAULT_IOU_THRESHOLD = 0.4


@dataclass(frozen=True)
class GroundPromptSpec:
    prompts: List[str]
    prompt_to_label: Dict[str, str]
    labels: List[str]


def _normalize_device_pref(device: str) -> str:
    choice = str(device or "auto").strip().lower()
    if choice == "gpu":
        return "cuda"
    return choice


def parse_prompt_spec(raw_prompt: str) -> GroundPromptSpec:
    prompt_value = str(raw_prompt or "").strip()
    if not prompt_value:
        raise ValueError("ground requires --prompt.")

    prompt_path = Path(prompt_value)
    if prompt_path.suffix.lower() not in {".yaml", ".yml"}:
        prompts = _normalize_prompts([prompt_value])
        return _identity_prompt_spec(prompts)

    if not prompt_path.is_absolute():
        prompt_path = Path.cwd() / prompt_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    return _normalize_prompt_spec(data)


def parse_prompt_items(raw_prompt: str) -> List[str]:
    return parse_prompt_spec(raw_prompt).prompts


def _identity_prompt_spec(prompts: List[str]) -> GroundPromptSpec:
    return GroundPromptSpec(
        prompts=list(prompts),
        prompt_to_label={prompt: prompt for prompt in prompts},
        labels=list(prompts),
    )


def _normalize_prompt_spec(data: object) -> GroundPromptSpec:
    if isinstance(data, list):
        prompts = _normalize_prompts(data)
        return _identity_prompt_spec(prompts)

    if not isinstance(data, dict):
        raise ValueError("Ground prompt YAML must contain either a top-level list of strings or a label-to-prompts mapping.")

    prompts: List[str] = []
    prompt_to_label: Dict[str, str] = {}
    labels: List[str] = []
    seen_labels: set[str] = set()

    for raw_label, raw_prompts in data.items():
        if not isinstance(raw_label, str):
            raise ValueError("Ground prompt YAML label keys must be strings.")
        label = raw_label.strip()
        if not label:
            raise ValueError("Ground prompt YAML label keys must be non-empty strings.")
        if label not in seen_labels:
            seen_labels.add(label)
            labels.append(label)

        if isinstance(raw_prompts, str):
            values: Iterable[object] = [raw_prompts]
        elif isinstance(raw_prompts, list):
            values = raw_prompts
        else:
            raise ValueError("Ground prompt YAML label values must be strings or lists of strings.")

        label_prompts = _normalize_prompts(values)
        for prompt in label_prompts:
            existing_label = prompt_to_label.get(prompt)
            if existing_label is not None and existing_label != label:
                raise ValueError(f"Ground prompt {prompt!r} is assigned to multiple labels.")
            if existing_label is None:
                prompt_to_label[prompt] = label
                prompts.append(prompt)

    if not prompts:
        raise ValueError("Ground requires at least one non-empty prompt.")

    return GroundPromptSpec(prompts=prompts, prompt_to_label=prompt_to_label, labels=labels)


def _normalize_prompts(values: Iterable[object]) -> List[str]:
    prompts: List[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            raise ValueError("Ground prompts must be strings.")
        prompt = item.strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    if not prompts:
        raise ValueError("Ground requires at least one non-empty prompt.")
    return prompts


def seed_ground_prompts(dataset: VisionDataset, prompts: List[str]) -> None:
    for rec in dataset.records:
        rec.attributes[GROUND_PROMPTS_KEY] = list(prompts)


def _ground_model_args(provider_name: str, model_id: str | None, threshold: float | None) -> Dict[str, float | str]:
    args: Dict[str, float | str] = {}
    if provider_name == "sam3" and threshold is not None:
        args["confidence_threshold"] = float(threshold)

    selected_model_id = str(model_id or "").strip()
    if not selected_model_id:
        return args
    model_id_providers = {"llmdet", "locate_anything"}
    if provider_name not in model_id_providers:
        supported_providers = ", ".join(sorted(model_id_providers))
        raise ValueError(f"--model-id is only supported by provider(s): {supported_providers}.")

    allowed = allowed_model_ids_for_provider(provider_name, family="ground")
    if selected_model_id not in allowed:
        supported = ", ".join(allowed)
        raise ValueError(f"Unsupported {provider_name} --model-id {selected_model_id!r}. Supported model ids: {supported}.")
    args["model_id"] = selected_model_id
    return args


def snapshot_annotation_counts(dataset: VisionDataset) -> Dict[int, Tuple[int, int]]:
    return {
        idx: (len(rec.boxes), len(rec.polys))
        for idx, rec in enumerate(dataset.records)
    }


def run_ground(
    dataset: VisionDataset,
    *,
    model: str,
    prompts: List[str],
    threshold: float | None = None,
    device: str,
    precision: str,
    batch_size: int,
    config_path: Path | None,
    model_id: str | None = None,
    no_resume: bool = False,
) -> VisionDataset:
    device_pref = _normalize_device_pref(device)
    provider_name = str(model or "").strip().lower()
    if provider_name == "sam3" and device_pref == "cpu":
        raise ValueError("sam3 does not support CPU execution; use --device auto or --device cuda.")

    seed_ground_prompts(dataset, prompts)
    dataset.fm_request = FMRequest(
        task="ground",
        provider=provider_name,
        prompt="",
        config_path=config_path,
        device=device_pref,
        precision=precision,
        batch_size=batch_size,
        model_args=_ground_model_args(provider_name, model_id, threshold),
        meta={"ground_prompt_key": GROUND_PROMPTS_KEY, "provider_family": "ground"},
    )
    return FMRunner.from_dataset(dataset).run(dataset, no_resume=no_resume)


def run_ground_reference(
    dataset: VisionDataset,
    *,
    model: str,
    reference_folder: Path,
    threshold: float | None = None,
    device: str,
    precision: str,
    batch_size: int,
    config_path: Path | None,
    no_resume: bool = False,
) -> VisionDataset:
    from cvsuite.label.core.router import ingest

    provider_name = str(model or "").strip().lower()
    device_pref = _normalize_device_pref(device)

    reference = ingest(Path(reference_folder))
    if len(reference.records) != 1:
        raise ValueError(
            f"--reference-folder must contain exactly one annotated reference image; "
            f"found {len(reference.records)} in {reference_folder}."
        )
    ref_record = reference.records[0]
    if not ref_record.boxes:
        raise ValueError(f"Reference image {ref_record.image.path} has no bounding-box labels.")

    anchor = reference.root or Path.cwd()
    if not ref_record.image.path.is_absolute():
        ref_record.image.path = (Path(anchor) / ref_record.image.path).resolve()

    tmp_dir = Path(tempfile.mkdtemp(prefix="cvsuite-ground-ref-"))
    grounded = None
    try:
        reference_path = tmp_dir / "reference.json"
        reference.to_json(reference_path)
        model_args: Dict[str, float | str] = {"reference_dataset_path": str(reference_path)}
        if threshold is not None:
            model_args["confidence_threshold"] = float(threshold)
        dataset.fm_request = FMRequest(
            task="ground",
            provider=provider_name,
            prompt="",
            config_path=config_path,
            device=device_pref,
            precision=precision,
            batch_size=batch_size,
            model_args=model_args,
            meta={"provider_family": "ground", "reference_folder": str(reference_folder)},
        )
        grounded = FMRunner.from_dataset(dataset).run(dataset, no_resume=no_resume)
        return grounded
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def filter_new_annotations(
    dataset: VisionDataset,
    before_counts: Dict[int, Tuple[int, int]],
    threshold: float,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    prompt_to_label: Mapping[str, str] | None = None,
    base_classes: List[str] | None = None,
) -> None:
    prompt_map = dict(prompt_to_label or {})
    current_classes = list(dataset.classes)
    classes = list(base_classes) if base_classes is not None else list(dataset.classes)
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    for idx, rec in enumerate(dataset.records):
        box_start, poly_start = before_counts.get(idx, (0, 0))
        old_boxes = rec.boxes[:box_start]
        new_boxes = rec.boxes[box_start:]
        old_polys = rec.polys[:poly_start]
        new_polys = rec.polys[poly_start:]

        _remap_new_annotations(new_boxes, new_polys, prompt_map, current_classes)
        rejected_group_keys = _threshold_rejected_group_keys(new_boxes, new_polys, threshold)
        rejected_group_keys.update(_iou_rejected_group_keys(new_boxes, new_polys, rejected_group_keys, iou_threshold))

        kept_boxes = [
            box
            for box_idx, box in enumerate(new_boxes)
            if _keep_annotation(
                box.score,
                _annotation_group_key(box.group_id, "box", box_idx),
                rejected_group_keys,
                threshold,
            )
        ]
        kept_polys = [
            poly
            for poly_idx, poly in enumerate(new_polys)
            if _keep_annotation(
                poly.score,
                _annotation_group_key(poly.group_id, "poly", poly_idx),
                rejected_group_keys,
                threshold,
            )
        ]

        _assign_annotation_classes(kept_boxes, kept_polys, class_to_id, classes)
        rec.boxes = old_boxes + kept_boxes
        rec.polys = old_polys + kept_polys

    dataset.classes = classes
    _normalize_ground_tasks(dataset)


def clear_ground_prompts(dataset: VisionDataset) -> None:
    for rec in dataset.records:
        rec.attributes.pop(GROUND_PROMPTS_KEY, None)


def _keep_annotation(score: float | None, group_key: Hashable, rejected_group_keys: set[Hashable], threshold: float) -> bool:
    if group_key in rejected_group_keys:
        return False
    if score is None:
        return True
    return score >= threshold


def _annotation_group_key(group_id: int | None, prefix: str, idx: int) -> Hashable:
    if group_id is not None:
        return ("group", int(group_id))
    return (prefix, int(idx))


def _threshold_rejected_group_keys(
    boxes: List[BBox],
    polys: List[Polygon],
    threshold: float,
) -> set[Hashable]:
    rejected: set[Hashable] = set()
    for key, entry in _ground_detection_entries(boxes, polys).items():
        score = entry["score"]
        if score is not None and score < threshold:
            rejected.add(key)
    return rejected


def _iou_rejected_group_keys(
    boxes: List[BBox],
    polys: List[Polygon],
    already_rejected: set[Hashable],
    iou_threshold: float,
) -> set[Hashable]:
    if iou_threshold <= 0:
        return set()

    entries = _ground_detection_entries(boxes, polys)
    active = [
        {"key": key, "box": entry["box"], "score": entry["score"], "label": entry["label"]}
        for key, entry in entries.items()
        if key not in already_rejected and entry["box"] is not None
    ]
    if len(active) <= 1:
        return set()

    rejected: set[Hashable] = set()
    active_by_label: Dict[str, List[Dict[str, object]]] = {}
    for entry in active:
        active_by_label.setdefault(str(entry["label"] or ""), []).append(entry)

    for label_entries in active_by_label.values():
        if len(label_entries) <= 1:
            continue
        boxes_np = np.stack([entry["box"] for entry in label_entries])
        scores_np = np.array(
            [0.0 if entry["score"] is None else float(entry["score"]) for entry in label_entries],
            dtype=np.float32,
        )
        keep = set(fm_utils.nms_indices(boxes_np, scores_np, iou_thresh=float(iou_threshold)))
        rejected.update(entry["key"] for idx, entry in enumerate(label_entries) if idx not in keep)
    return rejected


def _ground_detection_entries(
    boxes: List[BBox],
    polys: List[Polygon],
) -> Dict[Hashable, Dict[str, object]]:
    entries: Dict[Hashable, Dict[str, object]] = {}

    for idx, box in enumerate(boxes):
        key = _annotation_group_key(box.group_id, "box", idx)
        _merge_detection_entry(
            entries,
            key,
            _bbox_xyxy(box),
            box.score,
            box.label,
            prefer_box=True,
        )

    for idx, poly in enumerate(polys):
        key = _annotation_group_key(poly.group_id, "poly", idx)
        _merge_detection_entry(
            entries,
            key,
            _poly_xyxy(poly),
            poly.score,
            poly.label,
            prefer_box=False,
        )

    return entries


def _merge_detection_entry(
    entries: Dict[Hashable, Dict[str, object]],
    key: Hashable,
    box_xyxy: np.ndarray,
    score: float | None,
    label: str | None,
    *,
    prefer_box: bool,
) -> None:
    score_value = None if score is None else float(score)
    current = entries.get(key)
    if current is None:
        entries[key] = {
            "box": box_xyxy,
            "score": score_value,
            "label": label,
            "has_box": prefer_box,
        }
        return

    if prefer_box and not bool(current["has_box"]):
        current["box"] = box_xyxy
        current["has_box"] = True
    elif current["box"] is None:
        current["box"] = box_xyxy

    current_score = current["score"]
    if score_value is not None and (current_score is None or float(score_value) > float(current_score)):
        current["score"] = score_value
    if label and not current.get("label"):
        current["label"] = label


def _bbox_xyxy(box: BBox) -> np.ndarray:
    half_w = float(box.w) / 2.0
    half_h = float(box.h) / 2.0
    return np.array(
        [
            float(box.cx) - half_w,
            float(box.cy) - half_h,
            float(box.cx) + half_w,
            float(box.cy) + half_h,
        ],
        dtype=np.float32,
    )


def _poly_xyxy(poly: Polygon) -> np.ndarray:
    if not poly.points:
        return np.zeros(4, dtype=np.float32)
    xs = [float(x) for x, _ in poly.points]
    ys = [float(y) for _, y in poly.points]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)


def dataset_ground_prompts(dataset: VisionDataset) -> List[str]:
    prompts: List[str] = []
    seen: set[str] = set()
    for rec in dataset.records:
        for item in rec.attributes.get(GROUND_PROMPTS_KEY, []):
            if isinstance(item, str) and item not in seen:
                seen.add(item)
                prompts.append(item)
    return prompts


def _annotation_prompt(annotation: BBox | Polygon, classes: List[str]) -> str | None:
    prompt = getattr(annotation, "prompt", None)
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    label = getattr(annotation, "label", None)
    if isinstance(label, str) and label.strip():
        return label.strip()
    cls_idx = int(getattr(annotation, "cls", -1))
    if 0 <= cls_idx < len(classes):
        name = str(classes[cls_idx]).strip()
        if name:
            return name
    return None


def _annotation_label(annotation: BBox | Polygon, classes: List[str]) -> str:
    label = getattr(annotation, "label", None)
    if isinstance(label, str) and label.strip():
        return label.strip()
    cls_idx = int(getattr(annotation, "cls", -1))
    if 0 <= cls_idx < len(classes):
        return str(classes[cls_idx])
    return str(getattr(annotation, "cls", ""))


def _remap_new_annotations(
    boxes: List[BBox],
    polys: List[Polygon],
    prompt_to_label: Mapping[str, str],
    classes: List[str],
) -> None:
    for annotation in [*boxes, *polys]:
        prompt = _annotation_prompt(annotation, classes)
        if prompt:
            annotation.prompt = prompt
        mapped_label = prompt_to_label.get(prompt or "", None)
        annotation.label = mapped_label or _annotation_label(annotation, classes)


def _assign_annotation_classes(
    boxes: List[BBox],
    polys: List[Polygon],
    class_to_id: Dict[str, int],
    classes: List[str],
) -> None:
    for annotation in [*boxes, *polys]:
        label = _annotation_label(annotation, classes)
        if label not in class_to_id:
            class_to_id[label] = len(classes)
            classes.append(label)
        annotation.label = label
        annotation.cls = class_to_id[label]


def _normalize_ground_tasks(dataset: VisionDataset) -> None:
    for rec in dataset.records:
        if rec.task not in {None, Task.det, Task.seg}:
            continue
        if rec.polys:
            rec.task = Task.seg
        elif rec.boxes:
            rec.task = Task.det
        else:
            rec.task = None

    if dataset.task not in {None, Task.det, Task.seg}:
        return
    if any(rec.polys for rec in dataset.records):
        dataset.task = Task.seg
    elif any(rec.boxes for rec in dataset.records):
        dataset.task = Task.det
    else:
        dataset.task = None
