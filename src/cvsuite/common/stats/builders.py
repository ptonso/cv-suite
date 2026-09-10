from __future__ import annotations

from collections import Counter
from typing import Any

from cvsuite.classify.core import io as classify_io
from cvsuite.common.stats.utils import (
    ExportStatsContext,
    base_stats_document,
    image_key,
    label_name,
    score_summary,
    task_name,
)
from cvsuite.vlm.core import io as vlm_io


def build_classify_stats(dataset, ctx: ExportStatsContext) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    unlabeled_name = str(ctx.run_meta.get("unlabeled_name") or "unlabeled")
    multi_class = bool(ctx.run_meta.get("multi_class", False))
    threshold = ctx.run_meta.get("threshold")
    class_thresholds = dict(ctx.run_meta.get("class_thresholds") or {})

    assignments: Counter[str] = Counter()
    unlabeled_records = 0
    multi_class_records = 0
    for record in dataset.records:
        labels = classify_io.resolve_output_labels(
            dataset,
            record,
            threshold=threshold,
            unlabeled_name=unlabeled_name,
            multi_class=multi_class,
            class_thresholds=class_thresholds,
        )
        assignments.update(labels)
        if labels == [unlabeled_name]:
            unlabeled_records += 1
        if multi_class and len(labels) > 1:
            multi_class_records += 1

    summary["output_assignments_total"] = sum(assignments.values())
    summary["assignments_by_output_label"] = dict(sorted(assignments.items()))
    summary["unlabeled_records"] = unlabeled_records
    summary["confidence"] = score_summary(
        None if record.classification is None else record.classification.score for record in dataset.records
    )
    summary["multi_class_enabled"] = multi_class
    if multi_class:
        summary["multi_class_records"] = multi_class_records

    thresholds_applied: dict[str, Any] = {}
    if threshold is not None:
        thresholds_applied["default"] = float(threshold)
    if class_thresholds:
        thresholds_applied["per_class"] = dict(sorted((str(key), float(value)) for key, value in class_thresholds.items()))
    if unlabeled_name != "unlabeled":
        thresholds_applied["unlabeled_name"] = unlabeled_name
    if thresholds_applied:
        summary["thresholds_applied"] = thresholds_applied

    doc = base_stats_document(dataset, ctx)
    doc["summary"] = summary
    return doc


def build_label_stats(dataset, ctx: ExportStatsContext) -> dict[str, Any]:
    doc = base_stats_document(dataset, ctx)
    doc["dataset"]["task"] = task_name(dataset.task)

    classes = list(dataset.classes)
    annotated_images = 0
    empty_images = 0

    boxes_per_label: Counter[str] = Counter()
    polygons_per_label: Counter[str] = Counter()
    poses_per_label: Counter[str] = Counter()
    prompts_per_label: Counter[str] = Counter()
    box_scores: list[float | None] = []
    boxes_total = 0
    polygons_total = 0
    poses_total = 0
    keypoints_total = 0
    ocr_text_boxes = 0
    ocr_nonempty_text_boxes = 0

    for record in dataset.records:
        has_annotations = bool(record.boxes or record.polys or record.kpts)
        if has_annotations:
            annotated_images += 1
        else:
            empty_images += 1

        for box in record.boxes:
            boxes_total += 1
            name = label_name(box.label, box.cls, classes)
            boxes_per_label[name] += 1
            box_scores.append(box.score)
            if box.kind == "ocr":
                ocr_text_boxes += 1
                if str(box.text or "").strip():
                    ocr_nonempty_text_boxes += 1
            if str(box.prompt or "").strip():
                prompts_per_label[name] += 1

        for poly in record.polys:
            polygons_total += 1
            name = label_name(poly.label, poly.cls, classes)
            polygons_per_label[name] += 1
            if str(poly.prompt or "").strip():
                prompts_per_label[name] += 1

        for kpt in record.kpts:
            poses_total += 1
            keypoints_total += len(kpt.points)
            name = label_name(kpt.label, kpt.cls, classes)
            poses_per_label[name] += 1

    summary: dict[str, Any] = {
        "annotated_images": annotated_images,
        "empty_images": empty_images,
        "boxes_total": boxes_total,
        "boxes_per_label": dict(sorted(boxes_per_label.items())),
        "box_score_summary": score_summary(box_scores),
        "polygons_total": polygons_total,
        "polygons_per_label": dict(sorted(polygons_per_label.items())),
        "poses_total": poses_total,
        "poses_per_label": dict(sorted(poses_per_label.items())),
        "keypoints_total": keypoints_total,
        "ocr_text_boxes": ocr_text_boxes,
        "ocr_nonempty_text_boxes": ocr_nonempty_text_boxes,
        "prompts_per_label": dict(sorted(prompts_per_label.items())),
    }
    if ctx.writer_details:
        summary.update(ctx.writer_details)

    doc["summary"] = summary
    return doc


def build_vlm_stats(dataset, ctx: ExportStatsContext) -> dict[str, Any]:
    doc = base_stats_document(dataset, ctx)
    unique_images = len({image_key(record) for record in dataset.records})
    question_labels: Counter[str] = Counter()
    pairs_by_split: Counter[str] = Counter()
    mapped_answer_buckets: Counter[str] = Counter()
    qa_pairs_total = 0
    answered_pairs = 0

    for record in dataset.records:
        bucket = str(record.attributes.get(vlm_io.ANSWER_BUCKET_ATTR) or "").strip()
        if bucket:
            mapped_answer_buckets[bucket] += 1
        split_name = str(record.split or "train")
        for qa in record.vqas:
            qa_pairs_total += 1
            pairs_by_split[split_name] += 1
            if str(qa.answer or "").strip():
                answered_pairs += 1
            label = None
            if isinstance(qa.meta, dict):
                label = qa.meta.get("label")
            text = str(label or "").strip()
            if text:
                question_labels[text] += 1

    summary: dict[str, Any] = {
        "qa_pairs_total": qa_pairs_total,
        "answered_pairs": answered_pairs,
        "unanswered_pairs": qa_pairs_total - answered_pairs,
        "pairs_by_split": dict(sorted(pairs_by_split.items())),
        "pairs_by_question_label": dict(sorted(question_labels.items())),
        "mapped_answer_buckets": dict(sorted(mapped_answer_buckets.items())),
    }
    if "partitions_written" in ctx.writer_details:
        summary["partitions_written"] = int(ctx.writer_details["partitions_written"])
    if "shards_written" in ctx.writer_details:
        summary["shards_written"] = int(ctx.writer_details["shards_written"])

    doc["dataset"]["unique_images"] = unique_images
    doc["summary"] = summary
    return doc


def build_gen_stats(dataset, ctx: ExportStatsContext) -> dict[str, Any]:
    doc = base_stats_document(dataset, ctx)
    summary: dict[str, Any] = {
        "generated_images": len(dataset.records),
        "mode": str(dataset.meta.get("gen_mode") or "result"),
    }
    prompt_count = dataset.meta.get("gen_prompt_count")
    if prompt_count is not None:
        summary["prompt_count"] = int(prompt_count)
    fanout_mode = str(dataset.meta.get("gen_fanout_mode") or "").strip()
    if fanout_mode:
        summary["fanout_mode"] = fanout_mode
    if dataset.records:
        image = dataset.records[0].image
        summary["image_size"] = {"width": int(image.width), "height": int(image.height)}

    source_format = dataset.meta.get("source_format") or dataset.meta.get("gen_source_format")
    if source_format:
        summary["source"] = {
            "format": str(source_format),
            "records": int(dataset.meta.get("source_records", 0)),
            "boxes_total": int(dataset.meta.get("source_boxes_total", 0)),
            "polygons_total": int(dataset.meta.get("source_polygons_total", 0)),
            "keypoints_total": int(dataset.meta.get("source_keypoints_total", 0)),
        }

    doc["summary"] = summary
    return doc
