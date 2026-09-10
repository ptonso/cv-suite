from __future__ import annotations

import argparse

from cvsuite.common.core import VisionDataset
from cvsuite.common.core import copy_dataset
from cvsuite.vlm.core import io


def attach_split_filter_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--splits",
        default="",
        help="Optional comma-separated splits to include. Prefix matching is supported, so train matches train2014.",
    )


def filter_dataset_by_splits(dataset: VisionDataset, raw_splits: str) -> VisionDataset:
    requested = [item.strip() for item in str(raw_splits or "").split(",") if item.strip()]
    if not requested:
        return dataset

    requested_lower = [item.lower() for item in requested]

    def _matches(actual: str) -> bool:
        actual_lower = actual.lower()
        return any(actual_lower == token or actual_lower.startswith(token) for token in requested_lower)

    filtered = [record for record in dataset.records if _matches(record.split or "train")]
    if not filtered:
        available = sorted({record.split or "train" for record in dataset.records})
        requested_text = ", ".join(requested)
        available_text = ", ".join(available) if available else "(none)"
        raise SystemExit(f"No records matched --splits {requested_text!r}. Available splits: {available_text}")

    meta = dict(dataset.meta)
    meta["selected_splits"] = requested
    filtered_dataset = copy_dataset(
        dataset,
        records=filtered,
        classes=list(dataset.classes),
        task=dataset.task,
        root=dataset.root,
        meta=meta,
    )
    filtered_dataset.fm_request = dataset.fm_request
    return filtered_dataset


def answer_partition_labels(dataset: VisionDataset) -> list[str]:
    raw = dataset.meta.get(io.ANSWER_BUCKETS_META, [])
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        label = str(item or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def has_answer_partitions(dataset: VisionDataset) -> bool:
    return bool(answer_partition_labels(dataset))


def ensure_unpartitioned(dataset: VisionDataset, *, command_name: str) -> None:
    if not has_answer_partitions(dataset):
        return
    raise SystemExit(
        f"`cvsuite vlm {command_name}` does not support partitioned datasets. "
        "Use to-json, to-vqa-style, or to-shards instead."
    )


def iter_answer_partitions(dataset: VisionDataset) -> list[tuple[str, VisionDataset]]:
    labels = answer_partition_labels(dataset)
    if not labels:
        return []

    grouped: dict[str, list] = {label: [] for label in labels}
    for record in dataset.records:
        bucket = str(record.attributes.get(io.ANSWER_BUCKET_ATTR) or "").strip()
        if not bucket:
            raise SystemExit("Partitioned VLM dataset is missing `vlm_answer_bucket` on at least one record.")
        if bucket not in grouped:
            raise SystemExit(f"Partitioned VLM dataset references unknown answer bucket {bucket!r}.")
        grouped[bucket].append(record)

    meta = dict(dataset.meta)
    meta.pop(io.ANSWER_BUCKETS_META, None)
    partitions: list[tuple[str, VisionDataset]] = []
    for label in labels:
        partitions.append(
            (
                label,
                copy_dataset(
                    dataset,
                    records=list(grouped[label]),
                    classes=list(dataset.classes),
                    task=dataset.task,
                    root=dataset.root,
                    meta=dict(meta),
                ),
            )
        )
        partitions[-1][1].fm_request = dataset.fm_request
    return partitions
