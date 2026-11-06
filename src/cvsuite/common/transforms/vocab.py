"""Rewrite everything that names a class: shared by `select` and `remap-classes`. Contract: `specs/common/select_remap.md`."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from cvsuite.common.core import Classification, VisionDataset, VisionRecord
from cvsuite.common.io.common import CLASS_DIR_MULTI_LABELS_ATTR

CLASSIFY_MULTI_ATTR = "classify_multi_class_labels"
MULTI_ATTRS = (CLASS_DIR_MULTI_LABELS_ATTR, CLASSIFY_MULTI_ATTR)
EMPTY_CHOICES = ("drop", "keep")


def has_geometry(dataset: VisionDataset) -> bool:
    return any(r.boxes or r.polys or r.kpts for r in dataset.records)


def reject_masks(dataset: VisionDataset, what: str) -> None:
    if any(r.semantic_mask is not None for r in dataset.records):
        raise ValueError(f"{what} does not support datasets with semantic masks.")


def require_empty(dataset: VisionDataset, empty: str | None, what: str) -> None:
    if empty is not None and empty not in EMPTY_CHOICES:
        raise ValueError(f"--empty takes one of {', '.join(EMPTY_CHOICES)}, got {empty!r}.")
    if empty is None and has_geometry(dataset):
        raise SystemExit(f"{what} can leave images without annotations: pass --empty drop or --empty keep.")


def annotation_name(item, classes: list[str]) -> str:
    if 0 <= item.cls < len(classes):
        return classes[item.cls]
    if item.label:
        return str(item.label)
    raise ValueError(f"Annotation with class id {item.cls} has no name and the class list holds {len(classes)} entries.")


def _label_names(record: VisionRecord) -> list[str]:
    names: list[str] = []
    if record.classification is not None and record.classification.label:
        names.append(str(record.classification.label))
    for attr in MULTI_ATTRS:
        names.extend(str(x) for x in record.attributes.get(attr) or [])
    if record.classification is not None:
        names.extend(str(k) for k in record.classification.probs)
    return names


def check_names(dataset: VisionDataset, names: list[str], flag: str) -> None:
    known = set(dataset.classes)
    if not known:
        known = {n for r in dataset.records for n in _label_names(r)}
    missing = [n for n in names if n not in known]
    if missing:
        raise ValueError(f"{flag} names classes that the dataset does not have: {', '.join(missing)}. Known classes: {', '.join(sorted(known)) or 'none'}.")


def _map_names(names: list[str], name_map: Mapping[str, str | None]) -> list[str]:
    out: list[str] = []
    for name in names:
        new = name_map[name] if name in name_map else name
        if new is not None and new not in out:
            out.append(new)
    return out


def _map_classification(record: VisionRecord, name_map: Mapping[str, str | None]) -> tuple[Classification | None, dict] | None:
    """The record's classification and attributes after the map, or None when no label is left."""
    attrs = dict(record.attributes)
    multi: list[str] | None = None
    for attr in MULTI_ATTRS:
        if isinstance(attrs.get(attr), list):
            attrs[attr] = _map_names([str(x) for x in attrs[attr]], name_map)
            multi = attrs[attr]
    if multi is not None and not multi:
        return None
    info = record.classification
    if info is None:
        return None if multi == [] else (None, attrs)
    probs: dict[str, float] = {}
    for key, value in (info.probs or {}).items():
        new = name_map.get(str(key), str(key))
        if new is not None:
            probs[new] = probs.get(new, 0.0) + float(value)
    label = info.label
    if label is not None:
        label = name_map.get(str(label), str(label))
    if label is None or (probs and label not in probs):
        if probs:
            label = max(probs.items(), key=lambda kv: kv[1])[0]
        elif multi:
            label = multi[0]
        else:
            return None
    score = probs[label] if probs else info.score
    return replace(info, label=label, probs=probs, score=score), attrs


def _shape_label(shape) -> str:
    return str(shape.get("label", "") if isinstance(shape, dict) else shape.label)


def _remap_shapes(shapes: list, name_map: Mapping[str, str | None]) -> list:
    out = []
    for shape in shapes:
        old = _shape_label(shape)
        if old not in name_map:
            out.append(shape)
        elif name_map[old] is not None:
            out.append({**shape, "label": name_map[old]} if isinstance(shape, dict) else replace(shape, label=name_map[old]))
    return out


def reindex_classes(dataset: VisionDataset, name_map: Mapping[str, str | None], new_classes: list[str], *, empty: str | None = None) -> tuple[VisionDataset, dict]:
    """Apply `name_map` (old name to new name, or None to drop) to every carrier of class names.

    Classes absent from the map keep their name and must be listed in `new_classes`. Returns the new dataset
    and counts of dropped annotations and records.
    """
    reject_masks(dataset, "Changing classes")
    if len(set(new_classes)) != len(new_classes):
        raise ValueError(f"Class names must be unique, got {new_classes}.")
    index = {name: i for i, name in enumerate(new_classes)}
    old = list(dataset.classes)
    full_map = {name: name_map.get(name, name) for name in old}
    stats = {"annotations_removed": 0, "records_emptied": 0, "records_dropped": 0}

    records: list[VisionRecord] = []
    for record in dataset.records:
        had_geometry = bool(record.boxes or record.polys or record.kpts)
        new_items: dict[str, list] = {}
        for field in ("boxes", "polys", "kpts"):
            kept = []
            for item in getattr(record, field):
                new = full_map.get(annotation_name(item, old))
                if new is None:
                    stats["annotations_removed"] += 1
                    continue
                kept.append(replace(item, cls=index[new], label=new))
            new_items[field] = kept
        classification, attrs = record.classification, record.attributes
        if record.classification is not None or any(isinstance(record.attributes.get(a), list) for a in MULTI_ATTRS):
            mapped = _map_classification(record, full_map)
            if mapped is None:
                stats["records_dropped"] += 1
                continue
            classification, attrs = mapped
        emptied = had_geometry and not any(new_items.values())
        stats["records_emptied"] += emptied
        if emptied and empty == "drop":
            stats["records_dropped"] += 1
            continue
        round_trip = record.round_trip
        if "labelme" in round_trip:
            labelme = dict(round_trip["labelme"])
            labelme["shapes"] = _remap_shapes(list(labelme.get("shapes", [])), full_map)
            round_trip = {**round_trip, "labelme": labelme}
        records.append(
            replace(
                record,
                classification=classification,
                attributes=attrs,
                round_trip=round_trip,
                labelme_shapes=_remap_shapes(list(record.labelme_shapes), full_map),
                **new_items,
            )
        )

    if not records:
        raise SystemExit("No records are left after changing the classes.")
    round_trip = {k: v for k, v in dataset.round_trip.items() if k != "coco"}
    if "coco" in dataset.round_trip:
        round_trip["coco"] = {k: v for k, v in dataset.round_trip["coco"].items() if k != "categories"}
    out = replace(dataset, records=records, classes=list(new_classes), round_trip=round_trip)
    check_consistency(out)
    return out, stats


def check_consistency(dataset: VisionDataset) -> None:
    if len(set(dataset.classes)) != len(dataset.classes):
        raise AssertionError(f"Class names are not unique: {dataset.classes}")
    for record in dataset.records:
        for item in [*record.boxes, *record.polys, *record.kpts]:
            if not 0 <= item.cls < len(dataset.classes):
                raise AssertionError(f"Class id {item.cls} is outside the {len(dataset.classes)} classes.")
