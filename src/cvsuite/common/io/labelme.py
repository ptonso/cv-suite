"""Adapter for LabelMe image-plus-JSON annotations.

This module expects LabelMe-style JSON sidecars with `imagePath`, `shapes`, and
optional VQA or embedded image fields. In memory it reads and writes canonical
records populated with normalized boxes, polygons, keypoints, image metadata,
and round-trip LabelMe extras.

Example roots:
    dataset/
      image_001.jpg
      image_001.json

    dataset/
      train/
        image_001.jpg
        image_001.json
      val/
        image_101.jpg
        image_101.json
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from cvsuite.common.core import BBox, Keypoints, LabelMeShape, Polygon, VisionDataset, VisionRecord
from cvsuite.common.io import common


def _shape_from_payload(payload: dict[str, Any]) -> LabelMeShape:
    return LabelMeShape(
        label=str(payload.get("label", "")),
        shape_type=str(payload.get("shape_type", "polygon")),
        points=[(float(x), float(y)) for x, y in payload.get("points", [])],
        flags=dict(payload.get("flags") or {}),
        group_id=payload.get("group_id"),
        description=payload.get("description"),
        other_data=dict(payload.get("other_data") or {}),
    )


def _normalize_shape_type(shape: dict[str, Any]) -> str:
    shape_type = str(shape.get("shape_type", "")).lower()
    return "point" if shape_type == "points" else shape_type


def _resolve_image_path(json_path: Path, image_path_field: str, root: Path) -> Path | None:
    candidate = (json_path.parent / image_path_field).resolve()
    if candidate.exists():
        return candidate
    sibling_images = (json_path.parent.parent / "images" / Path(image_path_field).name).resolve()
    if sibling_images.exists():
        return sibling_images
    for base in (root, root.parent):
        hit = next(base.rglob(Path(image_path_field).name), None)
        if hit:
            return hit.resolve()
    return None


def _dataset_image_path(root: Path, image_abs: Path) -> Path:
    try:
        return image_abs.relative_to(root)
    except ValueError:
        return image_abs


def _infer_split(root: Path, json_path: Path, image_path: Path, payload: dict[str, Any]) -> str:
    if payload.get("split"):
        return common.canonical_split_id(payload["split"])
    try:
        rel = json_path.relative_to(root)
        if len(rel.parts) >= 2:
            return common.canonical_split_id(rel.parts[0])
    except ValueError:
        pass
    if not image_path.is_absolute() and len(image_path.parts) >= 2:
        return common.canonical_split_id(image_path.parts[0])
    return "train"


def _shape_metadata(shape: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    attrs = shape.get("attributes")
    if isinstance(attrs, dict) and attrs.get("score") is not None:
        meta["score"] = attrs["score"]
    nested = shape.get(common.LABELME_META_KEY)
    if isinstance(nested, dict):
        for key in common.LABELME_PROMOTED_FIELDS:
            if nested.get(key) is not None:
                meta[key] = nested[key]
    for key in common.LABELME_PROMOTED_FIELDS:
        if key not in meta and shape.get(key) is not None:
            meta[key] = shape[key]
    flags = shape.get("flags")
    if isinstance(flags, dict):
        for key in common.LABELME_PROMOTED_FIELDS:
            if key not in meta and key in flags and not isinstance(flags[key], bool):
                meta[key] = flags[key]
    return meta


def _shape_attributes(shape: dict[str, Any]) -> dict[str, Any]:
    raw = shape.get("attributes")
    attrs = dict(raw) if isinstance(raw, dict) else {}
    attrs.pop("score", None)
    attrs.pop("flags", None)
    for key in common.LABELME_PROMOTED_FIELDS:
        if key == "score":
            continue
        if key not in attrs and shape.get(key) is not None:
            attrs[key] = shape[key]
    nested = shape.get(common.LABELME_META_KEY)
    if isinstance(nested, dict):
        for key, value in nested.items():
            if key != "score" and key not in attrs and value is not None:
                attrs[key] = value
    meta = _shape_metadata(shape)
    for key in ("prompt", "model", "text"):
        if key not in attrs and meta.get(key) is not None:
            attrs[key] = meta[key]
    return attrs


def _shape_flags(shape: dict[str, Any]) -> dict[str, bool]:
    merged: dict[str, bool] = {}
    for raw in (shape.get("flags"), shape.get("attributes", {}).get("flags") if isinstance(shape.get("attributes"), dict) else None):
        if isinstance(raw, dict):
            merged.update({str(key): value for key, value in raw.items() if isinstance(value, bool)})
    return merged


def _annotation_flags(annotation: BBox | Polygon) -> dict[str, bool]:
    flags = annotation.attributes.get("flags", {})
    if not isinstance(flags, dict):
        raise TypeError("annotation.attributes['flags'] must be a dict[str, bool]")
    if any(not isinstance(key, str) or not isinstance(value, bool) for key, value in flags.items()):
        raise TypeError("annotation.attributes['flags'] must be a dict[str, bool]")
    return dict(flags)


def _parse_vqas(data: object) -> list:
    if not data:
        return []
    entries: list[dict[str, Any]] = []
    if isinstance(data, list):
        entries = [entry for entry in data if isinstance(entry, dict)]
    elif isinstance(data, dict):
        if isinstance(data.get("vqas"), list):
            entries = [entry for entry in data["vqas"] if isinstance(entry, dict)]
        elif "question" in data or "answer" in data:
            entries = [data]
    return [common.qa_from_payload(entry, source="labelme", question_keys=("question", "prompt")) for entry in entries]


def _base_label(label: str, *, shape_type: str) -> str:
    if shape_type == "point" and "-" in label:
        return label.split("-", 1)[0]
    return label


def _keypoint_order(path: Path | None) -> dict[str, dict[str, int]]:
    if path is None:
        return {}
    data = common.read_yaml_or_json(path)
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, int]] = {}
    for class_name, spec in data.items():
        if isinstance(spec, dict) and isinstance(spec.get("keypoints"), list):
            out[str(class_name)] = {str(name): idx for idx, name in enumerate(spec["keypoints"])}
    return out


def _pose_instances(shapes: list[dict[str, Any]], *, keypoints_file: Path | None) -> list[tuple[str, list[tuple[str, float, float]]]]:
    rects: list[tuple[str, tuple[float, float, float, float]]] = []
    points_all: list[tuple[str, float, float]] = []
    for shape in shapes:
        shape_type = _normalize_shape_type(shape)
        label = str(shape.get("label", ""))
        points = shape.get("points", [])
        if shape_type == "rectangle" and isinstance(points, list) and len(points) >= 2:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            rects.append((_base_label(label, shape_type=shape_type), (min(xs), min(ys), max(xs), max(ys))))
        elif shape_type == "point" and isinstance(points, list):
            for point in points:
                points_all.append((label, float(point[0]), float(point[1])))
    instances: list[tuple[str, list[tuple[str, float, float]]]] = []
    if rects:
        for cls_name, (x1, y1, x2, y2) in rects:
            pts = [(label, x, y) for label, x, y in points_all if x1 <= x <= x2 and y1 <= y <= y2]
            if pts:
                instances.append((cls_name, pts))
    else:
        buckets: dict[str, list[tuple[str, float, float]]] = {}
        for label, x, y in points_all:
            buckets.setdefault(_base_label(label, shape_type="point"), []).append((label, x, y))
        instances.extend((parent, pts) for parent, pts in buckets.items())

    ordering = _keypoint_order(keypoints_file)
    out: list[tuple[str, list[tuple[str, float, float]]]] = []
    for cls_name, pts in instances:
        order_map = ordering.get(cls_name, {})
        if order_map:
            pts = sorted(pts, key=lambda item: order_map.get(item[0], 10**9))
        out.append((cls_name, pts))
    return out


def _shape_key(shape: dict[str, Any]) -> tuple:
    pts = tuple((round(float(point[0]), 3), round(float(point[1]), 3)) for point in shape.get("points", []))
    return (str(shape.get("shape_type", "")), str(shape.get("label", "")), pts)


def _annotation_metadata(annotation: BBox | Polygon) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if annotation.score is not None:
        out["score"] = float(annotation.score)
    return out


def _annotation_attributes(annotation: BBox | Polygon) -> dict[str, Any]:
    out = {key: value for key, value in annotation.attributes.items() if key != "flags" and value is not None}
    for key in ("prompt", "model", "text"):
        value = getattr(annotation, key, None)
        if value is not None:
            out[key] = value
    return out


def _merge_shape_metadata(shape: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    out = dict(shape)
    for key, value in metadata.items():
        if value is not None:
            out[key] = value
    return out


def _round_trip_shape(shape: dict[str, Any]) -> dict[str, Any]:
    out = dict(shape)
    for key in ("prompt", "model", "text", common.LABELME_META_KEY):
        out.pop(key, None)
    flags = out.get("flags")
    if isinstance(flags, dict):
        out["flags"] = {str(key): value for key, value in flags.items() if isinstance(value, bool)}
    attrs = out.get("attributes")
    if isinstance(attrs, dict):
        clean_attrs = {key: value for key, value in attrs.items() if key not in {"flags", "score"}}
        if clean_attrs:
            out["attributes"] = clean_attrs
        else:
            out.pop("attributes", None)
    return out


def _record_split(record: VisionRecord) -> str:
    return common.canonical_split_id(record.split or "train")


def _uses_split_dirs(records: list[VisionRecord], split_layout: object) -> bool:
    if split_layout is not None and not isinstance(split_layout, bool):
        raise TypeError("split_layout must be bool | None.")
    non_train = sorted({_record_split(record) for record in records if _record_split(record) != "train"})
    if split_layout is False and non_train:
        raise ValueError("split_layout=False requires train-only LabelMe records.")
    return bool(split_layout) if split_layout is not None else bool(non_train)


class LabelMeAdapter:
    @classmethod
    def matches(cls, src: Path) -> bool:
        candidates = [src] if src.is_file() and src.suffix.lower() == ".json" else list(src.rglob("*.json")) if src.is_dir() else []
        for path in candidates:
            try:
                payload = common.read_json(path)
            except Exception:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("shapes"), list):
                return True
        return False

    @classmethod
    def read(cls, src: Path, *, task: str = "auto", splits: list[str] | None = None, **options: object) -> VisionDataset:
        root = src.resolve() if src.is_dir() else src.resolve().parent
        keypoints_file = options.get("keypoints_file")
        keypoints_path = Path(keypoints_file) if isinstance(keypoints_file, (str, Path)) else None
        json_paths = [src.resolve()] if src.is_file() else sorted(src.rglob("*.json"))

        raw_records: list[dict[str, Any]] = []
        class_names: list[str] = []
        seen_classes: set[str] = set()
        for json_path in json_paths:
            payload = common.read_json(json_path)
            if not isinstance(payload, dict):
                raise ValueError(f"LabelMe payload must be a JSON object: {json_path}")
            image_path_field = payload.get("imagePath")
            if image_path_field is None:
                raise ValueError(f"LabelMe JSON is missing imagePath: {json_path}")
            image_abs = _resolve_image_path(json_path, str(image_path_field), root)
            if image_abs is None:
                continue
            width = payload.get("imageWidth")
            height = payload.get("imageHeight")
            if not width or not height:
                image_info = common.image_info_from_path(image_abs)
                width, height = image_info.width, image_info.height
            image = common.image_info_from_path(image_abs)
            image.path = _dataset_image_path(root, image_abs)
            shapes = [shape for shape in payload.get("shapes", []) if isinstance(shape, dict)]
            for shape in shapes:
                label = str(shape.get("label", ""))
                if not label:
                    continue
                base = _base_label(label, shape_type=_normalize_shape_type(shape))
                if base not in seen_classes:
                    seen_classes.add(base)
                    class_names.append(base)
            split_guess = _infer_split(root, json_path, image.path, payload)
            record_attrs = payload.get("attributes") if isinstance(payload.get("attributes"), dict) else {}
            raw_records.append({"image": image, "width": int(width), "height": int(height), "split": common.canonical_split_id(split_guess), "shapes": shapes, "imageData": payload.get("imageData"), "vqas": _parse_vqas(payload.get("vqas")), "attributes": dict(record_attrs)})

        records: list[VisionRecord] = []
        for raw in raw_records:
            image = raw["image"]
            image.width = raw["width"]
            image.height = raw["height"]
            boxes: list[BBox] = []
            polys: list[Polygon] = []
            for shape in raw["shapes"]:
                shape_type = _normalize_shape_type(shape)
                label = str(shape.get("label", ""))
                base = _base_label(label, shape_type=shape_type)
                cls_idx = class_names.index(base) if base in class_names else 0
                meta = _shape_metadata(shape)
                flags = _shape_flags(shape)
                attributes = _shape_attributes(shape)
                if flags:
                    attributes["flags"] = flags
                overlay = {key: attributes.pop(key) for key in ("prompt", "model", "text") if key in attributes}
                group_id = shape.get("group_id")
                points = shape.get("points", [])
                if shape_type == "rectangle" and isinstance(points, list) and len(points) >= 2:
                    xs = [float(point[0]) for point in points]
                    ys = [float(point[1]) for point in points]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    boxes.append(BBox(cx=(x1 + x2) / (2 * image.width), cy=(y1 + y2) / (2 * image.height), w=(x2 - x1) / image.width, h=(y2 - y1) / image.height, cls=cls_idx, label=base, score=float(meta["score"]) if meta.get("score") is not None else None, group_id=group_id, attributes=attributes, **overlay))
                elif shape_type == "polygon" and isinstance(points, list) and len(points) >= 3:
                    pts = [(float(x) / image.width, float(y) / image.height) for x, y in points]
                    polys.append(Polygon(points=pts, cls=cls_idx, label=base, score=float(meta["score"]) if meta.get("score") is not None else None, group_id=group_id, attributes=attributes, **{key: value for key, value in overlay.items() if key != "text"}))

            kpts: list[Keypoints] = []
            for cls_name, pts in _pose_instances(raw["shapes"], keypoints_file=keypoints_path):
                if not pts:
                    continue
                cls_idx = class_names.index(cls_name) if cls_name in class_names else 0
                xs = [x for _label, x, _y in pts]
                ys = [y for _label, _x, y in pts]
                boxes.append(BBox(cx=(min(xs) + max(xs)) / (2 * image.width), cy=(min(ys) + max(ys)) / (2 * image.height), w=(max(xs) - min(xs)) / image.width, h=(max(ys) - min(ys)) / image.height, cls=cls_idx, label=cls_name, kind="pose"))
                kpts.append(Keypoints(points=[(x / image.width, y / image.height, 2.0) for _label, x, y in pts], cls=cls_idx, label=cls_name, kind="pose"))

            records.append(VisionRecord(image=image, split=raw["split"], boxes=boxes, polys=polys, kpts=kpts, rel_image_path=image.path, vqas=raw["vqas"], attributes=raw["attributes"], labelme_shapes=[_shape_from_payload(shape) for shape in raw["shapes"] if isinstance(shape, dict)], round_trip={"labelme": {"shapes": raw["shapes"], "imageData": raw["imageData"]}}))

        dataset = VisionDataset(records=records, classes=class_names, root=root)
        if splits:
            allowed = {common.canonical_split_id(item) for item in splits}
            dataset.records = [record for record in dataset.records if record.split in allowed]
        return dataset

    @classmethod
    def write(
        cls,
        dataset: VisionDataset,
        dst: Path,
        *,
        splits: list[str] | None = None,
        embed_image: bool = False,
        hardlink: bool = False,
        **options: object,
    ) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        filtered = dataset.records
        if splits:
            allowed = {common.canonical_split_id(item) for item in splits}
            filtered = [record for record in dataset.records if record.split in allowed]
        use_split_dirs = _uses_split_dirs(filtered, options.get("split_layout"))
        if not use_split_dirs and any((dst / name).exists() for name in ("train", "val", "valid", "validation", "test", "infer")):
            raise FileExistsError(f"Flat LabelMe output would mix with an existing split directory under {dst}.")

        for record in filtered:
            out_dir = dst / _record_split(record) if use_split_dirs else dst
            out_dir.mkdir(parents=True, exist_ok=True)
            src = common.resolve_record_image_path(dataset, record)
            stem = src.stem
            suffix = src.suffix
            counter = 0
            while True:
                name = f"{stem}_{counter}" if counter else stem
                img_dst = out_dir / f"{name}{suffix}"
                json_dst = out_dir / f"{name}.json"
                if not img_dst.exists() and not json_dst.exists():
                    break
                counter += 1

            image_data: str | None = None
            if embed_image:
                try:
                    image_data = base64.b64encode(src.read_bytes()).decode("utf-8")
                except FileNotFoundError:
                    image_data = record.round_trip.get("labelme", {}).get("imageData")
            else:
                common.copy_or_link(src, img_dst, hardlink=hardlink)

            original_shapes = list(record.round_trip.get("labelme", {}).get("shapes", []))
            merged_shapes: list[dict[str, Any]] = []
            seen: dict[tuple, int] = {}
            for shape in original_shapes:
                shape = _round_trip_shape(shape)
                key = _shape_key(shape)
                seen[key] = len(merged_shapes)
                merged_shapes.append(shape)

            def append_shape(shape: dict[str, Any]) -> None:
                key = _shape_key(shape)
                idx = seen.get(key)
                if idx is None:
                    seen[key] = len(merged_shapes)
                    merged_shapes.append(shape)
                    return
                current = dict(merged_shapes[idx])
                for field in ("group_id", "description"):
                    if current.get(field) is None and shape.get(field) is not None:
                        current[field] = shape[field]
                current_flags = current.get("flags") if isinstance(current.get("flags"), dict) else {}
                new_flags = shape.get("flags") if isinstance(shape.get("flags"), dict) else {}
                if new_flags:
                    current["flags"] = {**current_flags, **new_flags}
                current_attrs = current.get("attributes") if isinstance(current.get("attributes"), dict) else {}
                new_attrs = shape.get("attributes") if isinstance(shape.get("attributes"), dict) else {}
                if new_attrs:
                    current["attributes"] = {**current_attrs, **new_attrs}
                current.update({key: value for key, value in shape.items() if key not in {"points", "label", "shape_type", "flags", "attributes"} and value is not None})
                merged_shapes[idx] = current

            for box in record.boxes:
                width = box.w * record.image.width
                height = box.h * record.image.height
                cx = box.cx * record.image.width
                cy = box.cy * record.image.height
                x1 = cx - width / 2
                y1 = cy - height / 2
                x2 = cx + width / 2
                y2 = cy + height / 2
                label = dataset.classes[box.cls] if 0 <= box.cls < len(dataset.classes) else (box.label or str(box.cls))
                shape = _merge_shape_metadata({"label": label, "points": [[x1, y1], [x2, y2]], "group_id": box.group_id, "shape_type": "rectangle", "flags": _annotation_flags(box)}, _annotation_metadata(box))
                attrs = _annotation_attributes(box)
                if attrs:
                    shape["attributes"] = attrs
                append_shape(shape)

            for poly in record.polys:
                label = dataset.classes[poly.cls] if 0 <= poly.cls < len(dataset.classes) else (poly.label or str(poly.cls))
                shape = _merge_shape_metadata({"label": label, "points": [[x * record.image.width, y * record.image.height] for x, y in poly.points], "group_id": poly.group_id, "shape_type": "polygon", "flags": _annotation_flags(poly)}, _annotation_metadata(poly))
                attrs = _annotation_attributes(poly)
                if attrs:
                    shape["attributes"] = attrs
                append_shape(shape)

            for keypoints in record.kpts:
                label = dataset.classes[keypoints.cls] if 0 <= keypoints.cls < len(dataset.classes) else (keypoints.label or str(keypoints.cls))
                for x, y, _vis in keypoints.points:
                    append_shape({"label": label, "points": [[x * record.image.width, y * record.image.height]], "group_id": keypoints.group_id, "shape_type": "point", "flags": {}})

            payload: dict[str, Any] = {
                "version": "5.8.3",
                "flags": {},
                "shapes": merged_shapes,
                "imagePath": img_dst.name if not embed_image else src.name,
                "imageData": image_data,
                "imageHeight": record.image.height,
                "imageWidth": record.image.width,
            }
            if record.vqas:
                payload["vqas"] = [common.qa_to_payload(vqa, question_key="question", id_key="question_id") for vqa in record.vqas]
            if record.attributes:
                payload["attributes"] = dict(record.attributes)
            json_dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
