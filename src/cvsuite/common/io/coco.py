"""Adapter for COCO manifests and COCO-style split layouts.

This module expects COCO JSON objects or YAML-driven COCO dataset layouts with
`images`, `annotations`, and `categories`. In memory it works with canonical
records whose detection or instance-segmentation geometry is already normalized
and whose per-image VQA extras live on the record.

Example root:
    dataset/
      images/
        sample.jpg
      annotations.json

Example YAML-driven root:
    dataset/
      data.yaml
      images/
        train/
      annotations/
        train.json
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from cvsuite.common.core import BBox, Polygon, VisionDataset, VisionRecord
from cvsuite.common.io import common


def _read_json(path: Path) -> dict[str, Any]:
    data = common.read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"COCO payload must be a JSON object: {path}")
    return data


def _resolve_root(path: Path, data: dict[str, Any]) -> Path:
    base = path.parent
    root_field = data.get("path")
    if not root_field:
        return base
    root_path = Path(str(root_field))
    if root_path.is_absolute():
        return root_path
    candidate = (base / root_path).resolve()
    return candidate if candidate.exists() else base


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
    return [common.qa_from_payload(entry, source="coco", question_keys=("question", "prompt")) for entry in entries]


def _vqa_map(coco: dict[str, Any]) -> dict[int, list]:
    out: dict[int, list] = {}
    for item in coco.get("vqas", []) or []:
        if not isinstance(item, dict):
            continue
        image_id = item.get("image_id")
        if image_id is None:
            continue
        vqas = _parse_vqas(item)
        if vqas:
            out.setdefault(int(image_id), []).extend(vqas)
    for image in coco.get("images", []) or []:
        if not isinstance(image, dict):
            continue
        image_id = image.get("id")
        if image_id is None:
            continue
        vqas = _parse_vqas(image.get("vqas"))
        if vqas:
            out.setdefault(int(image_id), []).extend(vqas)
    return out


def _load_if_coco_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = _read_json(path)
    except Exception:
        return None
    if "images" in data and "annotations" in data:
        return data
    return None


def _find_coco_jsons(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    prioritized = list(root.rglob("*_annotations.coco.json"))
    candidates = prioritized or list(root.rglob("*.json"))
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(candidates):
        data = _load_if_coco_json(path)
        if data is not None:
            found.append((path, data))
    return found


def _infer_split(coco_path: Path, root: Path, default: str) -> str:
    candidates: list[str] = []
    try:
        rel_parts = list(coco_path.relative_to(root).parts)
        candidates.extend([part for part in rel_parts[:-1] if part])
    except Exception:
        pass
    parent = coco_path.parent
    if parent != root and parent.name:
        candidates.append(parent.name)
    stem = coco_path.stem.lower()
    for token in stem.replace(".", "_").split("_"):
        if token in {"train", "val", "valid", "validation", "test", "eval"}:
            candidates.append(token)
    for candidate in candidates:
        low = candidate.lower()
        if low in {"train", "val", "valid", "validation", "test", "eval"}:
            return common.canonical_split_id(low, default=default)
    return default


def _resolve_image_path(coco_path: Path, file_name: str, root: Path | None) -> Path:
    candidates = [
        coco_path.parent / file_name,
        coco_path.parent.parent / "images" / file_name,
    ]
    if root is not None:
        candidates.extend([root / "images" / file_name, root / file_name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _ingest_from_obj(path: Path, coco: dict[str, Any], *, task: str, split: str, root: Path | None) -> VisionDataset:
    images = {image["id"]: image for image in coco.get("images", [])}
    categories = sorted(coco.get("categories", []), key=lambda category: category.get("id", 0))
    id_to_idx = {category["id"]: idx for idx, category in enumerate(categories)}
    classes = [str(category.get("name", category.get("id", idx))) for idx, category in enumerate(categories)]
    items: dict[int, VisionRecord] = {}
    has_seg = False
    vqas_by_image = _vqa_map(coco)

    def record_for(image_id: int) -> VisionRecord:
        if image_id in items:
            return items[image_id]
        image_info = images[image_id]
        image_path = _resolve_image_path(path, str(image_info["file_name"]), root)
        image = common.image_info_from_path(image_path)
        if root is None:
            image.path = Path(str(image_info["file_name"]))
        else:
            try:
                image.path = image_path.relative_to(root)
            except ValueError:
                # image lives outside the dataset dir (e.g. a to-coco --no-images export)
                image.path = image_path
        record = VisionRecord(
            image=image,
            split=split,
            rel_image_path=image.path,
            attributes={"coco_image_id": image_id},
            vqas=list(vqas_by_image.get(image_id, [])),
        )
        items[image_id] = record
        return record

    for annotation in coco.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        image_id = annotation["image_id"]
        record = record_for(image_id)
        cls_idx = id_to_idx.get(annotation.get("category_id", 0), 0)
        x, y, width, height = annotation.get("bbox", [0, 0, 0, 0])
        record.boxes.append(
            BBox(
                cx=(x + width / 2) / record.image.width if record.image.width else 0.0,
                cy=(y + height / 2) / record.image.height if record.image.height else 0.0,
                w=width / record.image.width if record.image.width else 0.0,
                h=height / record.image.height if record.image.height else 0.0,
                cls=cls_idx,
                label=classes[cls_idx] if 0 <= cls_idx < len(classes) else None,
                id=annotation.get("id"),
                attributes={"coco_category_id": annotation.get("category_id")},
            )
        )
        segmentation = annotation.get("segmentation", [])
        if segmentation and isinstance(segmentation, list):
            has_seg = True
            seg_lists = segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
            for coords in seg_lists:
                if not coords:
                    continue
                pts = [(coords[idx] / record.image.width, coords[idx + 1] / record.image.height) for idx in range(0, len(coords), 2)]
                if len(pts) >= 3:
                    record.polys.append(
                        Polygon(
                            points=pts,
                            cls=cls_idx,
                            label=classes[cls_idx] if 0 <= cls_idx < len(classes) else None,
                            id=annotation.get("id"),
                            attributes={"coco_category_id": annotation.get("category_id")},
                        )
                    )

    requested = common.canonical_task_id(task, allow_auto=True)
    dataset_task = "inst-seg" if requested == "auto" and has_seg else "det" if requested == "auto" else common.canonical_task_id(requested, allow_auto=False)
    for record in items.values():
        record.task = dataset_task

    dataset = VisionDataset(records=list(items.values()), classes=classes, task=dataset_task, root=root or path.parent)
    dataset.round_trip["coco"] = {"categories": categories}
    return dataset


class CocoAdapter:
    @classmethod
    def matches(cls, src: Path) -> bool:
        if src.is_file():
            if src.suffix.lower() in {".yaml", ".yml"}:
                data = common.read_yaml(src) or {}
                if not isinstance(data, dict):
                    return False
                if str(data.get("format", "")).lower() == "coco":
                    return True
                for key in ("train", "val", "valid", "test"):
                    value = data.get(key)
                    entries = value if isinstance(value, list) else [value] if value is not None else []
                    if any(str(entry).lower().endswith(".json") for entry in entries):
                        return True
                return False
            return _load_if_coco_json(src) is not None
        if not src.is_dir():
            return False
        for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
            candidate = src / name
            if candidate.exists():
                data = common.read_yaml(candidate) or {}
                if isinstance(data, dict) and str(data.get("format", "")).lower() == "coco":
                    return True
        return bool(_find_coco_jsons(src))

    @classmethod
    def _ingest_from_data_yaml(cls, data_yaml: Path, *, task: str, splits: list[str] | None) -> VisionDataset:
        data = common.read_yaml(data_yaml) or {}
        if not isinstance(data, dict):
            raise ValueError(f"COCO data yaml must parse to an object: {data_yaml}")
        root = _resolve_root(data_yaml, data)
        names = [str(item) for item in data.get("names", [])]

        def resolve_annotation(entry: Any) -> Path:
            path = Path(str(entry))
            if path.is_absolute():
                return path
            candidates = [data_yaml.parent / path, root / path, data_yaml.parent / "annotations" / path, root / "annotations" / path]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[0]

        split_entries: dict[str, list[Path]] = {}
        for key in ("train", "val", "valid", "test"):
            value = data.get(key)
            if value is None:
                continue
            entries = value if isinstance(value, (list, tuple)) else [value]
            normalized = "val" if key == "valid" else key
            split_entries[normalized] = [resolve_annotation(entry) for entry in entries]

        use_splits = [common.canonical_split_id(item) for item in (splits or list(split_entries.keys()) or ["train"])]
        all_records: list[VisionRecord] = []
        classes: list[str] | None = names or None
        dataset_task: str | None = None
        has_seg_any = False
        categories_payload: list[dict[str, Any]] | None = None
        for split in use_splits:
            for annotation_path in split_entries.get(split, []):
                dataset = _ingest_from_obj(annotation_path, _read_json(annotation_path), task=task, split=split, root=root)
                has_seg_any = has_seg_any or any(record.polys for record in dataset.records)
                if classes is None:
                    classes = dataset.classes
                    categories_payload = dataset.round_trip.get("coco", {}).get("categories")
                elif classes != dataset.classes:
                    raise ValueError(f"Inconsistent class lists across COCO files; see: {annotation_path}")
                if dataset_task is None:
                    dataset_task = dataset.task
                elif dataset_task != dataset.task and has_seg_any:
                    dataset_task = "inst-seg"
                all_records.extend(dataset.records)

        if not all_records:
            raise FileNotFoundError(f"No COCO annotations found via {data_yaml}")

        dataset = VisionDataset(records=all_records, classes=classes or [], task=dataset_task or "det", root=root)
        if categories_payload is not None:
            dataset.round_trip["coco"] = {"categories": categories_payload}
        return dataset

    @classmethod
    def read(cls, src: Path, *, task: str = "auto", splits: list[str] | None = None, **options: object) -> VisionDataset:
        path = src.resolve()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            return cls._ingest_from_data_yaml(path, task=task, splits=splits)
        if path.is_dir():
            for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
                candidate = path / name
                if candidate.exists() and cls.matches(candidate):
                    return cls._ingest_from_data_yaml(candidate, task=task, splits=splits)
            coco_files = _find_coco_jsons(path)
            if not coco_files:
                raise FileNotFoundError(f"No COCO annotation JSONs found under: {path}")
            all_records: list[VisionRecord] = []
            classes: list[str] | None = None
            dataset_task: str | None = None
            has_seg_any = False
            categories_payload: list[dict[str, Any]] | None = None
            allowed_splits = {common.canonical_split_id(item) for item in splits} if splits else None
            for coco_path, payload in coco_files:
                split = _infer_split(coco_path, root=path, default="train")
                if allowed_splits and split not in allowed_splits:
                    continue
                dataset = _ingest_from_obj(coco_path, payload, task=task, split=split, root=path)
                has_seg_any = has_seg_any or any(record.polys for record in dataset.records)
                if classes is None:
                    classes = dataset.classes
                    categories_payload = dataset.round_trip.get("coco", {}).get("categories")
                elif classes != dataset.classes:
                    raise ValueError(f"Inconsistent class lists across COCO files; see: {coco_path}")
                if dataset_task is None:
                    dataset_task = dataset.task
                elif dataset_task != dataset.task and has_seg_any:
                    dataset_task = "inst-seg"
                all_records.extend(dataset.records)
            dataset = VisionDataset(records=all_records, classes=classes or [], task=dataset_task or "det", root=path)
            if categories_payload is not None:
                dataset.round_trip["coco"] = {"categories": categories_payload}
            return dataset
        return _ingest_from_obj(path, _read_json(path), task=task, split="train", root=path.parent)

    @classmethod
    def _write_one_split(
        cls,
        dataset: VisionDataset,
        dst: Path,
        *,
        split: str,
        task: str,
        overwrite: bool,
        coco_style: bool,
        hardlink: bool,
        no_images: bool = False,
    ) -> None:
        categories_payload = dataset.round_trip.get("coco", {}).get("categories")
        if isinstance(categories_payload, list) and len(categories_payload) == len(dataset.classes):
            cats = [{"id": category.get("id", idx), "name": dataset.classes[idx]} for idx, category in enumerate(categories_payload)]
            cat_ids = [int(category["id"]) for category in cats]
        else:
            cats = [{"id": idx, "name": name} for idx, name in enumerate(dataset.classes)]
            cat_ids = [item["id"] for item in cats]

        images = []
        anns = []
        vqa_entries: list[dict[str, object]] = []
        ann_id = 1

        if dst.suffix:
            raise ValueError(f"COCO output is a directory, not a file: {dst}")
        dst.mkdir(parents=True, exist_ok=True)
        if coco_style:
            img_root = dst / "images"
            ann_root = dst / "annotations"
            ann_root.mkdir(parents=True, exist_ok=True)
            dst_path = ann_root / f"annotations_{split}.json"
        else:
            split_root = dst / split
            split_root.mkdir(parents=True, exist_ok=True)
            img_root = split_root
            dst_path = split_root / "_annotations.coco.json"
        if not no_images:
            img_root.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not overwrite:
            raise FileExistsError(str(dst_path))

        split_records = [record for record in dataset.records if record.split == split]
        for img_id, record in enumerate(split_records, start=1):
            src = common.resolve_record_image_path(dataset, record)
            if no_images:
                if not src.exists():
                    raise FileNotFoundError(f"to-coco --no-images needs resolvable images; missing: {src}")
                file_name = src.as_posix()
            else:
                img_dst = img_root / src.name
                counter = 1
                while img_dst.exists():
                    img_dst = img_root / f"{src.stem}_{counter}{src.suffix}"
                    counter += 1
                if hardlink:
                    common.copy_or_link(src, img_dst, hardlink=True)
                else:
                    shutil.copy2(src, img_dst)
                file_name = img_dst.name
            images.append({"id": img_id, "file_name": file_name, "width": record.image.width, "height": record.image.height})

            for vqa in record.vqas:
                entry: dict[str, object] = {"image_id": img_id, "question": vqa.question, "answer": vqa.answer}
                if vqa.score is not None:
                    entry["score"] = vqa.score
                if vqa.model:
                    entry["model"] = vqa.model
                if vqa.meta:
                    entry["meta"] = vqa.meta
                vqa_entries.append(entry)

            if task == "det":
                for box in record.boxes:
                    width = box.w * record.image.width
                    height = box.h * record.image.height
                    x = box.cx * record.image.width - width / 2
                    y = box.cy * record.image.height - height / 2
                    anns.append({"id": ann_id, "image_id": img_id, "category_id": cat_ids[box.cls] if 0 <= box.cls < len(cat_ids) else box.cls, "bbox": [x, y, width, height], "area": width * height, "iscrowd": 0})
                    ann_id += 1
            else:
                for poly in record.polys:
                    seg = [sum(([x * record.image.width, y * record.image.height] for x, y in poly.points), [])]
                    xs = [x * record.image.width for x, _ in poly.points]
                    ys = [y * record.image.height for _, y in poly.points]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    anns.append({"id": ann_id, "image_id": img_id, "category_id": cat_ids[poly.cls] if 0 <= poly.cls < len(cat_ids) else poly.cls, "segmentation": seg, "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
                    ann_id += 1

        payload: dict[str, Any] = {"images": images, "annotations": anns, "categories": cats}
        if vqa_entries:
            payload["vqas"] = vqa_entries
        dst_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def write(
        cls,
        dataset: VisionDataset,
        dst: Path,
        *,
        splits: list[str] | None = None,
        task: str | None = None,
        overwrite: bool = False,
        coco_style: bool = False,
        hardlink: bool = False,
        no_images: bool = False,
        **options: object,
    ) -> None:
        requested = common.canonical_task_id(task or dataset.task or "auto")
        use_task = "inst-seg" if requested == "auto" and any(record.polys for record in dataset.records) else "det" if requested == "auto" else common.canonical_task_id(requested, allow_auto=False)
        split_values = splits or sorted({record.split for record in dataset.records}) or ["train"]
        normalized_splits = [common.canonical_split_id(item) for item in split_values]
        for split in normalized_splits:
            cls._write_one_split(
                dataset, dst, split=split, task=use_task, overwrite=overwrite,
                coco_style=coco_style, hardlink=hardlink, no_images=no_images,
            )
