"""Adapter for YOLO split trees and `data.yaml` datasets.

This module expects YOLO image and label layouts for detection, instance
segmentation, pose, and optional VQA sidecars. In memory it reads and writes
canonical records with normalized geometry, split-relative image and label
paths, and dataset classes already resolved.

Example root:
    dataset/
      data.yaml
      train/
        images/
          sample.jpg
        labels/
          sample.txt
        labels_vqa/
          sample.txt
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from cvsuite.common.core import BBox, Keypoints, Polygon, VisionDataset, VisionRecord
from cvsuite.common.io import common


@dataclass
class DataYaml:
    path: Path
    task: str | None
    names: list[str]
    root: Path
    splits: dict[str, list[Path]]
    raw: dict[str, Any]
    task_raw: str | None = None


def _resolve_yaml_root(path: Path, data: dict[str, Any]) -> Path:
    base = path.parent
    root_field = data.get("path")
    if not root_field:
        return base
    root_path = Path(str(root_field))
    if root_path.is_absolute():
        return root_path
    candidate = (base / root_path).resolve()
    return candidate if candidate.exists() else base


def _resolve_split_paths(yaml_path: Path, base_root: Path, data: dict[str, Any]) -> dict[str, list[Path]]:
    yaml_dir = yaml_path.parent

    def _pick_path(key: str, entry: str) -> Path:
        path = Path(entry)
        if path.is_absolute():
            return path
        alt_key = {"val": "valid", "valid": "val"}.get(key)
        candidates = [
            (yaml_dir / path).resolve(),
            (base_root / path).resolve(),
            (yaml_dir / key / "images").resolve(),
            (base_root / key / "images").resolve(),
        ]
        if alt_key:
            candidates.extend(
                [
                    (yaml_dir / alt_key / "images").resolve(),
                    (base_root / alt_key / "images").resolve(),
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    splits: dict[str, list[Path]] = {}
    for key in ("train", "val", "valid", "test"):
        value = data.get(key)
        if value is None:
            continue
        entries = value if isinstance(value, list) else [value]
        normalized = "val" if key == "valid" else key
        splits[normalized] = [_pick_path(key, str(entry)) for entry in entries]
    return splits


def _read_data_yaml(path: Path) -> DataYaml:
    data = common.read_yaml(path) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YOLO data yaml must parse to an object: {path}")
    task_raw = str(data.get("task")).lower() if data.get("task") is not None else None
    task = common.canonical_task_id(task_raw, allow_auto=False) if task_raw is not None else None
    names = [str(item) for item in data.get("names", [])]
    root = _resolve_yaml_root(path, data)
    splits = _resolve_split_paths(path, root, data)
    return DataYaml(path=path, task=task, names=names, root=root, splits=splits, raw=data, task_raw=task_raw)


def _resolve_data_yaml(src: Path) -> Path:
    if src.is_file() and src.suffix.lower() in {".yaml", ".yml"}:
        return src
    if src.is_dir():
        for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
            candidate = src / name
            if candidate.exists():
                return candidate
        hit = next(src.rglob("data.yaml"), None)
        if hit:
            return hit
    raise FileNotFoundError(str(src))


def _label_dir_name(task: str, default_task: str) -> str:
    return "labels" if task == default_task else f"labels_{common.yolo_task_name(task)}"


def _task_label_candidates(base_dir: Path, task: str) -> list[Path]:
    suffix = common.yolo_task_name(task)
    return [base_dir / f"labels_{suffix}", base_dir / "labels"]


def _ultralytics_label_candidates(img_dir: Path, task: str) -> list[Path]:
    """Ultralytics layout: `<root>/images/<split>` <-> `<root>/labels[_task]/<split>`."""
    if img_dir.parent.name != "images":
        return []
    ultra_root = img_dir.parent.parent
    split = img_dir.name
    return [ultra_root / f"labels_{common.yolo_task_name(task)}" / split, ultra_root / "labels" / split]


def _labels_dir_from_images(img_dir: Path, task: str, default_task: str | None = None) -> Path:
    if default_task is not None:
        return img_dir.parent / _label_dir_name(task, default_task)
    candidates = _task_label_candidates(img_dir.parent, task) + _ultralytics_label_candidates(img_dir, task)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _labels_matched(img_dir: Path, lbl_dir: Path) -> int:
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        return 0
    return sum(
        1
        for img in img_dir.glob("*")
        if img.suffix.lower() in common.IMG_EXTS and (lbl_dir / f"{img.stem}.txt").exists()
    )


def _vqa_dir_from_images(img_dir: Path) -> Path:
    return img_dir.parent / "labels_vqa"


def _locate_split_tree(root: Path, split: str, task: str, default_task: str | None = None) -> dict[str, Path]:
    img_dir = root / split / "images"
    lbl_dir = _labels_dir_from_images(img_dir, task, default_task=default_task)
    return {"images": img_dir, "labels": lbl_dir, "vqa": _vqa_dir_from_images(img_dir)}


def _ensure_split_tree(root: Path, split: str, task: str, *, include_vqa: bool, default_task: str | None = None) -> dict[str, Path]:
    tree = _locate_split_tree(root, split, task, default_task=default_task)
    tree["images"].mkdir(parents=True, exist_ok=True)
    tree["labels"].mkdir(parents=True, exist_ok=True)
    if include_vqa:
        tree["vqa"].mkdir(parents=True, exist_ok=True)
    return tree


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    return [] if not text else [line.strip() for line in text.splitlines() if line.strip()]


def _parse_vqa_entries(data: object) -> list[dict[str, Any]]:
    if not data:
        return []
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("vqas"), list):
            return [entry for entry in data["vqas"] if isinstance(entry, dict)]
        if "question" in data or "answer" in data:
            return [data]
    return []


def _read_vqas(path: Path) -> list:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        parsed = _parse_vqa_entries(data)
        if parsed:
            return [common.qa_from_payload(entry, source="yolo", question_keys=("question", "prompt")) for entry in parsed]
    except Exception:
        pass
    out = []
    for line in text.splitlines():
        row = line.strip()
        if not row:
            continue
        if "\t" in row:
            question, answer = row.split("\t", 1)
        elif "|" in row:
            question, answer = row.split("|", 1)
        else:
            question, answer = "", row
        out.append(common.qa_from_payload({"question": question.strip(), "answer": answer.strip()}, source="yolo", question_keys=("question",)))
    return out


def _write_vqas(path: Path, vqas: list) -> None:
    payload = [common.qa_to_payload(vqa, question_key="question", id_key="question_id") for vqa in vqas]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_det(lines: list[str], classes: list[str]) -> list[BBox]:
    out: list[BBox] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_idx = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        label = classes[cls_idx] if 0 <= cls_idx < len(classes) else None
        out.append(BBox(cx=cx, cy=cy, w=w, h=h, cls=cls_idx, label=label, kind="det"))
    return out


def _parse_seg(lines: list[str], classes: list[str]) -> tuple[list[Polygon], list[BBox]]:
    polys: list[Polygon] = []
    boxes: list[BBox] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        cls_idx = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        if len(pts) < 3:
            continue
        label = classes[cls_idx] if 0 <= cls_idx < len(classes) else None
        polys.append(Polygon(points=pts, cls=cls_idx, label=label))
        xs = [point[0] for point in pts]
        ys = [point[1] for point in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        boxes.append(BBox(cx=x1 + w / 2.0, cy=y1 + h / 2.0, w=w, h=h, cls=cls_idx, label=label, kind="seg"))
    return polys, boxes


def _parse_pose(lines: list[str], classes: list[str]) -> tuple[list[BBox], list[Keypoints]]:
    boxes: list[BBox] = []
    keypoints_sets: list[Keypoints] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 8:
            raise ValueError(f"Invalid pose triple count in YOLO label line: {line}")
        cls_idx = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        rest = list(map(float, parts[5:]))
        if len(rest) % 3 != 0:
            raise ValueError(f"Invalid pose triple count in YOLO label line: {line}")
        triples = [(rest[i], rest[i + 1], rest[i + 2]) for i in range(0, len(rest), 3)]
        label = classes[cls_idx] if 0 <= cls_idx < len(classes) else None
        boxes.append(BBox(cx=cx, cy=cy, w=w, h=h, cls=cls_idx, label=label, kind="pose"))
        keypoints_sets.append(Keypoints(points=triples, cls=cls_idx, label=label, kind="pose"))
    return boxes, keypoints_sets


def _infer_task_from_label_files(label_dirs: list[Path], *, pose_hint: bool = False) -> str | None:
    scores = {"det": 0, "inst-seg": 0, "pose": 0}
    examined = 0
    for label_dir in label_dirs:
        if not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            examined += 1
            lines = _read_lines(label_path)
            for line in lines:
                parts = line.split()
                count = len(parts)
                if count == 5:
                    scores["det"] += 1
                if count >= 7 and (count - 1) % 2 == 0:
                    scores["inst-seg"] += 2
                if count >= 8 and (count - 5) % 3 == 0:
                    scores["pose"] += 2
            if examined >= 25 or sum(scores.values()) >= 8:
                break
        if examined >= 25 or sum(scores.values()) >= 8:
            break
    if pose_hint and scores["pose"] > 0:
        return "pose"
    if scores["inst-seg"] > 0 and scores["inst-seg"] >= scores["pose"]:
        return "inst-seg"
    if scores["pose"] > 0:
        return "pose"
    if scores["det"] > 0:
        return "det"
    return None


def _unique_image_and_label(img_dir: Path, lbl_dir: Path, base_stem: str, suffix: str) -> tuple[Path, Path]:
    counter = 0
    while True:
        stem = f"{base_stem}_{counter}" if counter else base_stem
        img_candidate = img_dir / f"{stem}{suffix}"
        lbl_candidate = lbl_dir / f"{stem}.txt"
        if not img_candidate.exists() and not lbl_candidate.exists():
            return img_candidate, lbl_candidate
        counter += 1


class YoloAdapter:
    @classmethod
    def matches(cls, src: Path) -> bool:
        if src.is_file() and src.suffix.lower() in {".yaml", ".yml"}:
            data = common.read_yaml(src) or {}
            if not isinstance(data, dict):
                return False
            return str(data.get("format", "")).lower() not in {"coco", "labelme"}
        if src.is_dir():
            for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
                candidate = src / name
                if candidate.exists():
                    data = common.read_yaml(candidate) or {}
                    if isinstance(data, dict) and str(data.get("format", "")).lower() in {"coco", "labelme"}:
                        return False
                    return True
            for split in ("train", "val", "valid", "test"):
                split_dir = src / split
                if (split_dir / "images").exists() and any((split_dir / label_dir).exists() for label_dir in ("labels", "labels_det", "labels_seg", "labels_pose")):
                    return True
        return False

    @classmethod
    def _scan_split(
        cls,
        img_dir: Path,
        lbl_dir: Path,
        split: str,
        task: str,
        root: Path,
        *,
        classes: list[str],
        vqa_dir: Path | None = None,
    ) -> list[VisionRecord]:
        records: list[VisionRecord] = []
        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() not in common.IMG_EXTS:
                continue
            image = common.image_info_from_path(img)
            rel_img = common.root_relative_path(root, img)
            image.path = rel_img
            lbl = lbl_dir / f"{img.stem}.txt"
            rel_lbl = common.root_relative_path(root, lbl)
            boxes: list[BBox] = []
            polys: list[Polygon] = []
            kpts: list[Keypoints] = []
            if lbl.exists():
                lines = _read_lines(lbl)
                if task == "inst-seg":
                    polys, boxes = _parse_seg(lines, classes)
                elif task == "pose":
                    boxes, kpts = _parse_pose(lines, classes)
                else:
                    boxes = _parse_det(lines, classes)
            vqas = _read_vqas(vqa_dir / f"{img.stem}.txt") if vqa_dir is not None else []
            records.append(
                VisionRecord(
                    image=image,
                    split=split,
                    task=task,
                    boxes=boxes,
                    polys=polys,
                    kpts=kpts,
                    rel_image_path=rel_img,
                    rel_label_path=rel_lbl,
                    vqas=vqas,
                )
            )
        return records

    @classmethod
    def _ingest_split_tree(cls, root: Path, *, task: str, splits: list[str] | None, classes: list[str] | None = None) -> VisionDataset:
        available_splits = [name for name in ("train", "val", "test") if (root / name / "images").exists()]
        use_splits = splits or available_splits or ["train"]
        class_list = list(classes or [])
        if task == "auto":
            label_dirs: list[Path] = []
            found_task_specific: set[str] = set()
            for split in use_splits:
                base = (root / split / "images").parent
                for mapped in ("det", "inst-seg", "pose"):
                    typed = base / f"labels_{common.yolo_task_name(mapped)}"
                    if typed.exists():
                        found_task_specific.add(mapped)
                        label_dirs.append(typed)
                default_dir = base / "labels"
                if default_dir.exists():
                    label_dirs.append(default_dir)
            if len(found_task_specific) == 1:
                resolved_task = next(iter(found_task_specific))
            else:
                resolved_task = _infer_task_from_label_files(label_dirs) or "det"
        else:
            resolved_task = common.canonical_task_id(task, allow_auto=False)

        records: list[VisionRecord] = []
        labels_matched = 0
        checked_dirs: list[Path] = []
        for split in use_splits:
            tree = _locate_split_tree(root, common.canonical_split_id(split), resolved_task)
            records.extend(cls._scan_split(tree["images"], tree["labels"], common.canonical_split_id(split), resolved_task, root, classes=class_list, vqa_dir=tree["vqa"]))
            labels_matched += _labels_matched(tree["images"], tree["labels"])
            checked_dirs.append(tree["labels"])
        if records and labels_matched == 0:
            raise RuntimeError(
                f"YOLO source at {root} resolved {len(records)} images but 0 label files. "
                f"Checked: {', '.join(str(d) for d in checked_dirs)}. "
                "Image-only datasets are not read by the YOLO adapter."
            )
        return VisionDataset(records=records, classes=class_list, task=resolved_task, root=root)

    @classmethod
    def _ingest_from_data_yaml(cls, data_yaml_path: Path, *, task: str, splits: list[str] | None) -> VisionDataset:
        meta = _read_data_yaml(data_yaml_path)
        use_splits = splits or list(meta.splits.keys()) or ["train"]
        pose_hint = bool(meta.raw.get("kpt_shape") or meta.raw.get("keypoints"))

        def dirs_for(split: str) -> dict[str, Path]:
            img_dirs = meta.splits.get(split) or []
            img_dir = img_dirs[0] if img_dirs else (meta.root / split / "images")
            base = img_dir.parent
            ultra_root = img_dir.parent.parent if img_dir.parent.name == "images" else None

            def _dir(name: str) -> Path:
                if ultra_root is not None:
                    candidate = ultra_root / name / img_dir.name
                    if candidate.is_dir():
                        return candidate
                return base / name

            return {
                "images": img_dir,
                "labels_default": _dir("labels"),
                "labels_det": _dir("labels_det"),
                "labels_seg": _dir("labels_seg"),
                "labels_pose": _dir("labels_pose"),
                "labels_vqa": _dir("labels_vqa"),
            }

        if task == "auto":
            found_task_specific: set[str] = set()
            label_dirs: list[Path] = []
            for split in use_splits:
                dirs = dirs_for(split)
                if dirs["labels_seg"].exists():
                    found_task_specific.add("inst-seg")
                    label_dirs.append(dirs["labels_seg"])
                if dirs["labels_pose"].exists():
                    found_task_specific.add("pose")
                    label_dirs.append(dirs["labels_pose"])
                if dirs["labels_det"].exists():
                    found_task_specific.add("det")
                    label_dirs.append(dirs["labels_det"])
                if dirs["labels_default"].exists():
                    if meta.task is not None:
                        found_task_specific.add(meta.task)
                    label_dirs.append(dirs["labels_default"])
            if len(found_task_specific) == 1:
                resolved_task = next(iter(found_task_specific))
            else:
                resolved_task = _infer_task_from_label_files(label_dirs, pose_hint=pose_hint) or meta.task or "det"
        else:
            resolved_task = common.canonical_task_id(task, allow_auto=False)

        records: list[VisionRecord] = []
        labels_matched = 0
        checked_dirs: list[Path] = []
        for split in use_splits:
            dirs = dirs_for(common.canonical_split_id(split))
            lbl_dir = _labels_dir_from_images(dirs["images"], resolved_task, default_task=None)
            records.extend(
                cls._scan_split(
                    dirs["images"],
                    lbl_dir,
                    common.canonical_split_id(split),
                    resolved_task,
                    meta.root,
                    classes=meta.names,
                    vqa_dir=dirs["labels_vqa"],
                )
            )
            labels_matched += _labels_matched(dirs["images"], lbl_dir)
            checked_dirs.append(lbl_dir)
        if records and labels_matched == 0:
            raise RuntimeError(
                f"YOLO source described by {data_yaml_path} resolved {len(records)} images but 0 label files. "
                f"Checked: {', '.join(str(d) for d in checked_dirs)}. "
                "If this is intentionally image-only, it is not a YOLO dataset."
            )
        dataset = VisionDataset(records=records, classes=meta.names, task=resolved_task, root=meta.root)
        dataset.round_trip["yolo"] = {"data_yaml": str(data_yaml_path)}
        dataset.meta["nc"] = int(meta.raw.get("nc", len(meta.names) or 0))
        shape = meta.raw.get("kpt_shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            dataset.meta["kpt_shape"] = (int(shape[0]), int(shape[1]))
        return dataset

    @classmethod
    def read(cls, src: Path, *, task: str = "auto", splits: list[str] | None = None, **options: object) -> VisionDataset:
        path = src.resolve()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            return cls._ingest_from_data_yaml(path, task=task, splits=splits)
        if path.is_dir():
            try:
                data_yaml = _resolve_data_yaml(path)
            except FileNotFoundError:
                data_yaml = None
            if data_yaml is not None:
                return cls._ingest_from_data_yaml(data_yaml, task=task, splits=splits)
            return cls._ingest_split_tree(path, task=task, splits=splits)
        raise FileNotFoundError(f"YOLO dataset path not found: {src}")

    @classmethod
    def _dataset_has_vqa(cls, dataset: VisionDataset, splits: set[str] | None = None) -> bool:
        for record in dataset.records:
            if splits and record.split not in splits:
                continue
            if record.vqas:
                return True
        return False

    @classmethod
    def _detect_tasks(cls, dataset: VisionDataset) -> list[str]:
        tasks: list[str] = []

        def add(task_name: str) -> None:
            if task_name not in tasks:
                tasks.append(task_name)

        for record in dataset.records:
            if any(box.kind not in {"pose", "seg"} for box in record.boxes):
                add("det")
            if record.polys:
                add("inst-seg")
            if record.kpts or any(box.kind == "pose" for box in record.boxes):
                add("pose")
        if not tasks:
            tasks.append(str(dataset.task or "det"))
        return tasks

    @classmethod
    def _resolve_default_label_task(cls, tasks: list[str], dataset_task: str | None, requested_task: str | None) -> str:
        if requested_task and requested_task != "auto":
            canonical = common.canonical_task_id(requested_task, allow_auto=False)
            if canonical not in tasks:
                joined = ", ".join(tasks)
                raise ValueError(f"Requested YOLO task {requested_task!r} is not present in the dataset; available tasks: {joined}")
            return canonical
        if len(tasks) == 1:
            return tasks[0]
        if "det" in tasks:
            return "det"
        if dataset_task and dataset_task in tasks:
            return dataset_task
        return tasks[0]

    @classmethod
    def _write_item(
        cls,
        record: VisionRecord,
        tree: dict[str, Path],
        *,
        task: str,
        hardlink: bool,
        allow_existing_images: bool,
        write_vqa: bool,
        dataset: VisionDataset,
    ) -> None:
        src = common.resolve_record_image_path(dataset, record)
        base_img = tree["images"] / src.name
        base_lbl = tree["labels"] / f"{src.stem}.txt"
        if allow_existing_images and base_img.exists():
            img_dst = base_img
            lbl_dst = base_lbl
            if lbl_dst.exists():
                img_dst, lbl_dst = _unique_image_and_label(tree["images"], tree["labels"], src.stem, src.suffix)
        else:
            img_dst, lbl_dst = _unique_image_and_label(tree["images"], tree["labels"], src.stem, src.suffix)

        if not img_dst.exists():
            if src.suffix.lower() not in common.IMG_EXTS:
                raise ValueError(f"Unsupported image extension for YOLO export: {src}")
            if hardlink:
                common.copy_or_link(src, img_dst, hardlink=True)
            else:
                img_dst.write_bytes(src.read_bytes())

        lines: list[str] = []
        if task == "det":
            for box in record.boxes:
                if box.kind in {"pose", "seg"}:
                    continue
                lines.append(f"{box.cls} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}")
        elif task == "inst-seg":
            for poly in record.polys:
                flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly.points)
                lines.append(f"{poly.cls} {flat}")
        else:
            pose_boxes = [box for box in record.boxes if box.kind == "pose"]
            for box, keypoints in zip(pose_boxes, record.kpts):
                flat = " ".join(f"{x:.6f} {y:.6f} {v:.0f}" for x, y, v in keypoints.points)
                lines.append(f"{box.cls} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f} {flat}")

        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if write_vqa and record.vqas:
            _write_vqas(tree["vqa"] / f"{img_dst.stem}.txt", record.vqas)

    @classmethod
    def write(
        cls,
        dataset: VisionDataset,
        dst: Path,
        *,
        splits: list[str] | None = None,
        task: str | None = None,
        hardlink: bool = False,
        **options: object,
    ) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        target_splits = {common.canonical_split_id(item) for item in splits} if splits else None
        has_vqa = cls._dataset_has_vqa(dataset, target_splits)
        tasks = cls._detect_tasks(dataset)
        default_task = cls._resolve_default_label_task(tasks, dataset.task, task)
        first = True
        for current_task in tasks:
            cache: dict[str, dict[str, Path]] = {}
            for record in dataset.records:
                split = record.split or "train"
                if target_splits and split not in target_splits:
                    continue
                if split not in cache:
                    cache[split] = _ensure_split_tree(dst, split, current_task, include_vqa=has_vqa and first, default_task=default_task)
                cls._write_item(
                    record,
                    cache[split],
                    task=current_task,
                    hardlink=hardlink,
                    allow_existing_images=not first,
                    write_vqa=first,
                    dataset=dataset,
                )
            first = False
