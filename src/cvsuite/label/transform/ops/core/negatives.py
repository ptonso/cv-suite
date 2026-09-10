from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable
import random
import cv2
import numpy as np
from cvsuite.common.core import VisionDataset
from cvsuite.label.core.crops import resolve_image_path

@dataclass
class Crop:
    src_image: Path
    box_xyxy: Tuple[int, int, int, int]
    out_path: Path

def plan_crops(ds: VisionDataset, imgsz: int, per_image: int, min_gap: float, iou_thresh: float, seed: int) -> List[Crop]:
    rng = random.Random(seed)
    crops: List[Crop] = []
    sets = ds.by_split()
    root = ds.root or Path()
    for split, items in sets.items():
        for it in items:
            src_image = resolve_image_path(it.image.path, ds.root)
            im = cv2.imread(str(src_image))
            if im is None:
                continue
            h, w = im.shape[:2]
            for i in range(per_image):
                x1 = rng.randint(0, max(0, w - imgsz))
                y1 = rng.randint(0, max(0, h - imgsz))
                x2 = min(w, x1 + imgsz)
                y2 = min(h, y1 + imgsz)
                out = root / "negatives" / split / f"{it.image.path.stem}_{i}.jpg"
                crops.append(Crop(src_image=src_image, box_xyxy=(x1, y1, x2, y2), out_path=out))
    return crops

def write_crops(crops: List[Crop], out_root: Path, preview_sheet: Path | None) -> None:
    sheet: List[np.ndarray] = []
    for c in crops:
        im = cv2.imread(str(c.src_image))
        if im is None:
            continue
        x1, y1, x2, y2 = c.box_xyxy
        crop = im[y1:y2, x1:x2]
        dst = out_root / c.out_path.relative_to(c.out_path.parents[2])
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), crop)
        sheet.append(crop)
    if preview_sheet and sheet:
        grid = np.concatenate([cv2.resize(x, (256, 256)) for x in sheet[:24]], axis=1)
        preview_sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_sheet), grid)
