"""
Interactive, end-to-end smoke that exercises all formats plus visualization.
Skipped by default; set VISION_INTERACTIVE=1 to run manually.
"""

import os
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import pytest

from cvsuite.common.core.enums import Task
from cvsuite.common.core import BBox, ImageRecord, Keypoints, Polygon, Record
from cvsuite.common.core import VisionDataset
from cvsuite.label.core import router
import cvsuite.label.output.commands.to_labelme as cmd_to_labelme
import cvsuite.label.output.commands.to_yolo as cmd_to_yolo
import cvsuite.label.output.commands.to_coco as cmd_to_coco
import cvsuite.label.output.commands.audit as cmd_to_audit
import cvsuite.label.output.commands.to_results as cmd_to_results
import cvsuite.label.output.commands.inspect as cmd_to_inspect
from cvsuite.label.tests import test_label_commands as base_tests


def _imwrite_rgb(p: Path, w: int = 96, h: int = 64, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).convert("RGB").save(p)


def _synthetic_dataset(root: Path) -> VisionDataset:
    root.mkdir(parents=True, exist_ok=True)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)

    classes = ["robot", "rust", "human"]
    recs: List[Record] = []

    # Detection + segmentation in one image
    img1 = images / "scene_det_seg.jpg"
    _imwrite_rgb(img1, seed=10)
    w1, h1 = 96, 64
    recs.append(
        Record(
            image=ImageRecord(path=img1, width=w1, height=h1),
            split="train",
            boxes=[
                BBox(cx=0.35, cy=0.4, w=0.3, h=0.35, cls=0, kind="det"),
                BBox(cx=0.7, cy=0.6, w=0.2, h=0.25, cls=1, kind="det"),
            ],
            polys=[
                Polygon(points=[(0.5, 0.2), (0.8, 0.25), (0.78, 0.5), (0.55, 0.45)], cls=1),
            ],
        )
    )

    # Pure segmentation example
    img2 = images / "scene_seg.jpg"
    _imwrite_rgb(img2, seed=11)
    w2, h2 = 96, 64
    recs.append(
        Record(
            image=ImageRecord(path=img2, width=w2, height=h2),
            split="train",
            polys=[
                Polygon(points=[(0.15, 0.2), (0.3, 0.18), (0.32, 0.4), (0.12, 0.42)], cls=1),
            ],
        )
    )

    # Pose example
    img3 = images / "scene_pose.jpg"
    _imwrite_rgb(img3, seed=12)
    w3, h3 = 96, 64
    pose_box = BBox(cx=0.5, cy=0.5, w=0.35, h=0.45, cls=2, kind="pose")
    pose_kpts = Keypoints(
        points=[(0.45, 0.4, 2.0), (0.55, 0.4, 2.0), (0.55, 0.6, 2.0), (0.45, 0.6, 2.0)],
        cls=2,
        kind="pose",
    )
    recs.append(
        Record(
            image=ImageRecord(path=img3, width=w3, height=h3),
            split="train",
            boxes=[pose_box],
            kpts=[pose_kpts],
        )
    )

    return VisionDataset(records=recs, classes=classes, task=None, root=root)


def _audit_and_results(ds, base: Path, name: str) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    audit_path = base / f"{name}_audit.yaml"
    cmd_to_audit.run(ds, Namespace(dst=audit_path, split="", thresholds=""))
    outputs["audit"] = audit_path

    res_dir = base / f"{name}_results"
    cmd_to_results.run(
        ds,
        Namespace(
            dst=res_dir,
            names_mode="smart",
            imgsz=None,
            max=3,
            split="",
            skip_annotated=False,
            skip_crops=False,
            skip_masks=False,
            skip_keypoints=False,
        ),
    )
    outputs["results"] = res_dir
    return outputs


def _run_inspect(ds, title: str) -> None:
    print(f"[inspect] Showing: {title}")
    cmd_to_inspect.run(
        ds,
        Namespace(
            split="",
            task="all",
            imgsz=None,
            max=0,
            names_mode="smart",
        ),
    )


def _run_baseline_tests(tmp_root: Path) -> None:
    # Recreate the fixture data for the existing test and invoke it directly.
    work = {
        "root": tmp_root,
        "labelme_src": tmp_root / "labelme_src",
        "yolo_from_lm": tmp_root / "yolo_from_lm",
        "yolo_det": tmp_root / "yolo_det",
        "yolo_seg": tmp_root / "yolo_seg",
        "yolo_pose": tmp_root / "yolo_pose",
        "labelme_out": tmp_root / "labelme_out",
        "coco_out": tmp_root / "coco_out",
        "coco_src": tmp_root / "coco_src",
        "rebox_out": tmp_root / "rebox_out",
        "neg_out": tmp_root / "neg_out",
        "inspect_snaps": tmp_root / "inspect_snaps",
        "audit_out": tmp_root / "audit_report.yaml",
    }
    base_tests.test_label_pipeline_commands_end_to_end(work)


@pytest.mark.skipif(not os.environ.get("VISION_INTERACTIVE"), reason="Interactive manual smoke; set VISION_INTERACTIVE=1 to run.")
def test_manual_full_flow(tmp_path: Path):
    base = Path(__file__).parent / "data-test"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    bugs: List[str] = []
    try:
        # 1) Run the existing end-to-end pipeline test to ensure baseline coverage.
        _run_baseline_tests(tmp_path / "baseline")

        # 2) Build synthetic dataset with det/seg/pose.
        ds = _synthetic_dataset(base / "synthetic")

        # 3) Save as LabelMe.
        lm_dst = base / "labelme"
        cmd_to_labelme.run(ds, Namespace(dst=lm_dst, split="train", embed_image=False, hardlink=False))
        ds_lm = router.ingest(src=lm_dst, from_hint="labelme")
        _audit_and_results(ds_lm, base, "labelme")

        # 4) Save as YOLO (det/seg/pose).
        yolo_det = base / "yolo_det"
        yolo_seg = base / "yolo_seg"
        yolo_pose = base / "yolo_pose"
        cmd_to_yolo.run(ds, Namespace(dst=yolo_det, task="det", splits="", hardlink=False))
        cmd_to_yolo.run(ds, Namespace(dst=yolo_seg, task="seg", splits="", hardlink=False))
        cmd_to_yolo.run(ds, Namespace(dst=yolo_pose, task="pose", splits="", hardlink=False))
        ds_yolo_det = router.ingest(src=yolo_det / "data.yaml", from_hint="yolo", task="auto")
        ds_yolo_seg = router.ingest(src=yolo_seg / "data.yaml", from_hint="yolo", task="auto")
        ds_yolo_pose = router.ingest(src=yolo_pose / "data.yaml", from_hint="yolo", task="auto")
        _audit_and_results(ds_yolo_det, base, "yolo_det")
        _audit_and_results(ds_yolo_seg, base, "yolo_seg")
        _audit_and_results(ds_yolo_pose, base, "yolo_pose")

        # 5) Save as COCO (det and seg copies).
        ds_det = VisionDataset(records=ds.records, classes=ds.classes, task=Task.det, root=ds.root)
        ds_seg = VisionDataset(records=ds.records, classes=ds.classes, task=Task.seg, root=ds.root)
        coco_det_dir = base / "coco_det"
        coco_seg_dir = base / "coco_seg"
        cmd_to_coco.run(ds_det, Namespace(dst=coco_det_dir, split="train"))
        cmd_to_coco.run(ds_seg, Namespace(dst=coco_seg_dir, split="train"))
        ds_coco_det = router.ingest(src=coco_det_dir, from_hint="coco", task="det")
        ds_coco_seg = router.ingest(src=coco_seg_dir, from_hint="coco", task="seg")
        _audit_and_results(ds_coco_det, base, "coco_det")
        _audit_and_results(ds_coco_seg, base, "coco_seg")

        # 6) Interactive inspect for each dataset (user closes window to continue).
        for title, dataset in [
            ("LabelMe", ds_lm),
            ("YOLO det", ds_yolo_det),
            ("COCO det", ds_coco_det),
        ]:
            _run_inspect(dataset, title=title)

    except Exception as exc:
        bugs.append(str(exc))
        raise
    finally:
        shutil.rmtree(base, ignore_errors=True)
        if bugs:
            print("[manual-test] Bugs detected:\n", "\n".join(bugs))
        else:
            print("[manual-test] Completed without detected bugs.")


if __name__ == "__main__":
    import pytest
    # Default to running the interactive path if invoked directly
    os.environ.setdefault("VISION_INTERACTIVE", "1")
    raise SystemExit(pytest.main([__file__, "-q"]))
