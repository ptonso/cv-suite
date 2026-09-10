from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from PIL import Image
import yaml

from cvsuite.common.core import BBox, ImageRecord, Polygon, Record
from cvsuite.common.core import VisionDataset
from cvsuite.label.output.commands import to_results


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(255, 255, 255)).save(path)
    return path


def test_to_results_writes_stats_yaml_with_writer_details(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "src" / "scene.jpg")
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=image_path, width=32, height=24),
                boxes=[BBox(cx=0.5, cy=0.5, w=0.4, h=0.5, cls=0, label="cat", score=0.9)],
                polys=[Polygon(points=[(0.1, 0.1), (0.4, 0.1), (0.2, 0.3)], cls=1, label="rust")],
            )
        ],
        classes=["cat", "rust"],
        root=tmp_path / "src",
    )

    dst = tmp_path / "results"
    to_results.run(
        dataset,
        Namespace(
            dst=dst,
            names_mode="smart",
            imgsz=None,
            max=0,
            val_frac=0.0,
            test_frac=0.0,
            seed=0,
            preserve_splits=True,
            skip_annotated=False,
            skip_crops=False,
            skip_masks=False,
            skip_keypoints=True,
            with_stats=True,
        ),
    )

    stats = yaml.safe_load((dst / "stats.yaml").read_text(encoding="utf-8"))
    assert stats["command"] == "to-results"
    assert stats["summary"]["selected_records"] == 1
    assert stats["summary"]["annotated_images_written"] == 1
    assert stats["summary"]["crops_written_per_label"] == {"cat": 1}
    assert stats["summary"]["masks_written_per_label"] == {"rust": 1}
