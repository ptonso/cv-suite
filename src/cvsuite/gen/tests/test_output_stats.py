from __future__ import annotations

import types
from pathlib import Path

from PIL import Image
import yaml

from cvsuite.common.core import ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.gen.output.commands import to_dst


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), color=(0, 255, 0)).save(path)
    return path


def test_to_dst_writes_stats_yaml_for_file_targets(tmp_path: Path) -> None:
    src = _write_image(tmp_path / "runner" / "generated.png")
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=src, width=24, height=18), split="train")],
        root=src.parent,
        meta={
            "gen_mode": "edit",
            "gen_prompt_count": 2,
            "gen_fanout_mode": "cartesian",
            "source_format": "yolo",
            "source_records": 1,
            "source_boxes_total": 3,
            "source_polygons_total": 0,
            "source_keypoints_total": 0,
        },
    )

    dst = tmp_path / "exports" / "edited.png"
    to_dst.run(dataset, types.SimpleNamespace(dst=dst, with_stats=True))

    stats = yaml.safe_load((dst.parent / "stats.yaml").read_text(encoding="utf-8"))
    assert stats["command"] == "to-dst"
    assert stats["output"]["primary_artifacts"] == [str(dst)]
    assert stats["summary"]["prompt_count"] == 2
    assert stats["summary"]["fanout_mode"] == "cartesian"
    assert stats["summary"]["source"]["format"] == "yolo"
