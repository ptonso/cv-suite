from __future__ import annotations

from pathlib import Path

import yaml

from cvsuite.common.core.enums import Task
from cvsuite.common.core import BBox, Classification, ImageRecord, Keypoints, Polygon, Record, VQA
from cvsuite.common.core import VisionDataset
from cvsuite.common.stats.builders import (
    build_classify_stats,
    build_gen_stats,
    build_label_stats,
    build_vlm_stats,
)
from cvsuite.common.stats.utils import emit_stats_yaml, make_export_context, resolve_stats_path, score_summary


def test_resolve_stats_path_supports_directory_and_file_targets(tmp_path: Path) -> None:
    assert resolve_stats_path(tmp_path / "out") == tmp_path / "out" / "stats.yaml"
    assert resolve_stats_path(tmp_path / "out.json") == tmp_path / "stats.yaml"


def test_emit_stats_yaml_overwrites_existing_file(tmp_path: Path) -> None:
    ctx = make_export_context(branch="classify", command="to-class-dir", dst=tmp_path / "out")
    emit_stats_yaml({"branch": "first"}, ctx)
    emit_stats_yaml({"branch": "second"}, ctx)
    payload = yaml.safe_load(ctx.stats_path.read_text(encoding="utf-8"))
    assert payload == {"branch": "second"}


def test_score_summary_handles_values_and_empty_inputs() -> None:
    assert score_summary([0.2, 0.4, 0.6]) == {"available": 3, "min": 0.2, "mean": 0.4, "max": 0.6}
    assert score_summary([]) == {"available": 0}


def test_build_classify_stats_counts_labels_and_unlabeled(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=tmp_path / "a.jpg", width=16, height=16),
                split="train",
                classification=Classification(label="cat", score=0.9, probs={"cat": 0.9, "dog": 0.1}),
            ),
            Record(
                image=ImageRecord(path=tmp_path / "b.jpg", width=16, height=16),
                split="val",
                classification=Classification(label="dog", score=0.2, probs={"cat": 0.1, "dog": 0.2}),
            ),
        ],
        root=tmp_path,
    )
    ctx = make_export_context(
        branch="classify",
        command="to-class-dir",
        dst=tmp_path / "out",
        run_meta={"threshold": 0.5, "unlabeled_name": "unlabeled", "multi_class": False},
    )

    stats = build_classify_stats(dataset, ctx)
    assert stats["dataset"]["splits"] == {"train": 1, "val": 1}
    assert stats["summary"]["assignments_by_output_label"] == {"cat": 1, "unlabeled": 1}
    assert stats["summary"]["unlabeled_records"] == 1
    assert stats["summary"]["confidence"]["available"] == 2


def test_build_label_stats_summarizes_annotations(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=tmp_path / "scene.jpg", width=20, height=10),
                split="train",
                boxes=[
                    BBox(cx=0.5, cy=0.5, w=0.2, h=0.4, cls=0, label="cat", score=0.8, prompt="cat"),
                    BBox(cx=0.5, cy=0.5, w=0.2, h=0.4, cls=2, label="text", score=0.9, kind="ocr", text="HELLO"),
                ],
                polys=[Polygon(points=[(0.1, 0.1), (0.3, 0.1), (0.2, 0.3)], cls=1, label="rust", prompt="rust")],
                kpts=[Keypoints(points=[(0.1, 0.1, 2.0), (0.2, 0.2, 2.0)], cls=3, label="person")],
            ),
            Record(image=ImageRecord(path=tmp_path / "empty.jpg", width=20, height=10), split="val"),
        ],
        classes=["cat", "rust", "text", "person"],
        task=Task.det,
        root=tmp_path,
    )
    ctx = make_export_context(branch="label", command="to-yolo", dst=tmp_path / "out")

    stats = build_label_stats(dataset, ctx)
    assert stats["dataset"]["task"] == "det"
    assert stats["summary"]["annotated_images"] == 1
    assert stats["summary"]["empty_images"] == 1
    assert stats["summary"]["boxes_total"] == 2
    assert stats["summary"]["polygons_total"] == 1
    assert stats["summary"]["poses_total"] == 1
    assert stats["summary"]["keypoints_total"] == 2
    assert stats["summary"]["ocr_text_boxes"] == 1
    assert stats["summary"]["ocr_nonempty_text_boxes"] == 1
    assert stats["summary"]["prompts_per_label"] == {"cat": 1, "rust": 1}


def test_build_vlm_stats_summarizes_pairs_and_buckets(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=tmp_path / "a.jpg", width=10, height=10),
                split="train",
                attributes={"vlm_answer_bucket": "YES"},
                vqas=[
                    VQA(question="q1", answer="yes", meta={"label": "rust_present"}),
                    VQA(question="q2", answer="", meta={"label": "corrosion_severity"}),
                ],
            ),
            Record(
                image=ImageRecord(path=tmp_path / "b.jpg", width=10, height=10),
                split="val",
                attributes={"vlm_answer_bucket": "NO"},
                vqas=[VQA(question="q3", answer="no", meta={"label": "rust_present"})],
            ),
        ],
        root=tmp_path,
    )
    ctx = make_export_context(
        branch="vlm",
        command="to-json",
        dst=tmp_path / "out",
        writer_details={"partitions_written": 2},
    )

    stats = build_vlm_stats(dataset, ctx)
    assert stats["dataset"]["unique_images"] == 2
    assert stats["summary"]["qa_pairs_total"] == 3
    assert stats["summary"]["answered_pairs"] == 2
    assert stats["summary"]["unanswered_pairs"] == 1
    assert stats["summary"]["pairs_by_question_label"] == {"corrosion_severity": 1, "rust_present": 2}
    assert stats["summary"]["mapped_answer_buckets"] == {"NO": 1, "YES": 1}
    assert stats["summary"]["partitions_written"] == 2


def test_build_gen_stats_includes_source_summary(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "gen.png", width=64, height=32), split="train")],
        root=tmp_path,
        meta={
            "gen_mode": "edit",
            "source_format": "yolo",
            "source_records": 1,
            "source_boxes_total": 3,
            "source_polygons_total": 1,
            "source_keypoints_total": 2,
        },
    )
    ctx = make_export_context(
        branch="gen",
        command="to-dst",
        dst=tmp_path / "out.png",
        primary_artifacts=[tmp_path / "out.png"],
    )

    stats = build_gen_stats(dataset, ctx)
    assert stats["summary"]["mode"] == "edit"
    assert stats["summary"]["image_size"] == {"width": 64, "height": 32}
    assert stats["summary"]["source"]["format"] == "yolo"
    assert stats["output"]["primary_artifacts"] == [str(tmp_path / "out.png")]
