from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import yaml

from cvsuite.vlm import cli as vlm_cli
from cvsuite.vlm.transform.core import map_answers as answer_mapping


def _make_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (12, 8), color=color)
    image.save(path)
    image.close()


def _write_flat_sample(
    root: Path,
    stem: str,
    *,
    answers: list[tuple[int, str, str]],
    color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    _make_image(root / f"{stem}.jpg", color=color)
    payload = {
        "version": "1",
        "task": "vlm_sample",
        "sample_key": stem,
        "split": "train",
        "qa_pairs": [
            {
                "question_id": question_id,
                "question": question,
                "answer": answer,
            }
            for question_id, question, answer in answers
        ],
    }
    (root / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_load_label_registry_accepts_list_and_map_shapes(tmp_path: Path) -> None:
    list_path = tmp_path / "labels_list.yaml"
    list_path.write_text(yaml.safe_dump(["YES", "NO"]), encoding="utf-8")
    registry = answer_mapping.load_label_registry([], labels_file=list_path)
    assert registry.labels == ("YES", "NO", "unclear")

    map_path = tmp_path / "labels_map.yaml"
    map_path.write_text(yaml.safe_dump({"YES": ["yes", "affirmative"], "NO": ["no", "negative"]}, sort_keys=False), encoding="utf-8")
    registry = answer_mapping.load_label_registry(["MAYBE"], labels_file=map_path)
    assert registry.labels == ("MAYBE", "YES", "NO", "unclear")
    assert registry.alias_to_label[answer_mapping.normalize_alias("affirmative")] == "YES"
    assert registry.alias_to_label[answer_mapping.normalize_alias("negative")] == "NO"


def test_load_label_registry_rejects_normalized_alias_collisions(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.yaml"
    labels_path.write_text(yaml.safe_dump({"YES": ["affirmative"], "NO": ["AFFIRMATIVE"]}, sort_keys=False), encoding="utf-8")

    with pytest.raises(SystemExit, match="collision"):
        answer_mapping.load_label_registry([], labels_file=labels_path)


def test_match_answer_text_handles_exact_trimmed_and_symbolic_cases() -> None:
    yes_no = answer_mapping.load_label_registry(["YES", "NO"])
    assert answer_mapping.match_answer_text(" yes ", yes_no).label == "YES"
    assert answer_mapping.match_answer_text("YES.", yes_no).label == "YES"
    assert answer_mapping.match_answer_text("final answer: no because the sign is absent", yes_no).label == "NO"

    abc = answer_mapping.load_label_registry(["A", "B", "C"])
    assert answer_mapping.match_answer_text("option C", abc).label == "C"
    assert answer_mapping.match_answer_text("I choose B", abc).label == "B"


def test_match_answer_text_routes_hedged_and_ambiguous_outputs_to_unclear() -> None:
    registry = answer_mapping.load_label_registry(["YES", "NO"])
    assert answer_mapping.match_answer_text("probably yes", registry).label == "unclear"
    assert answer_mapping.match_answer_text("yes/no", registry).label == "unclear"

    abc = answer_mapping.load_label_registry(["A", "B"])
    assert answer_mapping.match_answer_text("A or B", abc).label == "unclear"


def test_vlm_cli_map_answers_to_json_partitions_and_hardlinks(tmp_path: Path) -> None:
    src = tmp_path / "flat"
    _write_flat_sample(
        src,
        "sample",
        answers=[
            (10, "Is there rust?", " yes "),
            (11, "Is there damage?", "final answer: no because it looks intact"),
        ],
    )

    dst = tmp_path / "mapped_json"
    exit_code = vlm_cli.main([str(src), "map-answers", "YES", "NO", "to-json", str(dst), "--hardlink"])

    assert exit_code == 0
    assert sorted(path.name for path in dst.iterdir()) == ["NO", "YES", "unclear"]
    assert list((dst / "unclear").iterdir()) == []

    source_image = src / "sample.jpg"
    for expected_label, question_id in (("YES", 10), ("NO", 11)):
        bucket = dst / expected_label
        sidecars = sorted(bucket.glob("*.json"))
        images = sorted(path for path in bucket.iterdir() if path.suffix.lower() == ".jpg")
        assert len(sidecars) == 1
        assert len(images) == 1
        assert images[0].stat().st_ino == source_image.stat().st_ino

        payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
        qa = payload["qa_pairs"][0]
        assert qa["meta"]["mapped_label"] == expected_label
        assert qa["meta"]["source_question_id"] == question_id
        assert qa["meta"]["source_question_index"] in {0, 1}

    assert not any(path.name.startswith("map_answers_") for path in dst.iterdir())


def test_vlm_cli_map_answers_to_json_output_is_re_ingestable(tmp_path: Path) -> None:
    src = tmp_path / "flat"
    # same image stem lands in two different buckets -- the collision that used
    # to make the partitioned output un-re-ingestable
    _write_flat_sample(src, "shared", answers=[(1, "q1", "yes"), (2, "q2", "no")])

    mapped = tmp_path / "mapped"
    assert vlm_cli.main([str(src), "map-answers", "YES", "NO", "to-json", str(mapped)]) == 0

    rt = tmp_path / "roundtrip"
    assert vlm_cli.main([str(mapped), "--from", "json", "to-json", str(rt)]) == 0
    assert len(sorted(rt.rglob("*.json"))) == 2


def test_vlm_cli_map_answers_to_vqa_style_writes_bucket_manifests_and_empty_unclear(tmp_path: Path) -> None:
    src = tmp_path / "flat"
    _write_flat_sample(src, "sample", answers=[(7, "Is it safe?", "probably yes")], color=(0, 255, 0))

    labels_path = tmp_path / "labels.yaml"
    labels_path.write_text(yaml.safe_dump({"YES": ["yes", "affirmative"], "NO": ["no", "negative"]}, sort_keys=False), encoding="utf-8")

    dst = tmp_path / "mapped_vqa"
    exit_code = vlm_cli.main([str(src), "map-answers", "--labels-file", str(labels_path), "to-vqa-style", str(dst), "--skip-images"])

    assert exit_code == 0
    yes_manifest = json.loads((dst / "YES" / "vqa.json").read_text(encoding="utf-8"))
    no_manifest = json.loads((dst / "NO" / "vqa.json").read_text(encoding="utf-8"))
    unclear_manifest = json.loads((dst / "unclear" / "vqa.json").read_text(encoding="utf-8"))

    assert yes_manifest["items"] == []
    assert no_manifest["items"] == []
    assert len(unclear_manifest["items"]) == 1
    assert unclear_manifest["items"][0]["qas"][0]["meta"]["mapped_label"] == "unclear"
    assert not (dst / "YES" / "images").exists()
    assert not (dst / "unclear" / "images").exists()


def test_vlm_cli_map_answers_rejects_inspect_on_partitioned_dataset(tmp_path: Path) -> None:
    src = tmp_path / "flat"
    _write_flat_sample(src, "sample", answers=[(1, "Is there rust?", "yes")])

    with pytest.raises(SystemExit, match="does not support partitioned datasets"):
        vlm_cli.main([str(src), "map-answers", "YES", "NO", "inspect"])
