from __future__ import annotations

import json
import os
from pathlib import Path
import types

from PIL import Image
import pytest
import yaml

from cvsuite.common.fm.providers.bases.gen import (
    GEN_OUTPUT_KEY_ATTR,
    GEN_PROMPT_ATTR,
    GEN_PROMPT_INDEX_ATTR,
    GEN_SAMPLE_INDEX_ATTR,
)
from cvsuite.common.core import ImageRecord, Keypoints, Polygon, Record
from cvsuite.common.core import VisionDataset
from cvsuite.gen import cli as gen_cli
from cvsuite.gen.core import parse_prompt_items, router as gen_router
from cvsuite.gen.output.commands import to_dst

TEST_CREATE_PROVIDER = "flux"
TEST_EDIT_PROVIDER = "qwen_image_edit"


def _write_image(path: Path, *, color: tuple[int, int, int] = (255, 0, 0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), color=color).save(path)
    return path


def _make_labelme_root(root: Path) -> Path:
    images = root / "images"
    labels = root / "labelme"
    image_path = _write_image(images / "scene.jpg")
    labels.mkdir(parents=True, exist_ok=True)
    (labels / "scene.json").write_text(
        json.dumps(
            {
                "imagePath": image_path.name,
                "imageWidth": 24,
                "imageHeight": 18,
                "shapes": [
                    {
                        "label": "panel",
                        "shape_type": "polygon",
                        "points": [[2, 2], [20, 2], [18, 14], [3, 13]],
                        "flags": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return labels


def _make_yolo_det_root(root: Path) -> Path:
    image_path = _write_image(root / "train" / "images" / "det.jpg")
    label_path = root / "train" / "labels" / "det.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.4 0.3\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "names": ["thing"],
                "task": "det",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root


def _make_yolo_pose_root(root: Path) -> Path:
    _write_image(root / "train" / "images" / "pose.jpg")
    label_path = root / "train" / "labels_pose" / "pose.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.4 0.3 0.1 0.1 2 0.9 0.1 2\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "names": ["person"],
                "task": "pose",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root


def _make_coco_root(root: Path) -> Path:
    _write_image(root / "scene.jpg")
    (root / "coco.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "scene.jpg", "width": 24, "height": 18}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [3, 4, 10, 6]}],
                "categories": [{"id": 0, "name": "sign"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root / "coco.json"


def _generated_dataset(path: Path) -> VisionDataset:
    return VisionDataset(
        records=[Record(image=ImageRecord(path=path, width=24, height=18), split="train")],
        root=path.parent,
        meta={"gen_mode": "result"},
    )


@pytest.mark.parametrize(
    ("factory", "expected_format", "expected_count"),
    [
        (lambda root: _write_image(root / "single.jpg"), "images", 1),
        (lambda root: root, "unstructured", 2),
        (_make_labelme_root, "labelme", 1),
        (_make_yolo_det_root, "yolo", 1),
        (_make_coco_root, "coco", 1),
    ],
)
def test_gen_router_detects_and_ingests_supported_sources(
    tmp_path: Path,
    factory,
    expected_format: str,
    expected_count: int,
) -> None:
    src_root = tmp_path / "src"
    if factory is not _make_labelme_root and factory is not _make_yolo_det_root and factory is not _make_coco_root:
        _write_image(src_root / "a.jpg")
        _write_image(src_root / "b.jpg")
    src = factory(src_root)
    assert gen_router.detect_format(src) == expected_format
    dataset = gen_router.ingest_edit_source(src)
    assert len(dataset.records) == expected_count


def test_gen_create_pipeline_sets_fm_request_and_writes_to_file(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    generated = _write_image(tmp_path / "runner" / "generated.png", color=(0, 255, 0))

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = dataset.fm_request
            captured["no_resume"] = no_resume
            captured["records_in"] = len(dataset.records)
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    dst = tmp_path / "out.png"
    exit_code = gen_cli.main(["create", "--provider", TEST_CREATE_PROVIDER, "--prompt", "a sunset over mountains", "to-dst", str(dst)])

    assert exit_code == 0
    assert dst.exists()
    request = captured["request"]
    assert request is not None
    assert request.task == "gen"
    assert request.provider == TEST_CREATE_PROVIDER
    assert request.prompt == json.dumps(["a sunset over mountains"])
    assert captured["records_in"] == 1


def test_gen_create_width_height_flags_are_forwarded_as_model_args(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    generated = _write_image(tmp_path / "runner" / "sized.png", color=(0, 255, 0))

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = dataset.fm_request
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    exit_code = gen_cli.main(
        [
            "create",
            "--provider",
            TEST_CREATE_PROVIDER,
            "--prompt",
            "a square image",
            "--model-arg",
            "width=256",
            "--model-arg",
            "height=256",
            "--width",
            "768",
            "--height",
            "512",
            "to-dst",
            str(tmp_path / "sized.png"),
        ]
    )

    assert exit_code == 0
    request = captured["request"]
    assert request is not None
    assert request.model_args["width"] == 768
    assert request.model_args["height"] == 512


def test_gen_edit_single_size_dimension_is_forwarded(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    source = _write_image(tmp_path / "source.jpg")
    generated = _write_image(tmp_path / "runner" / "height_only.png", color=(0, 255, 0))

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = dataset.fm_request
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    exit_code = gen_cli.main(
        [
            "edit",
            str(source),
            "--provider",
            TEST_EDIT_PROVIDER,
            "--prompt",
            "make it taller",
            "--height",
            "640",
            "to-dst",
            str(tmp_path / "height-only.png"),
        ]
    )

    assert exit_code == 0
    request = captured["request"]
    assert request is not None
    assert "width" not in request.model_args
    assert request.model_args["height"] == 640


def test_gen_create_expands_repeated_prompt_flags_in_order(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    generated = _write_image(tmp_path / "runner" / "generated_multi.png", color=(0, 255, 0))

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["dataset"] = dataset
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    exit_code = gen_cli.main(
        [
            "create",
            "--provider",
            TEST_CREATE_PROVIDER,
            "--prompt",
            "prompt one",
            "--prompt",
            "prompt one",
            "--prompt",
            "prompt two",
            "to-dst",
            str(tmp_path / "out.png"),
        ]
    )

    assert exit_code == 0
    dataset = captured["dataset"]
    assert dataset.fm_request is not None
    assert dataset.fm_request.prompt == json.dumps(["prompt one", "prompt one", "prompt two"])
    assert len(dataset.records) == 3
    assert [record.attributes[GEN_PROMPT_ATTR] for record in dataset.records] == ["prompt one", "prompt one", "prompt two"]
    assert [record.attributes[GEN_PROMPT_INDEX_ATTR] for record in dataset.records] == [0, 1, 2]
    assert [record.attributes[GEN_SAMPLE_INDEX_ATTR] for record in dataset.records] == [0, 0, 0]
    assert [record.attributes[GEN_OUTPUT_KEY_ATTR] for record in dataset.records] == [
        "0000_0001",
        "0001_0001",
        "0002_0001",
    ]
    assert dataset.meta["gen_prompt_count"] == 3
    assert dataset.meta["gen_num_images_per_prompt"] == 1
    assert dataset.meta["gen_fanout_mode"] == "prompt-list"


def test_gen_create_num_images_expands_each_prompt_with_sample_keys(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    generated = _write_image(tmp_path / "runner" / "generated_samples.png", color=(0, 255, 0))

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["dataset"] = dataset
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    exit_code = gen_cli.main(
        [
            "create",
            "--provider",
            TEST_CREATE_PROVIDER,
            "--prompt",
            "alpha prompt",
            "--prompt",
            "beta prompt",
            "--num-images",
            "3",
            "to-dst",
            str(tmp_path / "out-dir"),
        ]
    )

    assert exit_code == 0
    dataset = captured["dataset"]
    assert dataset.fm_request is not None
    assert dataset.fm_request.prompt == json.dumps(["alpha prompt", "beta prompt"])
    assert len(dataset.records) == 6
    assert [record.attributes[GEN_PROMPT_INDEX_ATTR] for record in dataset.records] == [0, 0, 0, 1, 1, 1]
    assert [record.attributes[GEN_SAMPLE_INDEX_ATTR] for record in dataset.records] == [0, 1, 2, 0, 1, 2]
    keys = [record.attributes[GEN_OUTPUT_KEY_ATTR] for record in dataset.records]
    assert keys == [
        "0000_0001",
        "0000_0002",
        "0000_0003",
        "0001_0001",
        "0001_0002",
        "0001_0003",
    ]
    assert len(set(keys)) == 6
    assert dataset.meta["gen_prompt_count"] == 2
    assert dataset.meta["gen_num_images_per_prompt"] == 3


def test_gen_create_prompt_value_can_match_output_command_name(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    generated = _write_image(tmp_path / "runner" / "generated_prompt_name.png", color=(0, 255, 0))

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["request"] = dataset.fm_request
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    dst = tmp_path / "prompt_name.png"
    exit_code = gen_cli.main(["create", "--provider", TEST_CREATE_PROVIDER, "--prompt", "to-dst", "to-dst", str(dst)])

    assert exit_code == 0
    assert dst.exists()
    request = captured["request"]
    assert request is not None
    assert request.prompt == json.dumps(["to-dst"])


def test_gen_prompt_file_expands_yaml_and_json_lists(tmp_path: Path) -> None:
    yaml_path = tmp_path / "prompts.yaml"
    yaml_path.write_text("- cat\n- dog\n", encoding="utf-8")
    json_path = tmp_path / "prompts.json"
    json_path.write_text(json.dumps(["bird", "fish"]), encoding="utf-8")

    prompts = parse_prompt_items([str(yaml_path), "night city", str(json_path)], caller_cwd=tmp_path)

    assert prompts == ["cat", "dog", "night city", "bird", "fish"]
    assert [prompt.prompt_id for prompt in prompts] == ["0000", "0001", "0002", "0003", "0004"]


def test_gen_prompt_file_mapping_keys_become_prompt_ids(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text("lubricant_indicator: full lubricant indicator\npanel-A: control panel\n", encoding="utf-8")

    prompts = parse_prompt_items([str(prompt_path)], caller_cwd=tmp_path)

    assert prompts == ["full lubricant indicator", "control panel"]
    assert [prompt.prompt_id for prompt in prompts] == ["lubricant_indicator", "panel-A"]


def test_gen_prompt_file_single_entry_mapping_list_items_become_prompt_ids(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text("- lubricant_indicator: full lubricant indicator\n- panel-A: control panel\n", encoding="utf-8")

    prompts = parse_prompt_items([str(prompt_path)], caller_cwd=tmp_path)

    assert prompts == ["full lubricant indicator", "control panel"]
    assert [prompt.prompt_id for prompt in prompts] == ["lubricant_indicator", "panel-A"]


def test_gen_prompt_file_mixed_list_preserves_order_and_assigns_ids(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text(
        "- cat\n- lubricant_indicator: full lubricant indicator\n- dog\n- panel-A: control panel\n",
        encoding="utf-8",
    )

    prompts = parse_prompt_items([str(prompt_path)], caller_cwd=tmp_path)

    assert prompts == ["cat", "full lubricant indicator", "dog", "control panel"]
    assert [prompt.prompt_id for prompt in prompts] == ["0000", "lubricant_indicator", "0002", "panel-A"]


def test_gen_prompt_file_rejects_multi_entry_mapping_list_items(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text("- {alpha: first prompt, beta: second prompt}\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="exactly one id-to-prompt entry"):
        parse_prompt_items([str(prompt_path)], caller_cwd=tmp_path)


def test_gen_prompt_file_rejects_non_string_keyed_values(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text("- lubricant_indicator: 3\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="string prompt values only"):
        parse_prompt_items([str(prompt_path)], caller_cwd=tmp_path)


def test_gen_prompt_file_rejects_duplicate_sanitized_prompt_ids(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text("- panel A: first prompt\n- panel_A: second prompt\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="unique after filename sanitization"):
        parse_prompt_items([str(prompt_path)], caller_cwd=tmp_path)


def test_gen_prompt_existing_non_yaml_json_path_is_treated_as_literal_string(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("- should not be loaded as prompt file\n", encoding="utf-8")

    prompts = parse_prompt_items([str(prompt_path)], caller_cwd=tmp_path)

    assert prompts == [str(prompt_path)]


def test_gen_edit_pipeline_preserves_boxes(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    source = _make_yolo_det_root(tmp_path / "det_src")
    generated = _write_image(tmp_path / "runner" / "det_out.png", color=(0, 255, 0))

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["dataset"] = dataset
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    exit_code = gen_cli.main(
        ["edit", str(source), "--provider", TEST_EDIT_PROVIDER, "--prompt", "swap the sign", "to-dst", str(tmp_path / "edited.png")]
    )

    assert exit_code == 0
    dataset = captured["dataset"]
    assert dataset.fm_request is not None
    assert dataset.fm_request.task == "gen"
    assert dataset.fm_request.prompt == json.dumps(["swap the sign"])
    assert dataset.meta["gen_mode"] == "edit"
    assert dataset.meta["gen_source_format"] == "yolo"
    assert len(dataset.records) == 1
    assert dataset.records[0].boxes
    assert dataset.records[0].boxes[0].cls == 0
    assert dataset.records[0].attributes[GEN_PROMPT_ATTR] == "swap the sign"
    assert dataset.records[0].attributes[GEN_OUTPUT_KEY_ATTR]


def test_gen_edit_pipeline_preserves_polygons(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    source = _make_labelme_root(tmp_path / "labelme_src")
    generated = _write_image(tmp_path / "runner" / "poly_out.png", color=(0, 255, 0))

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["dataset"] = dataset
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    exit_code = gen_cli.main(
        ["edit", str(source), "--provider", TEST_EDIT_PROVIDER, "--prompt", "paint it blue", "to-dst", str(tmp_path / "poly.png")]
    )

    assert exit_code == 0
    dataset = captured["dataset"]
    assert dataset.records[0].polys
    assert isinstance(dataset.records[0].polys[0], Polygon)


def test_gen_edit_pipeline_preserves_keypoints(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    source = _make_yolo_pose_root(tmp_path / "pose_src")
    generated = _write_image(tmp_path / "runner" / "pose_out.png", color=(0, 255, 0))

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["dataset"] = dataset
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    exit_code = gen_cli.main(
        [
            "edit",
            str(source),
            "--provider",
            TEST_EDIT_PROVIDER,
            "--prompt",
            "change the pose background",
            "to-dst",
            str(tmp_path / "pose.png"),
        ]
    )

    assert exit_code == 0
    dataset = captured["dataset"]
    assert dataset.records[0].kpts
    assert isinstance(dataset.records[0].kpts[0], Keypoints)


def test_gen_edit_expands_cartesian_product_for_multiple_sources_and_prompts(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    src_root = tmp_path / "images"
    _write_image(src_root / "a.jpg")
    _write_image(src_root / "b.jpg")
    generated = _write_image(tmp_path / "runner" / "fanout.png", color=(0, 255, 0))

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["dataset"] = dataset
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    exit_code = gen_cli.main(
        [
            "edit",
            str(src_root),
            "--provider",
            TEST_EDIT_PROVIDER,
            "--prompt",
            "make it blue",
            "--prompt",
            "make it green",
            "to-dst",
            str(tmp_path / "edited.png"),
        ]
    )

    assert exit_code == 0
    dataset = captured["dataset"]
    assert dataset.fm_request is not None
    assert dataset.fm_request.prompt == json.dumps(["make it blue", "make it green"])
    assert len(dataset.records) == 4
    assert dataset.meta["gen_prompt_count"] == 2
    assert dataset.meta["gen_num_images_per_prompt"] == 1
    assert dataset.meta["gen_source_count"] == 2
    assert dataset.meta["gen_fanout_mode"] == "cartesian"
    assert {record.attributes[GEN_PROMPT_ATTR] for record in dataset.records} == {"make it blue", "make it green"}


def test_gen_edit_num_images_expands_sources_prompts_and_samples(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    source = _make_yolo_det_root(tmp_path / "det_src")
    _write_image(tmp_path / "det_src" / "train" / "images" / "det2.jpg")
    (tmp_path / "det_src" / "train" / "labels" / "det2.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    generated = _write_image(tmp_path / "runner" / "edit_samples.png", color=(0, 255, 0))

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["dataset"] = dataset
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    exit_code = gen_cli.main(
        [
            "edit",
            str(source),
            "--provider",
            TEST_EDIT_PROVIDER,
            "--prompt",
            "make it blue",
            "--prompt",
            "make it green",
            "--num-images",
            "3",
            "to-dst",
            str(tmp_path / "edited"),
        ]
    )

    assert exit_code == 0
    dataset = captured["dataset"]
    assert dataset.fm_request is not None
    assert dataset.fm_request.prompt == json.dumps(["make it blue", "make it green"])
    assert len(dataset.records) == 12
    assert dataset.meta["gen_prompt_count"] == 2
    assert dataset.meta["gen_num_images_per_prompt"] == 3
    assert dataset.meta["gen_source_count"] == 2
    assert [record.attributes[GEN_SAMPLE_INDEX_ATTR] for record in dataset.records[:6]] == [0, 1, 2, 0, 1, 2]
    assert all(record.boxes and record.boxes[0].cls == 0 for record in dataset.records)
    keys = [record.attributes[GEN_OUTPUT_KEY_ATTR] for record in dataset.records]
    assert len(set(keys)) == 12
    assert all("__" not in key and "sample-" not in key for key in keys)


def test_gen_create_keyed_prompt_file_drives_output_keys_and_num_images(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    generated = _write_image(tmp_path / "runner" / "generated_keyed.png", color=(0, 255, 0))
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text("- weather:raining: make it rainy\n- oil_high: raise oil level\n", encoding="utf-8")

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["dataset"] = dataset
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    exit_code = gen_cli.main(
        [
            "create",
            "--provider",
            TEST_CREATE_PROVIDER,
            "--prompt",
            str(prompt_path),
            "--num-images",
            "3",
            "to-dst",
            str(tmp_path / "out-dir"),
        ]
    )

    assert exit_code == 0
    dataset = captured["dataset"]
    assert dataset.fm_request is not None
    assert dataset.fm_request.prompt == json.dumps(["make it rainy", "raise oil level"])
    assert len(dataset.records) == 6
    assert [record.attributes[GEN_PROMPT_INDEX_ATTR] for record in dataset.records] == [0, 0, 0, 1, 1, 1]
    assert [record.attributes[GEN_SAMPLE_INDEX_ATTR] for record in dataset.records] == [0, 1, 2, 0, 1, 2]
    assert [record.attributes[GEN_OUTPUT_KEY_ATTR] for record in dataset.records] == [
        "weather_raining_0001",
        "weather_raining_0002",
        "weather_raining_0003",
        "oil_high_0001",
        "oil_high_0002",
        "oil_high_0003",
    ]
    assert dataset.meta["gen_prompt_count"] == 2
    assert dataset.meta["gen_num_images_per_prompt"] == 3


def test_gen_edit_keyed_prompt_file_expands_sources_prompts_and_samples(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    src_root = tmp_path / "images"
    _write_image(src_root / "a.jpg")
    _write_image(src_root / "b.jpg")
    _write_image(src_root / "c.jpg")
    generated = _write_image(tmp_path / "runner" / "edit_keyed.png", color=(0, 255, 0))
    prompt_path = tmp_path / "prompts.yaml"
    prompt_path.write_text("- weather:raining: make it rainy\n- oil_high: raise oil level\n", encoding="utf-8")

    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            captured["dataset"] = dataset
            return _generated_dataset(generated)

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    exit_code = gen_cli.main(
        [
            "edit",
            str(src_root),
            "--provider",
            TEST_EDIT_PROVIDER,
            "--prompt",
            str(prompt_path),
            "--num-images",
            "2",
            "to-dst",
            str(tmp_path / "edited"),
        ]
    )

    assert exit_code == 0
    dataset = captured["dataset"]
    assert dataset.fm_request is not None
    assert dataset.fm_request.prompt == json.dumps(["make it rainy", "raise oil level"])
    assert len(dataset.records) == 12
    assert dataset.meta["gen_prompt_count"] == 2
    assert dataset.meta["gen_num_images_per_prompt"] == 2
    assert dataset.meta["gen_source_count"] == 3
    assert [record.attributes[GEN_OUTPUT_KEY_ATTR] for record in dataset.records] == [
        "weather_raining_0000_a_0001",
        "weather_raining_0000_a_0002",
        "oil_high_0000_a_0001",
        "oil_high_0000_a_0002",
        "weather_raining_0001_b_0001",
        "weather_raining_0001_b_0002",
        "oil_high_0001_b_0001",
        "oil_high_0001_b_0002",
        "weather_raining_0002_c_0001",
        "weather_raining_0002_c_0002",
        "oil_high_0002_c_0001",
        "oil_high_0002_c_0002",
    ]


@pytest.mark.parametrize("action", ["create", "edit"])
@pytest.mark.parametrize("num_images", ["0", "-2"])
def test_gen_num_images_rejects_non_positive_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    num_images: str,
) -> None:
    monkeypatch.setattr(
        "cvsuite.gen.transform.core.runtime.FMRunner.from_dataset",
        lambda dataset: pytest.fail("FMRunner should not be invoked for invalid --num-images"),
    )
    if action == "create":
        argv = [
            "create",
            "--provider",
            TEST_CREATE_PROVIDER,
            "--prompt",
            "a prompt",
            "--num-images",
            num_images,
            "to-dst",
            str(tmp_path / "out.png"),
        ]
    else:
        src = _write_image(tmp_path / f"source-{num_images}.jpg")
        argv = [
            "edit",
            str(src),
            "--provider",
            TEST_EDIT_PROVIDER,
            "--prompt",
            "a prompt",
            "--num-images",
            num_images,
            "to-dst",
            str(tmp_path / "out.png"),
        ]

    with pytest.raises(SystemExit, match="--num-images"):
        gen_cli.main(argv)


@pytest.mark.parametrize("action", ["create", "edit"])
@pytest.mark.parametrize("flag", ["--width", "--height"])
@pytest.mark.parametrize("value", ["0", "-2"])
def test_gen_size_flags_reject_non_positive_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    flag: str,
    value: str,
) -> None:
    monkeypatch.setattr(
        "cvsuite.gen.transform.core.runtime.FMRunner.from_dataset",
        lambda dataset: pytest.fail("FMRunner should not be invoked for invalid size flags"),
    )
    if action == "create":
        argv = [
            "create",
            "--provider",
            TEST_CREATE_PROVIDER,
            "--prompt",
            "a prompt",
            flag,
            value,
            "to-dst",
            str(tmp_path / "out.png"),
        ]
    else:
        src = _write_image(tmp_path / f"source-{flag[2:]}-{value}.jpg")
        argv = [
            "edit",
            str(src),
            "--provider",
            TEST_EDIT_PROVIDER,
            "--prompt",
            "a prompt",
            flag,
            value,
            "to-dst",
            str(tmp_path / "out.png"),
        ]

    with pytest.raises(SystemExit, match=flag):
        gen_cli.main(argv)


def test_gen_size_rejection_includes_model_and_requested_dimensions(monkeypatch, tmp_path: Path) -> None:
    class _Runner:
        def run(self, dataset: VisionDataset, *, no_resume: bool = False) -> VisionDataset:
            raise ValueError("invalid dimensions")

    monkeypatch.setattr("cvsuite.gen.transform.core.runtime.FMRunner.from_dataset", lambda dataset: _Runner())

    with pytest.raises(RuntimeError, match=r"Provider 'flux' rejected requested size width=123 height=456"):
        gen_cli.main(
            [
                "create",
                "--provider",
                TEST_CREATE_PROVIDER,
                "--prompt",
                "a prompt",
                "--width",
                "123",
                "--height",
                "456",
                "to-dst",
                str(tmp_path / "out.png"),
            ]
        )


def test_to_dst_directory_target_uses_timestamp_name(monkeypatch, tmp_path: Path) -> None:
    src = _write_image(tmp_path / "runner" / "generated.png", color=(0, 255, 0))
    dataset = _generated_dataset(src)

    class _FixedDatetime:
        @classmethod
        def now(cls):
            from datetime import datetime

            return datetime(2026, 4, 22, 13, 45, 59)

    monkeypatch.setattr("cvsuite.gen.output.commands.to_dst.datetime", _FixedDatetime)

    out_dir = tmp_path / "outputs"
    to_dst.run(dataset, types.SimpleNamespace(dst=out_dir))

    expected = out_dir / "20260422_134559.png"
    assert expected.exists()
    assert dataset.records[0].image.path == expected


def test_to_dst_rejects_invalid_generated_dataset(tmp_path: Path) -> None:
    missing = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "missing.png", width=24, height=18), split="train")],
        root=tmp_path,
    )
    with pytest.raises(SystemExit, match="Generated image not found"):
        to_dst.run(missing, types.SimpleNamespace(dst=tmp_path / "out.png"))


def test_to_dst_writes_multiple_generated_records_to_directory(tmp_path: Path) -> None:
    one = _write_image(tmp_path / "runner" / "one.png", color=(0, 255, 0))
    two = _write_image(tmp_path / "runner" / "two.png", color=(0, 0, 255))
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=one, width=24, height=18),
                split="train",
                attributes={GEN_OUTPUT_KEY_ATTR: "0000__alpha"},
            ),
            Record(
                image=ImageRecord(path=two, width=24, height=18),
                split="train",
                attributes={GEN_OUTPUT_KEY_ATTR: "0001__beta"},
            ),
        ],
        root=tmp_path / "runner",
        meta={"gen_mode": "create", "gen_prompt_count": 2, "gen_fanout_mode": "prompt-list"},
    )

    out_dir = tmp_path / "exports"
    to_dst.run(dataset, types.SimpleNamespace(dst=out_dir, with_stats=False))

    assert (out_dir / "0000__alpha.png").exists()
    assert (out_dir / "0001__beta.png").exists()
    assert dataset.root == out_dir
    assert dataset.records[0].image.path == Path("0000__alpha.png")
    assert dataset.records[1].image.path == Path("0001__beta.png")


def test_to_dst_writes_multiple_generated_records_to_file_stem_directory(tmp_path: Path) -> None:
    one = _write_image(tmp_path / "runner" / "one.png", color=(0, 255, 0))
    two = _write_image(tmp_path / "runner" / "two.png", color=(0, 0, 255))
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=one, width=24, height=18),
                split="train",
                attributes={GEN_OUTPUT_KEY_ATTR: "0000__alpha"},
            ),
            Record(
                image=ImageRecord(path=two, width=24, height=18),
                split="train",
                attributes={GEN_OUTPUT_KEY_ATTR: "0001__beta"},
            ),
        ],
        root=tmp_path / "runner",
        meta={"gen_mode": "edit", "gen_prompt_count": 1, "gen_num_images_per_prompt": 2, "gen_source_count": 1},
    )

    dst = tmp_path / "edited.png"
    to_dst.run(dataset, types.SimpleNamespace(dst=dst, with_stats=False))

    out_dir = tmp_path / "edited"
    assert out_dir.is_dir()
    assert (out_dir / "0000__alpha.png").exists()
    assert (out_dir / "0001__beta.png").exists()
    assert dataset.root == out_dir
    assert dataset.records[0].image.path == Path("0000__alpha.png")
    assert dataset.records[1].image.path == Path("0001__beta.png")


def test_to_dst_edit_exports_one_hardlinked_original_per_source(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "inputs" / "source.jpg", color=(255, 255, 0))
    one = _write_image(tmp_path / "runner" / "one.png", color=(0, 255, 0))
    two = _write_image(tmp_path / "runner" / "two.png", color=(0, 0, 255))
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=one, width=24, height=18),
                split="train",
                attributes={
                    GEN_OUTPUT_KEY_ATTR: "0000__alpha",
                    "gen_source_image": str(source),
                    "gen_source_key": "0000_source",
                },
            ),
            Record(
                image=ImageRecord(path=two, width=24, height=18),
                split="train",
                attributes={
                    GEN_OUTPUT_KEY_ATTR: "0001__beta",
                    "gen_source_image": str(source),
                    "gen_source_key": "0000_source",
                },
            ),
        ],
        root=tmp_path / "runner",
        meta={"gen_mode": "edit", "gen_prompt_count": 1, "gen_num_images_per_prompt": 2, "gen_source_count": 1},
    )

    out_dir = tmp_path / "exports"
    to_dst.run(dataset, types.SimpleNamespace(dst=out_dir, with_stats=False))

    original = out_dir / "0000_source.jpg"
    assert original.exists()
    assert os.path.samefile(original, source)
    assert (out_dir / "0000__alpha.png").exists()
    assert (out_dir / "0001__beta.png").exists()


def test_to_dst_edit_no_original_skips_source_export(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "inputs" / "source.jpg", color=(255, 255, 0))
    generated = _write_image(tmp_path / "runner" / "generated.png", color=(0, 255, 0))
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=generated, width=24, height=18),
                split="train",
                attributes={
                    GEN_OUTPUT_KEY_ATTR: "0000__alpha",
                    "gen_source_image": str(source),
                    "gen_source_key": "0000_source",
                },
            )
        ],
        root=tmp_path / "runner",
        meta={"gen_mode": "edit", "gen_prompt_count": 1, "gen_num_images_per_prompt": 1, "gen_source_count": 1},
    )

    dst = tmp_path / "edited.png"
    to_dst.run(dataset, types.SimpleNamespace(dst=dst, no_original=True, with_stats=False))

    assert dst.exists()
    assert not (tmp_path / "0000_source.jpg").exists()


def test_to_dst_overwrites_multiple_generated_records(tmp_path: Path) -> None:
    one = _write_image(tmp_path / "runner" / "one.png", color=(0, 255, 0))
    two = _write_image(tmp_path / "runner" / "two.png", color=(0, 0, 255))
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=one, width=24, height=18),
                split="train",
                attributes={GEN_OUTPUT_KEY_ATTR: "0000__alpha"},
            ),
            Record(
                image=ImageRecord(path=two, width=24, height=18),
                split="train",
                attributes={GEN_OUTPUT_KEY_ATTR: "0001__beta"},
            ),
        ],
        root=tmp_path / "runner",
    )
    out_dir = tmp_path / "exports"
    _write_image(out_dir / "0000__alpha.png", color=(255, 0, 0))
    _write_image(out_dir / "0001__beta.png", color=(255, 0, 0))

    to_dst.run(dataset, types.SimpleNamespace(dst=out_dir, overwrite=True, with_stats=False))

    with Image.open(out_dir / "0000__alpha.png") as img:
        assert img.getpixel((0, 0)) == (0, 255, 0)
    with Image.open(out_dir / "0001__beta.png") as img:
        assert img.getpixel((0, 0)) == (0, 0, 255)
    assert dataset.root == out_dir


def test_to_dst_overwrite_allows_same_source_and_destination(tmp_path: Path) -> None:
    image = _write_image(tmp_path / "existing.png", color=(0, 255, 0))
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=image, width=24, height=18), split="train")],
        root=tmp_path,
    )

    to_dst.run(dataset, types.SimpleNamespace(dst=image, overwrite=True, with_stats=False))

    assert image.exists()
    assert dataset.records[0].image.path == image
