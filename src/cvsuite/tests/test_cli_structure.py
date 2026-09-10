from __future__ import annotations

from pathlib import Path

import pytest

from cvsuite.branch_cli import discover_commands
from cvsuite import cli
from cvsuite.classify import cli as classify_cli
from cvsuite.gen import cli as gen_cli
from cvsuite.label import cli as label_cli
from cvsuite.vlm import cli as vlm_cli


def test_cli_discovers_expected_branches() -> None:
    branches = cli._discover_branches()
    assert set(branches) >= {"class", "common", "gen", "label", "prep", "vlm"}
    assert "classify" not in set(branches)
    assert "ocr" not in set(branches)


def test_generic_branch_command_discovery() -> None:
    assert {spec.cli_name for spec in discover_commands("cvsuite.common")} >= {
        "cache-list",
        "cache-purge",
    }
    assert {spec.cli_name for spec in discover_commands("cvsuite.prep")} >= {
        "arrange",
        "orient",
        "process",
        "sample",
    }


def test_vlm_branch_keeps_expected_registries() -> None:
    assert "to-json" in vlm_cli.COMMANDS
    assert "to-shards" in vlm_cli.COMMANDS
    assert "to-vqa-style" in vlm_cli.COMMANDS
    assert "json" not in vlm_cli.COMMANDS
    assert "shards" not in vlm_cli.COMMANDS
    assert "vqa-style" not in vlm_cli.COMMANDS
    assert "caption" in vlm_cli.TRANSFORM_COMMANDS
    assert "ask" in vlm_cli.TRANSFORM_COMMANDS
    assert "map-answers" in vlm_cli.TRANSFORM_COMMANDS
    assert "vqa" in vlm_cli.TRANSFORM_COMMANDS


def test_gen_branch_help_uses_action_then_output_usage(capsys) -> None:
    assert gen_cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "usage: cvsuite gen <action> [action args] <output-command> [args]" in out
    assert "create" in out
    assert "edit" in out
    assert "to-dst" in out


def test_gen_create_help_lists_dynamic_create_models_without_weights(capsys) -> None:
    assert gen_cli.main(["create", "--help"]) == 0
    out = capsys.readouterr().out
    assert "Local providers:" in out
    assert "API providers:" in out
    assert "  flux: default-model-id=black-forest-labs/FLUX.1-dev params=12B" in out
    assert "  stable_diffusion: default-model-id=stabilityai/stable-diffusion-xl-base-1.0 params=3.5B" in out
    assert "  openrouter: default-model-id=google/gemini-2.5-flash-image params=api" in out
    assert "  - black-forest-labs/FLUX.1-dev" not in out
    assert "qwen_image_edit" not in out
    assert "--num-images" in out
    assert "--width" in out
    assert "--height" in out
    assert "--weights" not in out


def test_gen_edit_help_lists_dynamic_edit_models_without_weights(capsys) -> None:
    assert gen_cli.main(["edit", "--help"]) == 0
    out = capsys.readouterr().out
    assert "Local providers:" in out
    assert "API providers:" in out
    assert "  qwen_image_edit: default-model-id=Qwen/Qwen-Image-Edit params=20B" in out
    assert "  step1x_edit: default-model-id=stepfun-ai/Step1X-Edit-v1p1-diffusers params=unknown" in out
    assert "  openrouter: default-model-id=google/gemini-2.5-flash-image params=api" in out
    assert "  - Qwen/Qwen-Image-Edit" not in out
    assert "stable_diffusion" not in out
    assert "--num-images" in out
    assert "--width" in out
    assert "--height" in out
    assert "--weights" not in out


def test_gen_branch_keeps_expected_registries() -> None:
    assert "create" in gen_cli.ACTION_COMMANDS
    assert "edit" in gen_cli.ACTION_COMMANDS
    assert "to-dst" in gen_cli.OUTPUT_COMMANDS


def test_top_level_class_alias_dispatches_to_classify_branch(monkeypatch) -> None:
    captured = {}

    def _fake_main(argv=None):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr("cvsuite.classify.cli.main", _fake_main)
    assert cli.main(["class", "--help"]) == 0
    assert captured["argv"] == ["--help"]


def test_top_level_classify_branch_name_is_unknown() -> None:
    with pytest.raises(SystemExit, match="Unknown branch: classify"):
        cli.main(["classify"])


def test_class_branch_help_uses_pipeline_usage(capsys) -> None:
    assert classify_cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "usage: cvsuite class <src> [ingest opts] [<transform> <transform args>] <command> [args]" in out
    assert "--from {auto,images,unstructured,multi-class}" in out
    assert "formats: auto, images, unstructured, multi-class" in out
    assert "to-class-dir" in out
    assert "infer" in out
    assert "sample" in out


def test_class_branch_keeps_expected_registries() -> None:
    assert "to-class-dir" in classify_cli.COMMANDS
    assert "infer" in classify_cli.TRANSFORM_COMMANDS
    assert "sample" in classify_cli.TRANSFORM_COMMANDS
    assert "ops-sample" not in classify_cli.TRANSFORM_COMMANDS


def test_class_branch_pipeline_runs_infer_then_output(monkeypatch, tmp_path: Path) -> None:
    events = []

    def _fake_ingest(src, from_hint="auto"):
        events.append(("ingest", src, from_hint))
        return {"stage": "ingested"}

    def _fake_infer(dataset, args):
        events.append(("infer", dataset, args.prompt, args.provider))
        return {"stage": "inferred"}

    def _fake_output(dataset, args):
        events.append(("output", dataset, args.dst))
        return 0

    monkeypatch.setattr("cvsuite.classify.cli.io.ingest", _fake_ingest)
    monkeypatch.setattr("cvsuite.classify.transform.commands.infer.run", _fake_infer)
    monkeypatch.setattr("cvsuite.classify.output.commands.to_class_dir.run", _fake_output)

    src = tmp_path / "images"
    dst = tmp_path / "classes"
    assert classify_cli.main([str(src), "infer", "--provider", "clip", "--prompt", "cat,dog", "to-class-dir", str(dst)]) == 0
    assert events[0] == ("ingest", src, "auto")
    assert events[1][0] == "infer"
    assert events[1][2] == "cat,dog"
    assert events[2] == ("output", {"stage": "inferred"}, dst)


def test_class_branch_pipeline_runs_sample_then_output(monkeypatch, tmp_path: Path) -> None:
    events = []

    def _fake_ingest(src, from_hint="auto"):
        events.append(("ingest", src, from_hint))
        return {"stage": "ingested"}

    def _fake_sample(dataset, args):
        events.append(
            ("sample", dataset, args.mode, args.max_n_per_class, args.max_frac_per_class, args.hardlink, args.seed)
        )
        return {"stage": "sampled"}

    def _fake_output(dataset, args):
        events.append(("output", dataset, args.dst, args.hardlink))
        return 0

    monkeypatch.setattr("cvsuite.classify.cli.io.ingest", _fake_ingest)
    monkeypatch.setattr("cvsuite.classify.transform.commands.sample.run", _fake_sample)
    monkeypatch.setattr("cvsuite.classify.output.commands.to_class_dir.run", _fake_output)

    src = tmp_path / "images"
    dst = tmp_path / "classes"
    assert classify_cli.main(
        [str(src), "sample", "--mode", "balance", "--max-n-per-class", "9", "--hardlink", "--seed", "4", "to-class-dir", str(dst)]
    ) == 0
    assert events[0] == ("ingest", src, "auto")
    assert events[1] == ("sample", {"stage": "ingested"}, "balance", 9, None, True, 4)
    assert events[2] == ("output", {"stage": "sampled"}, dst, True)


def test_class_branch_pipeline_accepts_multi_class_ingest(monkeypatch, tmp_path: Path) -> None:
    events = []

    def _fake_ingest(src, from_hint="auto"):
        events.append(("ingest", src, from_hint))
        return {"stage": "ingested"}

    def _fake_output(dataset, args):
        events.append(("output", dataset, args.dst))
        return 0

    monkeypatch.setattr("cvsuite.classify.cli.io.ingest", _fake_ingest)
    monkeypatch.setattr("cvsuite.classify.output.commands.to_class_dir.run", _fake_output)

    src = tmp_path / "images"
    dst = tmp_path / "classes"
    assert classify_cli.main([str(src), "--from", "multi-class", "to-class-dir", str(dst)]) == 0
    assert events[0] == ("ingest", src, "multi-class")
    assert events[1] == ("output", {"stage": "ingested"}, dst)


def test_label_branch_keeps_expected_registries() -> None:
    assert "to-yolo" in label_cli.COMMANDS
    assert "to-crop" not in label_cli.COMMANDS
    assert "to-classified" not in label_cli.COMMANDS
    assert "filter" in label_cli.TRANSFORM_COMMANDS
    assert "ground" in label_cli.TRANSFORM_COMMANDS
    assert "ocr" in label_cli.TRANSFORM_COMMANDS
    assert "ops" in label_cli.TRANSFORM_COMMANDS
    assert "sample" in label_cli.TRANSFORM_COMMANDS
    assert "filter-logistic-opt" not in label_cli.TRANSFORM_COMMANDS
    assert "filter-threshold" not in label_cli.TRANSFORM_COMMANDS
    assert "ops-rebox" not in label_cli.TRANSFORM_COMMANDS
    assert "ops-make-negatives" not in label_cli.TRANSFORM_COMMANDS
    assert "ops-crop-dets" not in label_cli.TRANSFORM_COMMANDS
    assert "ops-crop-detections" not in label_cli.TRANSFORM_COMMANDS
    assert "fm-run" not in label_cli.TRANSFORM_COMMANDS
    assert not hasattr(label_cli, "STANDALONE_COMMANDS")


def test_classify_and_vlm_active_transforms_use_flat_command_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "classify" / "transform" / "commands" / "infer.py").exists()
    assert (root / "classify" / "transform" / "commands" / "sample.py").exists()
    assert not (root / "classify" / "transform" / "infer" / "commands" / "run.py").exists()
    assert not (root / "classify" / "transform" / "ops" / "commands" / "sample.py").exists()

    assert (root / "vlm" / "transform" / "commands" / "ask.py").exists()
    assert (root / "vlm" / "transform" / "commands" / "caption.py").exists()
    assert (root / "vlm" / "transform" / "commands" / "map_answers.py").exists()
    assert (root / "vlm" / "transform" / "commands" / "vqa.py").exists()
    assert not (root / "vlm" / "transform" / "ask" / "commands" / "run.py").exists()
    assert not (root / "vlm" / "transform" / "caption" / "commands" / "run.py").exists()
    assert not (root / "vlm" / "transform" / "map_answers" / "commands" / "run.py").exists()
    assert not (root / "vlm" / "transform" / "vqa" / "commands" / "run.py").exists()


def test_docs_tree_exists() -> None:
    root = Path(__file__).resolve().parents[3]
    specs = root / "specs"
    for name in [
        "architecture.md",
        "overview.md",
        "dataclasses.md",
        "vocabulary.md",
        "cli.md",
    ]:
        assert (specs / name).exists()
    for branch in ["label", "classify", "prep", "vlm", "gen", "common"]:
        assert (specs / branch).is_dir()
