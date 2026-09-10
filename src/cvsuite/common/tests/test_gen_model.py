from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from cvsuite.common.core import FMRequest, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.providers.base import ProcessArgs
from cvsuite.common.fm.providers.bases.create import BaseCreateGenerationModel, PromptCardCreateBackend
from cvsuite.common.fm.providers.bases.edit import BaseEditGenerationModel
from cvsuite.gen.core.batch import build_create_dataset


class DummyCreateModel(BaseCreateGenerationModel[dict[str, object]]):
    model_name = "dummy-gen-create"


class DummyEditModel(BaseEditGenerationModel[dict[str, object]]):
    model_name = "dummy-gen-edit"


class FlakyPromptCardBackend(PromptCardCreateBackend):
    failed = False

    def generate_batch(self, *, mode: str, jobs, **kwargs):
        if not self.__class__.failed and any("second" in str(job.prompt) for job in jobs):
            self.__class__.failed = True
            raise RuntimeError("synthetic gen failure")
        return super().generate_batch(mode=mode, jobs=jobs, **kwargs)


def test_gen_model_runs_create_with_prompt_card_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = build_create_dataset(["a bright bird", "an orange cat"], root=tmp_path)
    dataset.fm_request = FMRequest(
        provider="dummy-gen-create",
        task="gen",
        prompt='["a bright bird", "an orange cat"]',
        device="cpu",
        model_args={"backend": "prompt-card", "width": 96, "height": 64},
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)
    monkeypatch.setenv("FM_RUNS_ROOT", str(tmp_path / "runs"))

    result = DummyCreateModel().run_from_paths(ProcessArgs(input=input_path, output=output_path))

    assert result.root is not None
    assert result.meta["fm"]["provider"] == "dummy-gen-create"
    assert result.meta["fm"]["backend"] == "prompt-card"
    assert result.meta["fm"]["prompt_count"] == 2
    assert len(result.records) == 2
    for record in result.records:
        path = Path(result.root) / record.image.path
        assert path.exists()
        assert record.image.width == 96
        assert record.image.height == 64
        assert record.attributes["fm_tasks"] == ["gen"]


def test_gen_model_resumes_multi_prompt_batches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FlakyPromptCardBackend.failed = False
    dataset = build_create_dataset(["first prompt", "second prompt"], root=tmp_path)
    dataset.fm_request = FMRequest(
        provider="dummy-gen-create",
        task="gen",
        prompt='["first prompt", "second prompt"]',
        device="cpu",
        batch_size=1,
        model_args={"backend": "cvsuite.common.tests.test_gen_model:FlakyPromptCardBackend", "width": 80, "height": 48},
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)
    monkeypatch.setenv("FM_RUNS_ROOT", str(tmp_path / "runs"))

    with pytest.raises(RuntimeError, match="synthetic gen failure"):
        DummyCreateModel().run_from_paths(ProcessArgs(input=input_path, output=output_path))

    result = DummyCreateModel().run_from_paths(ProcessArgs(input=input_path, output=output_path))

    assert len(result.records) == 2
    for record in result.records:
        path = Path(result.root) / record.image.path
        assert path.exists()


def test_gen_model_reruns_completed_job_when_resumed_image_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = build_create_dataset(["first prompt"], root=tmp_path)
    dataset.fm_request = FMRequest(
        provider="dummy-gen-create",
        task="gen",
        prompt='["first prompt"]',
        device="cpu",
        batch_size=1,
        model_args={"backend": "prompt-card", "width": 80, "height": 48},
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)
    monkeypatch.setenv("FM_RUNS_ROOT", str(tmp_path / "runs"))

    first = DummyCreateModel().run_from_paths(ProcessArgs(input=input_path, output=output_path))
    generated_path = Path(first.root) / first.records[0].image.path
    generated_path.unlink()

    second = DummyCreateModel().run_from_paths(ProcessArgs(input=input_path, output=output_path))

    assert (Path(second.root) / second.records[0].image.path).exists()


def test_gen_model_edits_source_images_with_prompt_card_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_path = tmp_path / "source.jpg"
    Image.new("RGB", (40, 30), color=(255, 0, 0)).save(source_path)
    dataset = VisionDataset(
        records=[
            Record(
                image=ImageRecord(path=source_path.name, width=40, height=30),
                attributes={
                    "gen_prompt": "add a blue banner",
                    "gen_prompt_index": 0,
                    "gen_output_key": "0000__source__add_a_blue_banner",
                },
            )
        ],
        root=tmp_path,
        meta={"gen_mode": "edit", "gen_prompt_count": 1, "gen_source_count": 1, "gen_fanout_mode": "cartesian"},
        fm_request=FMRequest(
            provider="dummy-gen-edit",
            task="gen",
            prompt='["add a blue banner"]',
            device="cpu",
            model_args={"backend": "prompt-card"},
        ),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)
    monkeypatch.setenv("FM_RUNS_ROOT", str(tmp_path / "runs"))

    result = DummyEditModel().run_from_paths(ProcessArgs(input=input_path, output=output_path))

    record = result.records[0]
    output_image = Path(result.root) / record.image.path
    assert output_image.exists()
    assert record.image.width == 40
    assert record.image.height == 30
