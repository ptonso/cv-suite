from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from cvsuite.common.core import FMRequest, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.providers.base import BaseFMModel, BatchResult, ProcessArgs, RuntimeContext


class _FakeModel(BaseFMModel[dict, int, None]):
    model_name = "fake"
    options_cls = dict

    def __init__(self) -> None:
        super().__init__()
        self.fail_once = True

    def build_jobs(self, dataset: VisionDataset, ctx: RuntimeContext[dict]):
        return list(range(len(dataset.records)))

    def job_id(self, job: int, dataset: VisionDataset, ctx: RuntimeContext[dict]) -> str:
        return f"record:{job}"

    def load_runtime(self, dataset: VisionDataset, ctx: RuntimeContext[dict]) -> None:
        return None

    def process_batch(self, dataset: VisionDataset, batch, runtime, ctx: RuntimeContext[dict]) -> BatchResult:
        idx = batch[0]
        if idx == 1 and self.fail_once:
            self.fail_once = False
            raise RuntimeError("synthetic failure")
        rec = dataset.records[idx]
        rec.attributes["count"] = int(rec.attributes.get("count", 0)) + 1
        rec.attributes["done"] = True
        return BatchResult(modified_record_indices=[idx])

    def build_fm_meta(self, dataset: VisionDataset, runtime, ctx: RuntimeContext[dict]) -> dict[str, object]:
        return {"task": "fake", "provider": self.model_name}


class _ImplicitMainModel(BaseFMModel[dict, int, None]):
    __module__ = "__main__"
    options_cls = dict


@dataclass(frozen=True)
class _BaseStrictOptions:
    base_flag: int | None = None


@dataclass(frozen=True)
class _StrictOptions(_BaseStrictOptions):
    allowed: str | None = None


@dataclass(frozen=True)
class _NoArgOptions:
    pass


class _StrictModel(_FakeModel):
    model_name = "strict"
    options_cls = _StrictOptions


class _NoArgModel(_FakeModel):
    model_name = "no_args"
    options_cls = _NoArgOptions


def test_base_fm_model_infers_module_name_when_launched_with_python_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys.modules["__main__"],
        "__spec__",
        SimpleNamespace(name="cvsuite.common.fm.providers.create.qwen_image"),
        raising=False,
    )

    assert _ImplicitMainModel().model_name == "qwen_image"


def test_base_fm_model_resumes_from_jsonl_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[
            Record(image=ImageRecord(path=tmp_path / f"img-{idx}.jpg", width=10, height=10))
            for idx in range(3)
        ],
        fm_request=FMRequest(provider="fake", task="test", device="cpu", batch_size=1),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)
    monkeypatch.setenv("FM_RUNS_ROOT", str(tmp_path / "runs"))

    model = _FakeModel()
    with pytest.raises(RuntimeError, match="synthetic failure"):
        model.run_from_paths(ProcessArgs(input=input_path, output=output_path))

    model.run_from_paths(ProcessArgs(input=input_path, output=output_path))
    restored = VisionDataset.from_json(output_path)

    assert [rec.attributes["count"] for rec in restored.records] == [1, 1, 1]
    assert [rec.attributes["done"] for rec in restored.records] == [True, True, True]
    assert restored.meta["fm"]["provider"] == "fake"


def test_compute_run_id_changes_when_input_image_stats_change(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"first")
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=image_path, width=10, height=10))],
        fm_request=FMRequest(provider="fake", task="test", batch_size=1),
        root=tmp_path,
    )
    model = _FakeModel()

    first = model.compute_run_id(dataset, prompt="", config_path=None, caller_cwd=tmp_path)
    image_path.write_bytes(b"second-version")
    second = model.compute_run_id(dataset, prompt="", config_path=None, caller_cwd=tmp_path)

    assert first != second


def test_base_fm_model_rejects_auto_for_unsupported_model(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "img.jpg", width=10, height=10))],
        fm_request=FMRequest(provider="fake", task="test", device="auto"),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)

    with pytest.raises(RuntimeError, match="`--device auto` is not implemented"):
        _FakeModel().run_from_paths(ProcessArgs(input=input_path, output=output_path))


def test_base_fm_model_rejects_auto_nf4_even_before_runtime_load(tmp_path: Path) -> None:
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=tmp_path / "img.jpg", width=10, height=10))],
        fm_request=FMRequest(provider="fake", task="test", device="auto", precision="nf4"),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)

    with pytest.raises(RuntimeError, match="`--device auto --precision nf4` is not supported yet"):
        _FakeModel().run_from_paths(ProcessArgs(input=input_path, output=output_path))


def test_base_fm_model_rejects_unknown_model_args_with_supported_keys(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(
            provider="strict",
            task="test",
            device="cpu",
            model_args={"allowed": "ok", "unknown_flag": 7},
        ),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)

    with pytest.raises(SystemExit) as excinfo:
        _StrictModel().run_from_paths(ProcessArgs(input=input_path, output=output_path))
    captured = capsys.readouterr()
    schema_name = f"{_StrictOptions.__module__}.{_StrictOptions.__qualname__}"

    assert str(excinfo.value) == (
        f"[strict] Unsupported --model-arg key(s): unknown_flag. "
        f"Supported keys: allowed, base_flag. Options class: {schema_name}."
    )
    assert captured.err == ""
    assert not output_path.exists()


def test_base_fm_model_rejects_unknown_model_args_when_none_are_supported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(
            provider="no_args",
            task="test",
            device="cpu",
            model_args={"mystery": True},
        ),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)

    with pytest.raises(SystemExit) as excinfo:
        _NoArgModel().run_from_paths(ProcessArgs(input=input_path, output=output_path))
    captured = capsys.readouterr()
    schema_name = f"{_NoArgOptions.__module__}.{_NoArgOptions.__qualname__}"

    assert str(excinfo.value) == (
        f"[no_args] Unsupported --model-arg key(s): mystery. "
        f"Supported keys: none. Options class: {schema_name}."
    )
    assert captured.err == ""
    assert not output_path.exists()


def test_base_fm_model_accepts_inherited_dataclass_model_args(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = VisionDataset(
        records=[],
        fm_request=FMRequest(
            provider="strict",
            task="test",
            device="cpu",
            model_args={"allowed": "ok", "base_flag": 3},
        ),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)

    result = _StrictModel().run_from_paths(ProcessArgs(input=input_path, output=output_path))
    captured = capsys.readouterr()
    restored = VisionDataset.from_json(output_path)

    assert result.fm_request is not None
    assert result.fm_request.model_args == {"allowed": "ok", "base_flag": 3}
    assert restored.fm_request is not None
    assert restored.fm_request.model_args == {"allowed": "ok", "base_flag": 3}
    assert "warnings" not in restored.meta["fm"]
    assert captured.err == ""
