from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
import pytest

from cvsuite.common.core import FMRequest, ImageRecord, Record
from cvsuite.common.core import VisionDataset
from cvsuite.common.fm.providers.api import openrouter_gen, openrouter_vlm
from cvsuite.common.fm.providers.base import ProcessArgs


def _write_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color=color).save(path)
    return path


def test_openrouter_gen_rejects_unsupported_model_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    dataset = VisionDataset(
        records=[],
        meta={"gen_mode": "create"},
        fm_request=FMRequest(
            provider="openrouter",
            task="gen",
            model_args={
                "model_id": "google/gemini-2.5-flash-image",
                "max_new_tokens": 32,
            },
        ),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)

    with pytest.raises(SystemExit) as excinfo:
        openrouter_gen.OpenRouterGenerationProvider().run_from_paths(
            ProcessArgs(input=input_path, output=output_path)
        )
    captured = capsys.readouterr()
    schema_name = (
        f"{openrouter_gen.OpenRouterGenerationOptions.__module__}."
        f"{openrouter_gen.OpenRouterGenerationOptions.__qualname__}"
    )

    assert str(excinfo.value) == (
        "[openrouter] Unsupported --model-arg key(s): max_new_tokens. "
        f"Supported keys: height, model_id, width. Options class: {schema_name}."
    )
    assert captured.err == ""
    assert not output_path.exists()


def test_openrouter_vlm_rejects_unsupported_model_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    image_path = _write_image(tmp_path / "sample.png")
    dataset = VisionDataset(
        records=[Record(image=ImageRecord(path=Path(image_path.name), width=12, height=8))],
        root=tmp_path,
        fm_request=FMRequest(
            provider="openrouter",
            task="vlm",
            prompt="What color is the object?",
            model_args={
                "model_id": "qwen/qwen2.5-vl-72b-instruct",
                "max_new_tokens": 32,
            },
        ),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    dataset.to_json(input_path)

    with pytest.raises(SystemExit) as excinfo:
        openrouter_vlm.OpenRouterVLMProvider().run_from_paths(
            ProcessArgs(input=input_path, output=output_path)
        )
    captured = capsys.readouterr()
    schema_name = (
        f"{openrouter_vlm.OpenRouterVLMOptions.__module__}."
        f"{openrouter_vlm.OpenRouterVLMOptions.__qualname__}"
    )

    assert str(excinfo.value) == (
        "[openrouter] Unsupported --model-arg key(s): max_new_tokens. "
        f"Supported keys: model_id. Options class: {schema_name}."
    )
    assert captured.err == ""
    assert not output_path.exists()


def test_require_openrouter_api_key_loads_from_callers_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text('OPENROUTER_API_KEY="dotenv-test-key"\n', encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_ENV", raising=False)
    monkeypatch.setenv("FM_CALLER_CWD", str(workspace))

    api_key = openrouter_gen.require_openrouter_api_key()

    assert api_key == "dotenv-test-key"
    assert os.environ["OPENROUTER_API_KEY"] == "dotenv-test-key"


def test_require_openrouter_api_key_accepts_legacy_alias_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text('OPENROUTER_API_KEY_ENV="alias-test-key"\n', encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_ENV", raising=False)
    monkeypatch.setenv("FM_CALLER_CWD", str(workspace))

    api_key = openrouter_gen.require_openrouter_api_key()

    assert api_key == "alias-test-key"
    assert os.environ["OPENROUTER_API_KEY"] == "alias-test-key"
