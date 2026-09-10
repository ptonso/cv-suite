from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvsuite.common import cli as common_cli
from cvsuite.common.cache import core


def _write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _use_home(monkeypatch, home: Path, hf_hub: Path | None = None) -> None:
    """Point every path resolver at `home` by replacing the import-time env snapshot."""

    env = {"CVSUITE_HOME": str(home)}
    if hf_hub is not None:
        env["HF_HUB_CACHE"] = str(hf_hub)
    monkeypatch.setattr("cvsuite.common.fm.core.paths._ENV_AT_IMPORT", env)


def _seed_cache(home: Path) -> None:
    providers = home / "providers"
    _write_bytes(providers / "deep_orientation" / "venv" / "bin" / "python", 10)
    _write_bytes(providers / "deep_orientation" / "weights" / "model.bin", 20)
    _write_bytes(providers / "clip" / "venv" / "bin" / "python", 5)
    _write_bytes(providers / "clip" / "weights" / "weights.bin", 30)


def test_cache_list_json_reports_sizes(monkeypatch, tmp_path: Path, capsys) -> None:
    home = tmp_path / "cvsuite"
    _seed_cache(home)
    _use_home(monkeypatch, home)

    exit_code = common_cli.main(["cache-list", "--json", "--sort", "size"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    providers = payload["providers"]
    assert [item["provider"] for item in providers] == ["clip", "deep_orientation"]
    assert providers[0]["total_size"] == 35
    assert providers[0]["venv_size"] == 5
    assert providers[0]["weights_size"] == 30
    assert providers[1]["total_size"] == 30
    assert payload["cache_root"] == str(home)
    assert payload["hub_dir"] == str(home / "hub")


def test_cache_list_distinguishes_owned_and_shared_hub_repos(monkeypatch, tmp_path: Path, capsys) -> None:
    home = tmp_path / "cvsuite"
    hf_hub = tmp_path / "hf"
    shared_target = hf_hub / "models--facebook--sam3"
    _write_bytes(shared_target / "snapshots" / "abc" / "config.json", 40)
    _write_bytes(home / "hub" / "models--laion--CLIP" / "snapshots" / "def" / "config.json", 12)
    (home / "hub" / "models--facebook--sam3").symlink_to(shared_target, target_is_directory=True)
    _use_home(monkeypatch, home, hf_hub)

    assert common_cli.main(["cache-list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    repos = {item["repo"]: item for item in payload["hub"]["repos"]}
    assert repos["models--facebook--sam3"]["shared"] is True
    assert repos["models--facebook--sam3"]["size"] == 0
    assert repos["models--laion--CLIP"]["shared"] is False
    assert repos["models--laion--CLIP"]["size"] == 12
    # Only bytes cvsuite actually owns are counted as reclaimable.
    assert payload["hub"]["owned_size"] == 12
    assert payload["hub"]["shared_count"] == 1


def test_cache_purge_specific_weights_only(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "cvsuite"
    _seed_cache(home)
    _use_home(monkeypatch, home)

    exit_code = common_cli.main(["cache-purge", "deep_orientation", "--part", "weights", "--yes"])

    assert exit_code == 0
    providers = home / "providers"
    assert (providers / "deep_orientation" / "venv").exists()
    assert not (providers / "deep_orientation" / "weights").exists()
    assert (providers / "clip").exists()


def test_cache_purge_all_dry_run_keeps_cache(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "cvsuite"
    _seed_cache(home)
    _use_home(monkeypatch, home)

    exit_code = common_cli.main(["cache-purge", "--all", "--dry-run"])

    assert exit_code == 0
    assert (home / "providers" / "deep_orientation").exists()
    assert (home / "providers" / "clip").exists()


def test_cache_purge_all_removes_entire_cache(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "cvsuite"
    _seed_cache(home)
    _use_home(monkeypatch, home)

    exit_code = common_cli.main(["cache-purge", "--all", "--yes"])

    assert exit_code == 0
    assert not home.exists()


def test_cache_purge_accepts_minicpm_alias(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "cvsuite"
    providers = home / "providers"
    _write_bytes(providers / "minicpm_v" / "venv" / "bin" / "python", 10)
    _write_bytes(providers / "minicpm_v" / "weights" / "model.bin", 20)
    _use_home(monkeypatch, home)

    exit_code = common_cli.main(["cache-purge", "minicpm-v", "--part", "venv", "--yes"])

    assert exit_code == 0
    assert not (providers / "minicpm_v" / "venv").exists()
    assert (providers / "minicpm_v" / "weights").exists()


def test_cache_purge_hub_unlinks_share_without_touching_target(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "cvsuite"
    hf_hub = tmp_path / "hf"
    shared_target = hf_hub / "models--facebook--sam3"
    _write_bytes(shared_target / "snapshots" / "abc" / "model.safetensors", 40)
    (home / "hub").mkdir(parents=True)
    (home / "hub" / "models--facebook--sam3").symlink_to(shared_target, target_is_directory=True)
    _write_bytes(home / "hub" / "models--laion--CLIP" / "snapshots" / "def" / "config.json", 12)
    _use_home(monkeypatch, home, hf_hub)

    exit_code = common_cli.main(["cache-purge", "--all", "--part", "hub", "--yes"])

    assert exit_code == 0
    assert not (home / "hub" / "models--facebook--sam3").exists()
    assert not (home / "hub" / "models--laion--CLIP").exists()
    # The user's own Hugging Face cache is untouched.
    assert (shared_target / "snapshots" / "abc" / "model.safetensors").read_bytes() == b"x" * 40


def test_cache_purge_hub_rejects_provider_names(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "cvsuite"
    _seed_cache(home)
    _use_home(monkeypatch, home)

    with pytest.raises(SystemExit, match="pass --all"):
        common_cli.main(["cache-purge", "clip", "--part", "hub", "--yes"])


def test_apply_purge_refuses_paths_outside_the_cache(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "cvsuite"
    home.mkdir()
    outside = tmp_path / "elsewhere"
    _write_bytes(outside / "precious.bin", 8)
    _use_home(monkeypatch, home)

    action = core.PurgeAction(provider="rogue", part="all", path=outside, size=8)
    with pytest.raises(RuntimeError, match="outside the cvsuite cache"):
        core.apply_purge([action])
    assert (outside / "precious.bin").exists()
