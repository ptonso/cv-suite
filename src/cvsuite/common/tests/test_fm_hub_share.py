"""Sharing the user's Hugging Face cache instead of re-downloading into cvsuite's hub."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cvsuite.common.fm.core import paths, utils


def _snapshot_env(monkeypatch, home: Path, hf_hub: Path) -> None:
    monkeypatch.setattr(paths, "_ENV_AT_IMPORT", {"CVSUITE_HOME": str(home), "HF_HUB_CACHE": str(hf_hub)})


def _seed_repo(hub: Path, model_id: str, revision: str = "rev-1", files: tuple[str, ...] = ("config.json",)) -> Path:
    repo_dir = hub / f"models--{model_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in files:
        (snapshot / name).write_text("{}", encoding="utf-8")
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(revision, encoding="utf-8")
    (repo_dir / "blobs").mkdir(parents=True, exist_ok=True)
    return repo_dir


def _forbid_download(monkeypatch) -> None:
    def _boom(**kwargs):
        raise AssertionError("snapshot_download must not run when a usable share exists")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=_boom))


def _fake_download(monkeypatch, calls: list[str], revision: str = "rev-new") -> None:
    def _download(*, repo_id: str, cache_dir: str, revision: str | None = None, token=None, allow_patterns=None):
        calls.append(repo_id)
        repo_dir = _seed_repo(Path(cache_dir), repo_id, revision or "rev-new")
        return str(repo_dir / "snapshots" / (revision or "rev-new"))

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=_download))


def test_existing_hf_model_is_symlinked_not_redownloaded(monkeypatch, tmp_path: Path) -> None:
    home, hf_hub = tmp_path / "cvsuite", tmp_path / "hf"
    target = _seed_repo(hf_hub, "facebook/sam3")
    _snapshot_env(monkeypatch, home, hf_hub)
    _forbid_download(monkeypatch)

    source = utils.materialize_hf_model_source(
        "facebook/sam3",
        hub_dir=home / "hub",
        stage_dir=home / "providers" / "sam3" / "stage",
        caller_cwd=tmp_path,
    )

    link = home / "hub" / "models--facebook--sam3"
    assert link.is_symlink()
    assert link.resolve() == target.resolve()
    assert source.local_files_only is True
    assert source.revision == "rev-1"
    assert source.snapshot_path is not None
    assert (source.snapshot_path / "config.json").exists()


def test_partial_share_falls_back_to_a_private_copy(monkeypatch, tmp_path: Path) -> None:
    """The user has the repo but not the file we need, so we must not link to it."""

    home, hf_hub = tmp_path / "cvsuite", tmp_path / "hf"
    _seed_repo(hf_hub, "facebook/sam2.1-hiera-small", files=("config.json",))
    _snapshot_env(monkeypatch, home, hf_hub)
    calls: list[str] = []
    _fake_download(monkeypatch, calls)

    source = utils.materialize_hf_model_source(
        "facebook/sam2.1-hiera-small",
        hub_dir=home / "hub",
        stage_dir=home / "providers" / "gsam" / "stage",
        caller_cwd=tmp_path,
        allow_patterns=["sam2.1_hiera_small.pt"],
    )

    repo_cache = home / "hub" / "models--facebook--sam2.1-hiera-small"
    assert calls == ["facebook/sam2.1-hiera-small"]
    assert repo_cache.is_dir() and not repo_cache.is_symlink()
    assert source.repo_cache_dir == repo_cache


def test_broken_share_is_unlinked_and_refetched(monkeypatch, tmp_path: Path) -> None:
    home, hf_hub = tmp_path / "cvsuite", tmp_path / "hf"
    target = _seed_repo(hf_hub, "facebook/sam3")
    _snapshot_env(monkeypatch, home, hf_hub)
    (home / "hub").mkdir(parents=True)
    link = home / "hub" / "models--facebook--sam3"
    link.symlink_to(target, target_is_directory=True)

    shutil.rmtree(target)
    calls: list[str] = []
    _fake_download(monkeypatch, calls)

    source = utils.materialize_hf_model_source(
        "facebook/sam3",
        hub_dir=home / "hub",
        stage_dir=home / "providers" / "sam3" / "stage",
        caller_cwd=tmp_path,
    )

    assert calls == ["facebook/sam3"]
    assert not link.is_symlink()
    assert link.is_dir()
    assert source.snapshot_path is not None and source.snapshot_path.exists()


def test_promotion_unlinks_a_share_instead_of_deleting_through_it(monkeypatch, tmp_path: Path) -> None:
    """Guards the promotion step: rmtree through a symlink would wipe the user's models.

    drop_stale_share is stubbed out to simulate a share going stale between the probe and
    the promotion -- the only window in which repo_cache can still be a symlink here.
    """

    home, hf_hub = tmp_path / "cvsuite", tmp_path / "hf"
    target = _seed_repo(hf_hub, "facebook/sam3")
    _snapshot_env(monkeypatch, home, hf_hub)
    (home / "hub").mkdir(parents=True)
    link = home / "hub" / "models--facebook--sam3"
    link.symlink_to(target, target_is_directory=True)

    monkeypatch.setattr(utils, "drop_stale_share", lambda *a, **k: False)
    monkeypatch.setattr(utils, "resolve_cached_hf_snapshot", lambda *a, **k: (None, None))
    calls: list[str] = []
    _fake_download(monkeypatch, calls)

    utils.materialize_hf_model_source(
        "facebook/sam3",
        hub_dir=home / "hub",
        stage_dir=home / "providers" / "sam3" / "stage",
        caller_cwd=tmp_path,
    )

    assert calls == ["facebook/sam3"]
    assert not link.is_symlink()
    # The user's Hugging Face cache survived intact.
    assert (target / "snapshots" / "rev-1" / "config.json").exists()
    assert (target / "refs" / "main").read_text(encoding="utf-8") == "rev-1"


def test_no_share_when_hf_cache_is_empty(monkeypatch, tmp_path: Path) -> None:
    home, hf_hub = tmp_path / "cvsuite", tmp_path / "hf"
    hf_hub.mkdir()
    _snapshot_env(monkeypatch, home, hf_hub)
    calls: list[str] = []
    _fake_download(monkeypatch, calls)

    utils.materialize_hf_model_source(
        "laion/CLIP-ViT-H-14",
        hub_dir=home / "hub",
        stage_dir=home / "providers" / "clip" / "stage",
        caller_cwd=tmp_path,
    )

    repo_cache = home / "hub" / "models--laion--CLIP-ViT-H-14"
    assert calls == ["laion/CLIP-ViT-H-14"]
    assert repo_cache.is_dir() and not repo_cache.is_symlink()


def test_staging_dir_is_left_clean_after_a_download(monkeypatch, tmp_path: Path) -> None:
    """Containment: nothing a download pulls in transitively may survive in the cache."""

    home, hf_hub = tmp_path / "cvsuite", tmp_path / "hf"
    hf_hub.mkdir()
    _snapshot_env(monkeypatch, home, hf_hub)
    stage_dir = home / "providers" / "clip" / "stage"
    _fake_download(monkeypatch, [])

    utils.materialize_hf_model_source(
        "laion/CLIP-ViT-H-14",
        hub_dir=home / "hub",
        stage_dir=stage_dir,
        caller_cwd=tmp_path,
    )

    assert not any(path.name.startswith("hf-") for path in stage_dir.iterdir())
    assert not (stage_dir / "models--laion--CLIP-ViT-H-14").exists()


def test_hf_hub_cache_env_points_at_the_shared_hub_after_materializing(monkeypatch, tmp_path: Path) -> None:
    """Transitive fetches should land in the shared hub, not a per-provider stage dir."""

    home, hf_hub = tmp_path / "cvsuite", tmp_path / "hf"
    _seed_repo(hf_hub, "facebook/sam3")
    _snapshot_env(monkeypatch, home, hf_hub)
    _forbid_download(monkeypatch)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    utils.materialize_hf_model_source(
        "facebook/sam3",
        hub_dir=home / "hub",
        stage_dir=home / "providers" / "sam3" / "stage",
        caller_cwd=tmp_path,
    )

    import os

    assert os.environ["HF_HUB_CACHE"] == str(home / "hub")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(home / "hub")


@pytest.mark.parametrize("model_id", ["local-only-name", ""])
def test_non_repo_model_ids_skip_the_hub_entirely(monkeypatch, tmp_path: Path, model_id: str) -> None:
    home, hf_hub = tmp_path / "cvsuite", tmp_path / "hf"
    _snapshot_env(monkeypatch, home, hf_hub)
    _forbid_download(monkeypatch)

    source = utils.materialize_hf_model_source(
        model_id,
        hub_dir=home / "hub",
        stage_dir=home / "providers" / "x" / "stage",
        caller_cwd=tmp_path,
    )
    assert source.repo_cache_dir is None
