from __future__ import annotations

from pathlib import Path

import pytest

from cvsuite.common.fm.core import paths


def _snapshot(monkeypatch, **env: str) -> None:
    monkeypatch.setattr(paths, "_ENV_AT_IMPORT", dict(env))


def _seed_repo(hub: Path, repo: str, revision: str = "abc123", files: tuple[str, ...] = ("config.json",)) -> Path:
    repo_dir = hub / f"models--{repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in files:
        (snapshot / name).write_text("{}", encoding="utf-8")
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(revision, encoding="utf-8")
    return repo_dir


def test_cvsuite_home_precedence(monkeypatch, tmp_path: Path) -> None:
    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "explicit"), XDG_CACHE_HOME=str(tmp_path / "xdg"))
    assert paths.cvsuite_home() == tmp_path / "explicit"

    _snapshot(monkeypatch, XDG_CACHE_HOME=str(tmp_path / "xdg"))
    assert paths.cvsuite_home() == tmp_path / "xdg" / "cvsuite"

    _snapshot(monkeypatch)
    assert paths.cvsuite_home() == Path.home() / ".cache" / "cvsuite"


def test_derived_roots_follow_home(monkeypatch, tmp_path: Path) -> None:
    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "home"))
    assert paths.hub_dir() == tmp_path / "home" / "hub"
    assert paths.providers_dir() == tmp_path / "home" / "providers"

    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "home"), CVSUITE_HUB_DIR=str(tmp_path / "bigdisk"))
    assert paths.hub_dir() == tmp_path / "bigdisk"


def test_user_hf_hub_precedence(monkeypatch, tmp_path: Path) -> None:
    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "home"), HF_HUB_CACHE=str(tmp_path / "explicit"))
    assert paths.user_hf_hub_dir() == tmp_path / "explicit"

    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "home"), HF_HOME=str(tmp_path / "hfhome"))
    assert paths.user_hf_hub_dir() == tmp_path / "hfhome" / "hub"

    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "home"))
    assert paths.user_hf_hub_dir() == Path.home() / ".cache" / "huggingface" / "hub"


def test_user_hf_hub_is_none_when_it_is_our_own_hub(monkeypatch, tmp_path: Path) -> None:
    """Guards against symlinking the hub to itself when HF_HUB_CACHE already points here."""

    home = tmp_path / "home"
    (home / "hub").mkdir(parents=True)
    _snapshot(monkeypatch, CVSUITE_HOME=str(home), HF_HUB_CACHE=str(home / "hub"))
    assert paths.user_hf_hub_dir() is None


def test_env_snapshot_survives_later_environment_mutation(monkeypatch, tmp_path: Path) -> None:
    """ensure_hf_caches rewrites HF_HUB_CACHE at runtime; resolution must not follow it."""

    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "home"), HF_HUB_CACHE=str(tmp_path / "real-hf"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "stage" / "hub"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "stage" / "xdg"))

    assert paths.user_hf_hub_dir() == tmp_path / "real-hf"
    assert paths.cvsuite_home() == tmp_path / "home"


def test_shared_repo_target_accepts_complete_snapshot(monkeypatch, tmp_path: Path) -> None:
    hf_hub = tmp_path / "hf"
    _seed_repo(hf_hub, "facebook/sam3")
    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "home"), HF_HUB_CACHE=str(hf_hub))

    assert paths.shared_repo_target("facebook/sam3") == hf_hub / "models--facebook--sam3"


def test_shared_repo_target_rejects_dangling_ref(monkeypatch, tmp_path: Path) -> None:
    """A ref pointing at a snapshot that was never completed must not be shared."""

    hf_hub = tmp_path / "hf"
    repo_dir = hf_hub / "models--facebook--sam3"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "home"), HF_HUB_CACHE=str(hf_hub))

    assert paths.shared_repo_target("facebook/sam3") is None


def test_shared_repo_target_rejects_wrong_revision(monkeypatch, tmp_path: Path) -> None:
    hf_hub = tmp_path / "hf"
    _seed_repo(hf_hub, "facebook/sam3", revision="abc123")
    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "home"), HF_HUB_CACHE=str(hf_hub))

    assert paths.shared_repo_target("facebook/sam3", "deadbeef") is None


def test_shared_repo_target_rejects_snapshot_missing_required_file(monkeypatch, tmp_path: Path) -> None:
    """A user may have fetched only part of a repo; a partial share is worse than none."""

    hf_hub = tmp_path / "hf"
    _seed_repo(hf_hub, "facebook/sam2.1-hiera-small", files=("config.json",))
    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "home"), HF_HUB_CACHE=str(hf_hub))

    assert paths.shared_repo_target("facebook/sam2.1-hiera-small") is not None
    assert paths.shared_repo_target("facebook/sam2.1-hiera-small", require_patterns=["config.json"]) is not None
    assert paths.shared_repo_target("facebook/sam2.1-hiera-small", require_patterns=["model.safetensors"]) is None


def test_shared_repo_target_none_when_hf_cache_absent(monkeypatch, tmp_path: Path) -> None:
    _snapshot(monkeypatch, CVSUITE_HOME=str(tmp_path / "home"), HF_HUB_CACHE=str(tmp_path / "nope"))
    assert paths.shared_repo_target("facebook/sam3") is None


def test_link_shared_repo_is_idempotent_and_retargets(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    first = tmp_path / "hf" / "models--facebook--sam3"
    second = tmp_path / "other" / "models--facebook--sam3"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    link = paths.link_shared_repo(hub, "facebook/sam3", first)
    assert link.is_symlink() and link.resolve() == first.resolve()

    assert paths.link_shared_repo(hub, "facebook/sam3", first) == link
    relinked = paths.link_shared_repo(hub, "facebook/sam3", second)
    assert relinked.resolve() == second.resolve()


def test_drop_stale_share_unlinks_broken_link_but_keeps_real_dirs(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    hub.mkdir()
    target = tmp_path / "hf" / "models--facebook--sam3"
    _seed_repo(tmp_path / "hf", "facebook/sam3")

    link = paths.link_shared_repo(hub, "facebook/sam3", target)
    assert paths.drop_stale_share(link) is False

    import shutil

    shutil.rmtree(target)
    assert paths.drop_stale_share(link) is True
    assert not link.is_symlink()

    owned = hub / "models--laion--CLIP"
    owned.mkdir()
    assert paths.drop_stale_share(owned) is False
    assert owned.is_dir()


def test_drop_stale_share_never_deletes_the_link_target(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    hub.mkdir()
    hf = tmp_path / "hf"
    # Present but unusable: no refs/main and no snapshots.
    target = hf / "models--facebook--sam3"
    target.mkdir(parents=True)
    (target / "blobs").mkdir()
    (target / "blobs" / "deadbeef").write_text("weights", encoding="utf-8")

    link = paths.link_shared_repo(hub, "facebook/sam3", target)
    assert paths.drop_stale_share(link) is True
    assert not link.is_symlink()
    assert (target / "blobs" / "deadbeef").read_text(encoding="utf-8") == "weights"


@pytest.mark.parametrize("model_id", ["facebook/sam3", "laion/CLIP-ViT-H-14"])
def test_hf_repo_cache_dir_matches_hub_layout(tmp_path: Path, model_id: str) -> None:
    expected = tmp_path / f"models--{model_id.replace('/', '--')}"
    assert paths.hf_repo_cache_dir(tmp_path, model_id) == expected
