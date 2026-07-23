import os
from pathlib import Path

import pytest

from utils import (
    append_restricted_text,
    ensure_restricted_directory,
    sensitive_artifact_modes,
    write_restricted_text,
)

POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")


@pytest.fixture
def deployment_policy(monkeypatch):
    import hermes_cli.config as config

    monkeypatch.delenv("HERMES_SKIP_CHMOD", raising=False)
    monkeypatch.delenv("HERMES_FORCE_OWNER_ONLY", raising=False)
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    monkeypatch.setattr(config, "is_managed", lambda: False)
    monkeypatch.setattr(config, "_is_container", lambda: False)
    return config


@POSIX_ONLY
def test_sensitive_modes_native_unmanaged_are_owner_only(deployment_policy):
    assert sensitive_artifact_modes() == (0o700, 0o600)


@POSIX_ONLY
def test_sensitive_modes_managed_are_group_shared(deployment_policy, monkeypatch):
    monkeypatch.setattr(deployment_policy, "is_managed", lambda: True)
    assert sensitive_artifact_modes() == (0o2770, 0o660)


def test_sensitive_modes_container_preserves_volume_policy(deployment_policy, monkeypatch):
    monkeypatch.setattr(deployment_policy, "_is_container", lambda: True)
    assert sensitive_artifact_modes() == (None, None)


@POSIX_ONLY
def test_sensitive_modes_container_can_explicitly_force_owner_only(
    deployment_policy, monkeypatch
):
    monkeypatch.setattr(deployment_policy, "_is_container", lambda: True)
    monkeypatch.setenv("HERMES_FORCE_OWNER_ONLY", "1")
    assert sensitive_artifact_modes() == (0o700, 0o600)


def test_sensitive_modes_skip_chmod_wins_over_force(deployment_policy, monkeypatch):
    monkeypatch.setenv("HERMES_SKIP_CHMOD", "1")
    monkeypatch.setenv("HERMES_FORCE_OWNER_ONLY", "1")
    assert sensitive_artifact_modes() == (None, None)


@POSIX_ONLY
def test_sensitive_modes_skip_chmod_wins_in_managed_mode(
    deployment_policy, monkeypatch
):
    monkeypatch.setattr(deployment_policy, "is_managed", lambda: True)
    monkeypatch.setenv("HERMES_SKIP_CHMOD", "1")
    monkeypatch.setenv("HERMES_FORCE_OWNER_ONLY", "1")
    assert sensitive_artifact_modes() == (None, None)


@POSIX_ONLY
def test_sensitive_modes_respect_explicit_home_mode(deployment_policy, monkeypatch):
    monkeypatch.setenv("HERMES_HOME_MODE", "0710")
    assert sensitive_artifact_modes() == (0o710, 0o600)


@POSIX_ONLY
def test_restricted_directory_tightens_existing_mode(tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir(mode=0o755)
    os.chmod(directory, 0o755)

    ensure_restricted_directory(directory)

    assert directory.stat().st_mode & 0o777 == 0o700


@POSIX_ONLY
def test_restricted_text_tightens_existing_file_before_write(tmp_path):
    path = tmp_path / "cache" / "result.txt"
    path.parent.mkdir()
    path.write_text("old", encoding="utf-8")
    os.chmod(path, 0o644)

    write_restricted_text(path, "new")

    assert path.read_text(encoding="utf-8") == "new"
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600


@POSIX_ONLY
def test_restricted_append_recreates_owner_only_after_unlink(tmp_path):
    path = tmp_path / "cache" / "events.log"
    write_restricted_text(path, "header\n")
    path.unlink()

    old_umask = os.umask(0o022)
    try:
        append_restricted_text(path, "event\n")
    finally:
        os.umask(old_umask)

    assert path.read_text(encoding="utf-8") == "event\n"
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="platform has no O_NOFOLLOW")
def test_restricted_append_refuses_symlink(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("unchanged", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    with pytest.raises(OSError):
        append_restricted_text(link, "secret")

    assert target.read_text(encoding="utf-8") == "unchanged"


@POSIX_ONLY
def test_restricted_directory_refuses_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(OSError):
        ensure_restricted_directory(link)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="platform has no O_NOFOLLOW")
def test_restricted_write_refuses_symlink(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("unchanged", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    with pytest.raises(OSError):
        write_restricted_text(link, "replacement")

    assert target.read_text(encoding="utf-8") == "unchanged"
