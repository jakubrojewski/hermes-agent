import importlib.util
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "migrate_owner_only_permissions.py"
_SPEC = importlib.util.spec_from_file_location("owner_only_migration", SCRIPT)
assert _SPEC and _SPEC.loader
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)

POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")


def _broad_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "logs"
    root.mkdir(mode=0o755)
    os.chmod(root, 0o755)
    nested = root / "archive"
    nested.mkdir(mode=0o755)
    os.chmod(nested, 0o755)
    active = root / "agent.log"
    backup = nested / "agent.log.1"
    active.write_text("active", encoding="utf-8")
    backup.write_text("backup", encoding="utf-8")
    os.chmod(active, 0o644)
    os.chmod(backup, 0o644)
    return root, nested, active, backup


@POSIX_ONLY
def test_plan_apply_covers_active_and_rotated_logs(tmp_path):
    root, nested, active, backup = _broad_tree(tmp_path)
    plan = migration.make_plan((root,))

    assert {Path(item["path"]).name for item in plan["candidates"]} == {
        "logs",
        "archive",
        "agent.log",
        "agent.log.1",
    }
    assert migration.apply_plan(plan) == 4
    assert root.stat().st_mode & 0o777 == 0o700
    assert nested.stat().st_mode & 0o777 == 0o700
    assert active.stat().st_mode & 0o777 == 0o600
    assert backup.stat().st_mode & 0o777 == 0o600
    assert migration.make_plan((root,))["candidates"] == []


@POSIX_ONLY
def test_symlink_blocks_planning(tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    target = root / "target"
    target.write_text("x", encoding="utf-8")
    (root / "link").symlink_to(target)

    with pytest.raises(ValueError, match="symlink refused"):
        migration.make_plan((root,))


@POSIX_ONLY
def test_mode_drift_blocks_before_first_mutation(tmp_path):
    root, _nested, active, backup = _broad_tree(tmp_path)
    plan = migration.make_plan((root,))
    os.chmod(backup, 0o640)

    with pytest.raises(ValueError, match="candidate set drifted"):
        migration.apply_plan(plan)

    assert root.stat().st_mode & 0o777 == 0o755
    assert active.stat().st_mode & 0o777 == 0o644


@POSIX_ONLY
def test_new_rotated_backup_invalidates_stale_plan(tmp_path):
    root, _nested, active, _backup = _broad_tree(tmp_path)
    plan = migration.make_plan((root,))
    newer = root / "agent.log.2"
    newer.write_text("new", encoding="utf-8")
    os.chmod(newer, 0o644)

    with pytest.raises(ValueError, match="candidate set drifted"):
        migration.apply_plan(plan)

    assert root.stat().st_mode & 0o777 == 0o755
    assert active.stat().st_mode & 0o777 == 0o644
    assert newer.stat().st_mode & 0o777 == 0o644


@POSIX_ONLY
def test_hash_changes_when_plan_changes(tmp_path):
    root, *_ = _broad_tree(tmp_path)
    plan = migration.make_plan((root,))
    first = migration.canonical_hash(plan)
    plan["candidates"][0]["before"] = "0o750"
    assert migration.canonical_hash(plan) != first


@POSIX_ONLY
def test_overlapping_roots_are_deduplicated_before_apply(tmp_path):
    root, nested, active, backup = _broad_tree(tmp_path)
    plan = migration.make_plan((root, nested, root))

    identities = [(item["device"], item["inode"]) for item in plan["candidates"]]
    assert len(identities) == len(set(identities)) == 4
    assert migration.apply_plan(plan) == 4
    assert active.stat().st_mode & 0o777 == 0o600
    assert backup.stat().st_mode & 0o777 == 0o600


@POSIX_ONLY
def test_apply_failure_rolls_back_prior_modes(tmp_path, monkeypatch):
    root, nested, active, backup = _broad_tree(tmp_path)
    plan = migration.make_plan((root,))
    real_fchmod = migration.os.fchmod
    calls = 0

    def fail_second_fchmod(fd, mode):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected apply failure")
        return real_fchmod(fd, mode)

    monkeypatch.setattr(migration.os, "fchmod", fail_second_fchmod)
    with pytest.raises(OSError, match="injected apply failure"):
        migration.apply_plan(plan)

    assert root.stat().st_mode & 0o777 == 0o755
    assert nested.stat().st_mode & 0o777 == 0o755
    assert active.stat().st_mode & 0o777 == 0o644
    assert backup.stat().st_mode & 0o777 == 0o644


@POSIX_ONLY
def test_plan_output_preserves_parent_mode(tmp_path, monkeypatch):
    root, *_ = _broad_tree(tmp_path)
    output_parent = tmp_path / "shared"
    output_parent.mkdir()
    os.chmod(output_parent, 0o1777)
    output = output_parent / "plan.json"
    monkeypatch.setattr(migration, "DEFAULT_ROOTS", (root,))
    monkeypatch.setattr(migration, "require_owner_only_policy", lambda: None)
    monkeypatch.setattr(migration.sys, "argv", ["migrate", "--plan", str(output)])

    assert migration.main() == 0
    assert output.is_file()
    assert output.stat().st_mode & 0o777 == 0o600
    assert output_parent.stat().st_mode & 0o7777 == 0o1777


@POSIX_ONLY
def test_plan_output_refuses_symlink_parent(tmp_path, monkeypatch):
    root, *_ = _broad_tree(tmp_path)
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    link_parent = tmp_path / "linked"
    link_parent.symlink_to(real_parent, target_is_directory=True)
    monkeypatch.setattr(migration, "DEFAULT_ROOTS", (root,))
    monkeypatch.setattr(migration, "require_owner_only_policy", lambda: None)
    monkeypatch.setattr(
        migration.sys, "argv", ["migrate", "--plan", str(link_parent / "plan.json")]
    )

    with pytest.raises(ValueError, match="existing real directory"):
        migration.main()
    assert not (real_parent / "plan.json").exists()


def test_cli_policy_gate_rejects_non_owner_only(monkeypatch):
    import utils

    monkeypatch.setattr(utils, "sensitive_artifact_modes", lambda: (None, None))
    with pytest.raises(ValueError, match="requires the active 0700/0600"):
        migration.require_owner_only_policy()
