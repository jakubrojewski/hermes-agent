#!/usr/bin/env python3
"""Plan/apply an exact owner-only permission migration.

Default mode is read-only planning. Apply accepts only a previously generated
plan whose complete candidate set, paths, owners, file types, device/inode
identities, and current modes still match. Symlinks and non-regular entries are
never followed or changed. The command refuses to run unless the active runtime
policy is explicitly 0700/0600.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
DEFAULT_ROOTS = (_HOME / "logs", _HOME / "cache" / "delegation")


def mode_string(mode: int) -> str:
    return oct(stat.S_IMODE(mode))


def expected_mode(mode: int) -> int | None:
    if stat.S_ISDIR(mode):
        return 0o700
    if stat.S_ISREG(mode):
        return 0o600
    return None


def snapshot_path(path: Path, uid: int) -> dict[str, Any]:
    st = path.lstat()
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"symlink refused: {path}")
    target = expected_mode(st.st_mode)
    if target is None:
        raise ValueError(f"unsupported file type: {path}")
    if st.st_uid != uid:
        raise ValueError(f"wrong owner uid={st.st_uid}: {path}")
    return {
        "path": str(path),
        "kind": "dir" if stat.S_ISDIR(st.st_mode) else "file",
        "device": st.st_dev,
        "inode": st.st_ino,
        "uid": st.st_uid,
        "gid": st.st_gid,
        "before": mode_string(st.st_mode),
        "after": oct(target),
    }


def iter_paths(roots: tuple[Path, ...]):
    for root in roots:
        if not root.exists():
            continue
        yield root
        for base, dirs, files in os.walk(root, topdown=True, followlinks=False):
            base_path = Path(base)
            safe_dirs = []
            for name in sorted(dirs):
                candidate = base_path / name
                if candidate.is_symlink():
                    raise ValueError(f"symlink refused: {candidate}")
                safe_dirs.append(name)
                yield candidate
            dirs[:] = safe_dirs
            for name in sorted(files):
                candidate = base_path / name
                if candidate.is_symlink():
                    raise ValueError(f"symlink refused: {candidate}")
                yield candidate


def make_plan(roots: tuple[Path, ...]) -> dict[str, Any]:
    uid = os.getuid()
    candidates = []
    normalized_roots = tuple(
        dict.fromkeys(Path(os.path.abspath(os.path.normpath(root))) for root in roots)
    )
    seen_inodes: set[tuple[int, int]] = set()
    for path in iter_paths(normalized_roots):
        item = snapshot_path(path, uid)
        identity = (item["device"], item["inode"])
        if identity in seen_inodes:
            continue
        seen_inodes.add(identity)
        if item["before"] != item["after"]:
            candidates.append(item)
    candidates.sort(key=lambda item: item["path"])
    return {
        "schema": 1,
        "uid": uid,
        "roots": [str(root) for root in normalized_roots],
        "candidates": candidates,
    }


def canonical_hash(plan: dict[str, Any]) -> str:
    payload = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_item(item: dict[str, Any], uid: int) -> os.stat_result:
    path = Path(item["path"])
    current = snapshot_path(path, uid)
    for key in ("kind", "device", "inode", "uid", "gid", "before", "after"):
        if current[key] != item[key]:
            raise ValueError(f"preflight drift {key}: {path}")
    return path.lstat()


def apply_plan(plan: dict[str, Any]) -> int:
    uid = os.getuid()
    if plan.get("schema") != 1 or plan.get("uid") != uid:
        raise ValueError("plan schema/uid mismatch")
    items = plan.get("candidates")
    roots = plan.get("roots")
    if not isinstance(items, list) or not isinstance(roots, list):
        raise ValueError("invalid candidates/roots")

    current = make_plan(tuple(Path(root) for root in roots))
    if current["candidates"] != items:
        raise ValueError("candidate set drifted; generate a fresh plan")

    # Open and bind every candidate descriptor before the first mutation.  This
    # closes the path-swap window between preflight and apply and also ensures
    # overlapping roots cannot partially mutate the same inode twice.
    opened: list[tuple[dict[str, Any], int]] = []
    for item in items:
        validate_item(item, uid)
        path = Path(item["path"])
        flags = os.O_RDONLY
        if item["kind"] == "dir":
            flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except Exception:
            for _opened_item, opened_fd in opened:
                os.close(opened_fd)
            raise
        try:
            st = os.fstat(fd)
            actual_kind = "dir" if stat.S_ISDIR(st.st_mode) else "file" if stat.S_ISREG(st.st_mode) else "other"
            if (
                actual_kind,
                st.st_dev,
                st.st_ino,
                st.st_uid,
                st.st_gid,
                mode_string(st.st_mode),
            ) != (
                item["kind"],
                item["device"],
                item["inode"],
                item["uid"],
                item["gid"],
                item["before"],
            ):
                raise ValueError(f"apply-time drift: {path}")
            opened.append((item, fd))
        except Exception:
            os.close(fd)
            for _opened_item, opened_fd in opened:
                os.close(opened_fd)
            raise

    changed: list[tuple[dict[str, Any], int]] = []
    try:
        for item, fd in opened:
            os.fchmod(fd, int(item["after"], 8))
            changed.append((item, fd))
        return len(changed)
    except Exception as exc:
        rollback_errors = []
        for item, fd in reversed(changed):
            try:
                os.fchmod(fd, int(item["before"], 8))
            except OSError as rollback_exc:
                rollback_errors.append(f"{item['path']}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                f"apply failed ({exc}); rollback incomplete: {'; '.join(rollback_errors)}"
            ) from exc
        raise
    finally:
        for _item, fd in opened:
            os.close(fd)


def require_owner_only_policy() -> None:
    from utils import sensitive_artifact_modes

    if sensitive_artifact_modes() != (0o700, 0o600):
        raise ValueError(
            "owner-only migration requires the active 0700/0600 deployment policy"
        )


def main() -> int:
    require_owner_only_policy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--apply", type=Path)
    parser.add_argument("--expect-sha256")
    args = parser.parse_args()
    if args.plan and args.apply:
        parser.error("choose --plan or --apply")

    if args.apply:
        plan = json.loads(args.apply.read_text(encoding="utf-8"))
        digest = canonical_hash(plan)
        if not args.expect_sha256 or digest != args.expect_sha256:
            raise ValueError("exact --expect-sha256 is required and must match")
        changed = apply_plan(plan)
        print(
            "owner_only_permission_migration=APPLIED "
            f"changed={changed} plan_sha256={digest}"
        )
        return 0

    roots = tuple(args.root) if args.root else DEFAULT_ROOTS
    plan = make_plan(roots)
    digest = canonical_hash(plan)
    output = args.plan
    if output:
        parent = output.parent
        if not parent.is_dir() or parent.is_symlink():
            raise ValueError("plan output parent must be an existing real directory")
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        parent_flags |= getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open(parent, parent_flags)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = -1
        try:
            fd = os.open(output.name, flags, 0o600, dir_fd=parent_fd)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = -1
                json.dump(plan, handle, indent=2)
                handle.write("\n")
        finally:
            if fd >= 0:
                os.close(fd)
            os.close(parent_fd)
    print(
        "owner_only_permission_migration=PLAN "
        f"candidates={len(plan['candidates'])} plan_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"owner_only_permission_migration=BLOCKED error={exc}", file=sys.stderr)
        raise SystemExit(1)
