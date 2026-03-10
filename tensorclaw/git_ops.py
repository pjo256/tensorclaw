from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .shell import run_git


@dataclass
class GitState:
    branch: str
    commit: str


def _normalize_relpath(path: str, cwd: str) -> Optional[str]:
    resolved = Path(path).expanduser().resolve()
    root = Path(cwd).expanduser().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return None


def get_state(cwd: str) -> GitState:
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd).stdout.strip()
    commit = run_git(["rev-parse", "--short", "HEAD"], cwd=cwd).stdout.strip()
    return GitState(branch=branch, commit=commit)


def has_changes(cwd: str) -> bool:
    status = run_git(["status", "--porcelain"], cwd=cwd).stdout.strip()
    return bool(status)


def commit_all(cwd: str, message: str, exclude_paths: list[str] | None = None) -> Optional[str]:
    if not has_changes(cwd):
        return None

    run_git(["add", "-A"], cwd=cwd)

    for path in exclude_paths or []:
        relpath = _normalize_relpath(path, cwd)
        if relpath:
            run_git(["reset", "-q", "HEAD", "--", relpath], cwd=cwd, check=False)

    staged = run_git(["diff", "--cached", "--name-only"], cwd=cwd).stdout.strip()
    if not staged:
        return None

    run_git(["commit", "-m", message], cwd=cwd)
    return run_git(["rev-parse", "--short", "HEAD"], cwd=cwd).stdout.strip()


def reset_hard(cwd: str, commit: str) -> None:
    run_git(["reset", "--hard", commit], cwd=cwd)
