from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import sweep_bench


class CodeProvenance(TypedDict):
    commit: str | None
    dirty: bool | None
    package: str
    writer_sha: str | None
    writer: str | None
    package_dirty: bool | None
    package_tree: str | None
    dirty_hash: str | None
    package_dirty_hash: str | None
    dirty_paths: list[str]
    unknown_reason: str | None


_PORCELAIN = re.compile(r"^\s*(\S{1,2})\s+(.*)$")


def _package_diff_hash(tree: Path, paths: tuple[str, ...]) -> str | None:
    if not paths:
        return None
    diff = _git(tree, "diff", "HEAD", "--", *paths)
    return hashlib.sha256(diff.encode()).hexdigest() if diff is not None else None


def _porcelain_paths(status: str | None) -> tuple[str, ...]:
    if not status:
        return ()
    out: list[str] = []
    for line in status.splitlines():
        m = _PORCELAIN.match(line)
        if m is None:
            continue
        path = m.group(2)
        for part in path.split(" -> ", 1) if " -> " in path else [path]:
            out.append(part.strip().strip('"'))
    return tuple(out)


def home_relative(path: Path) -> str:
    home = Path.home().resolve()
    return str(Path("~") / path.relative_to(home)) if path.is_relative_to(home) else str(path)


def _git(tree: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(tree), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def code_provenance() -> CodeProvenance:
    package = Path(sweep_bench.__file__).resolve()
    tree = package.parent.parent
    toplevel = _git(tree, "rev-parse", "--show-toplevel")
    reason: str | None = None
    if toplevel is None:
        reason = f"{home_relative(tree)} is not a git work tree (installed wheel, or git unavailable)"
    elif Path(toplevel).resolve() != tree.resolve():
        reason = (
            f"{home_relative(tree)} is not the work tree of the repository at "
            f"{home_relative(Path(toplevel).resolve())}; the package is nested inside it"
        )
    commit = _git(tree, "rev-parse", "HEAD") if reason is None else None
    if reason is None and commit is None:
        reason = f"git could not read HEAD at {home_relative(tree)}"
    status = _git(tree, "status", "--porcelain") if commit else None
    writer_path: str | None = None
    writer_sha: str | None = None
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        try:
            wp = Path(argv0).resolve()
            if wp.is_file():
                writer_path = home_relative(wp)
                writer_sha = hashlib.sha256(wp.read_bytes()).hexdigest()
        except OSError:
            pass
    paths = _porcelain_paths(status)
    package_tree = _git(tree, "rev-parse", "HEAD:sweep_bench") if commit else None
    diff = _git(tree, "diff", "HEAD") if commit else None
    return CodeProvenance(
        commit=commit,
        package_tree=package_tree,
        dirty_hash=hashlib.sha256((diff or "").encode()).hexdigest() if diff is not None else None,
        package_dirty_hash=_package_diff_hash(tree, paths) if commit else None,
        dirty=None if status is None else bool(status),
        package_dirty=None if status is None else bool(paths),
        dirty_paths=list(paths),
        package=home_relative(package),
        unknown_reason=reason,
        writer_sha=writer_sha,
        writer=writer_path,
    )
