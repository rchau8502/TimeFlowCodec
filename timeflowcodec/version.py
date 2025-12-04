from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict

__version__ = "0.1.0"


def _run_git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent, capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        return None
    return None


def get_build_meta() -> Dict[str, str]:
    git_hash = _run_git(["rev-parse", "--short", "HEAD"]) or "unknown"
    dirty = _run_git(["status", "--porcelain"])
    is_dirty = bool(dirty)
    return {
        "version": __version__,
        "git_hash": git_hash,
        "dirty": "1" if is_dirty else "0",
    }


def get_version_string() -> str:
    meta = get_build_meta()
    dirty_suffix = "+dirty" if meta.get("dirty") == "1" else ""
    return f"{meta.get('version')} ({meta.get('git_hash')}{dirty_suffix})"
