from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def test_cli_version_outputs_git_hash():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "timeflowcodec", "--version"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    out = (proc.stdout or "").strip()
    # Expect a short git hash in the version string when inside a git repo.
    assert re.search(r"[0-9a-f]{7,}", out), f"version output missing git hash: {out}"
