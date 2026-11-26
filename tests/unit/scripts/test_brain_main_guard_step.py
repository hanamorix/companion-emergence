"""The brain-main "Newer than what is published?" step must not publish blind (#363).

It used to read `gh release download ... 2>/dev/null || true`, which feeds `should-publish`, which
treats a missing published manifest as "nothing published yet". So ANY gh failure (a network blip, a
rate limit, an auth error) published this build even when it was older than what is live, and the
reason was thrown away. The real script is run here against a fake `gh`.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash step; runs on ubuntu CI")

REPO = Path(__file__).resolve().parents[3]
_WF = yaml.safe_load((REPO / ".github" / "workflows" / "brain-main.yml").read_text(encoding="utf-8"))
_GUARD = next(
    s for s in _WF["jobs"]["publish"]["steps"] if s.get("name", "").startswith("Newer than what is published")
)


def _run_guard(tmp_path: Path, gh_stderr: str, gh_rc: int = 1) -> tuple[subprocess.CompletedProcess, str]:
    fake = tmp_path / "bin"
    fake.mkdir()
    gh = fake / "gh"
    gh.write_text(f"#!/bin/sh\necho '{gh_stderr}' >&2\nexit {gh_rc}\n", encoding="utf-8")
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / "github_output"
    out.write_text("", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_EVENT_NAME": "workflow_run",
        "GITHUB_OUTPUT": str(out),
        "RUNNER_TEMP": str(tmp_path),
        "TAG": "brain-main",
        "SHA": "0" * 40,
    }
    r = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", _GUARD["run"]],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return r, out.read_text(encoding="utf-8")


def test_a_gh_failure_that_is_not_a_missing_release_stops_the_job_instead_of_publishing(tmp_path) -> None:
    r, output = _run_guard(tmp_path, "HTTP 502: Bad Gateway")
    assert r.returncode != 0
    assert "publish=" not in output
    assert "HTTP 502" in r.stderr  # the reason is no longer discarded


@pytest.mark.parametrize("gh_stderr", ["release not found", "no assets match the file pattern"])
def test_a_missing_release_or_manifest_still_means_publish(tmp_path, gh_stderr) -> None:
    # the two messages `gh release download` prints (gh 2.89, checked against this repo)
    r, output = _run_guard(tmp_path, gh_stderr)
    assert r.returncode == 0, r.stderr
    assert "publish=true" in output
