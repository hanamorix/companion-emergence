"""C7 real-brain proof for the drop-in capability — NETWORK-ENABLED.

Marked ``@pytest.mark.integration`` so the local pre-check gate
``-m "not live and not requires_claude_cli and not integration"`` deselects it. The CI workflow
(``.github/workflows/test.yml``) runs bare ``pytest`` with no ``-m`` filter, so it DOES run there, on all three
operating systems. It ingests the REAL ``brain`` in this worktree with ``deps="auto"`` (``uv pip install -e`` —
needs network/build backend) and proves, end to end, that the drop-in venv + startup hook resolve ``brain`` from
the copy and fail loud on a shadowing source. The install is the expensive part (about 50 s on the Windows
runner), so one module fixture does it once and both tests use that build; neither test changes it.

NOT run by the offline builder. Out of scope here (F4 residual): whether the external ``claude`` CLI
escalates a dead ``brain-tools`` MCP child into a run-abort vs proceeding tool-less — that needs a live
``claude`` turn, deferred to the Testing lane.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.harness import ingest_version

# #540: the network install of a drop-in can legitimately outlast the default per-test timeout in pyproject.toml.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(3600), pytest.mark.xdist_group("network-install")]


def _worktree_root() -> Path:
    """The worktree root (contains ``brain/``): three parents up from this test file."""
    root = Path(__file__).resolve().parents[3]
    assert (root / "brain" / "__init__.py").is_file(), root
    return root


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory):
    """The drop-in copy of the real brain with its venv, installed once for the module."""
    dest = tmp_path_factory.mktemp("dropin") / "dropin"
    return ingest_version(_worktree_root(), dest, on_existing="delete", deps="auto", install_guard=True)


def test_c7_real_brain_positive_resolves_under_copy_and_boots(build, tmp_path: Path) -> None:

    # POSITIVE: the venv python imports the REAL brain + brain.mcp_server, resolving under the copy
    # (hook ran and passed; deps installed so the import succeeds).
    proc = subprocess.run(
        [
            str(build.python),
            "-c",
            "import brain, brain.mcp_server; print(brain.__file__)",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    resolved = Path(proc.stdout.strip())
    assert str(resolved).startswith(str(build.repo.resolve())), (resolved, build.repo)

    # Boot the exact production launch construction briefly, then terminate.
    persona_dir = tmp_path / "personas" / "Canary"
    persona_dir.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [str(build.python), "-m", "brain.mcp_server", "--persona-dir", str(persona_dir)],
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        try:
            server.wait(timeout=3)
            # If it exited on its own, it must not have exited 70 (the guard fired wrongly).
            assert server.returncode != 70, server.stderr.read().decode(errors="replace")
        except subprocess.TimeoutExpired:
            pass  # still running under the venv = booted successfully
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


def test_c7_real_brain_negative_shadowed_source_exits_loud(build, tmp_path: Path) -> None:
    src = _worktree_root()

    # NEGATIVE: point PYTHONPATH at the worktree source (a DIFFERENT real brain checkout, outside the
    # copy). The hook must make the process exit non-zero and loud.
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    proc = subprocess.run(
        [str(build.python), "-c", "import brain"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 70, (proc.returncode, proc.stdout, proc.stderr)
    assert "ce-dropin guard" in proc.stderr


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-m", "integration", "-v"]))
