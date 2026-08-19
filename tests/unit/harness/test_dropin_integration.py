"""C7 real-brain proof for the drop-in capability — NETWORK-ENABLED, EXCLUDED from the CI marker set.

Marked ``@pytest.mark.integration`` so it is deselected by the default
``-m "not live and not requires_claude_cli and not integration"`` run. It ingests the REAL ``brain``
in this worktree with ``deps="auto"`` (``uv pip install -e`` — needs network/build backend) and proves,
end to end, that the drop-in venv + startup hook resolve ``brain`` from the copy and fail loud on a
shadowing source. Run by hand with the network enabled; record the result in ``8-harness.md``.

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

pytestmark = pytest.mark.integration


def _worktree_root() -> Path:
    """The worktree root (contains ``brain/``): three parents up from this test file."""
    root = Path(__file__).resolve().parents[3]
    assert (root / "brain" / "__init__.py").is_file(), root
    return root


def test_c7_real_brain_positive_resolves_under_copy_and_boots(tmp_path: Path) -> None:
    src = _worktree_root()
    dest = tmp_path / "dropin"
    build = ingest_version(src, dest, on_existing="delete", deps="auto", install_guard=True)

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


def test_c7_real_brain_negative_shadowed_source_exits_loud(tmp_path: Path) -> None:
    src = _worktree_root()
    dest = tmp_path / "dropin"
    build = ingest_version(src, dest, on_existing="delete", deps="auto", install_guard=True)

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
