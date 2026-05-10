"""Regression test for the bundled nell wrapper's symlink-resolution.

The desktop app's "install nell to ~/.local/bin" button writes a symlink
at ``~/.local/bin/nell`` pointing at the bundled
``<App>/Contents/Resources/python-runtime/bin/nell`` wrapper. The
wrapper has to resolve ``$0`` through that symlink before computing
``SCRIPT_DIR`` — otherwise ``dirname "$0"`` returns ``~/.local/bin``
and ``exec "$SCRIPT_DIR/python3"`` fails to find python3 there.

Live smoke (2026-05-10) caught a wrapper that DIDN'T resolve through
symlinks; this test pins the fix so a future template tweak can't
reintroduce it.

Test shape: build a fake bundled layout in a tempdir (stub python3 +
the wrapper template from app/build_python_runtime.sh), create a
symlink elsewhere, invoke the symlink, assert it reaches the bundled
python3.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _extract_unix_wrapper_template() -> str:
    """Pull the heredoc body of the Unix nell wrapper out of build_python_runtime.sh.

    Keeps the test reading from the canonical source-of-truth instead of
    duplicating the wrapper body — a refactor of the build script that
    breaks this assertion is exactly what we want to catch.
    """
    build_script = REPO_ROOT / "app" / "build_python_runtime.sh"
    text = build_script.read_text(encoding="utf-8")
    match = re.search(
        r"cat > \"\$NELL_BIN\" <<'NELL_WRAPPER'\n(.*?)\nNELL_WRAPPER",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError(
            "could not find NELL_WRAPPER heredoc in app/build_python_runtime.sh — "
            "refactor changed the marker; update _extract_unix_wrapper_template."
        )
    return match.group(1) + "\n"


def _make_bundled_layout(td: Path) -> Path:
    """Build a fake <App>/Contents/Resources/python-runtime/bin/ with a
    stub python3 + the real wrapper template. Returns the bundled nell
    path."""
    bundled_bin = td / "Companion Emergence.app/Contents/Resources/python-runtime/bin"
    bundled_bin.mkdir(parents=True)

    # Stub python3 — print the args so the test can verify the wrapper
    # reached the bundled python and not some other one on PATH.
    python3 = bundled_bin / "python3"
    python3.write_text(
        '#!/bin/sh\n'
        'echo "BUNDLED_PYTHON_INVOKED $0"\n'
        # Suppress the "from brain.cli import main" - we don't have it here.
        'echo "$@" | grep -q "from brain.cli" && exit 0\n'
        'exit 0\n'
    )
    python3.chmod(0o755)

    bundled_nell = bundled_bin / "nell"
    bundled_nell.write_text(_extract_unix_wrapper_template())
    bundled_nell.chmod(0o755)
    return bundled_nell


def test_wrapper_works_when_invoked_via_symlink():
    """The ~/.local/bin/nell symlink path must reach the bundled python3."""
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        bundled_nell = _make_bundled_layout(td)

        user_local_bin = td / "home/.local/bin"
        user_local_bin.mkdir(parents=True)
        symlink_target = user_local_bin / "nell"
        symlink_target.symlink_to(bundled_nell)

        result = subprocess.run(
            [str(symlink_target)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"symlinked invocation failed:\n"
            f"  stdout: {result.stdout!r}\n"
            f"  stderr: {result.stderr!r}"
        )
        assert "BUNDLED_PYTHON_INVOKED" in result.stdout, (
            f"wrapper did not reach bundled python3 — got: {result.stdout!r}"
        )


def test_wrapper_works_when_invoked_directly():
    """Direct invocation of the bundled path must also still work."""
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        bundled_nell = _make_bundled_layout(td)

        result = subprocess.run(
            [str(bundled_nell)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "BUNDLED_PYTHON_INVOKED" in result.stdout


def test_wrapper_works_through_chained_symlinks():
    """A symlink-of-a-symlink (rare but possible if the user re-runs the
    install button after manually moving the link) still resolves."""
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        bundled_nell = _make_bundled_layout(td)

        # First symlink at ~/.local/bin/nell.
        first = td / "home/.local/bin/nell"
        first.parent.mkdir(parents=True)
        first.symlink_to(bundled_nell)

        # Second symlink elsewhere pointing at the first.
        second_dir = td / "other-bin"
        second_dir.mkdir()
        second = second_dir / "nell"
        second.symlink_to(first)

        result = subprocess.run(
            [str(second)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "BUNDLED_PYTHON_INVOKED" in result.stdout


def test_wrapper_works_with_relative_symlink_target():
    """If the user (or a tool) creates a relative-path symlink, the
    wrapper still resolves it correctly."""
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        bundled_nell = _make_bundled_layout(td)

        # Symlink with a relative target — must walk the case branch in
        # the resolver loop that prepends LINK_DIR.
        sym_dir = td / "some-other-dir"
        sym_dir.mkdir()
        rel_target = os.path.relpath(bundled_nell, sym_dir)
        symlink = sym_dir / "nell-link"
        symlink.symlink_to(rel_target)

        result = subprocess.run(
            [str(symlink)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "BUNDLED_PYTHON_INVOKED" in result.stdout
