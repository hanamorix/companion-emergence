"""Regression test for the bundled Windows ``nell.bat`` wrapper's bytes (#261).

``cmd.exe`` reads a ``.bat`` in the console's OEM codepage (cp850 / cp1252), not
UTF-8. The wrapper once carried an em-dash in a ``rem`` comment: on a non-UTF-8
console those three bytes split the line into extra tokens and every call printed
``'m' is not recognized as an internal or external command`` before doing its
work. LF-only endings (Git Bash heredoc) were the second hazard.

This test pulls the generator out of ``app/build_python_runtime.sh`` (the
source of truth, not a copy), runs it through bash, and pins the two properties
the fix guarantees: pure ASCII, and CRLF on every line.

**Unix-only.** The generator is a bash snippet; on Windows the bundled build runs
it under Git Bash, which this test does not assume is present.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the wrapper generator is a bash snippet; not exercised on a Windows runner",
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _extract_bat_generator() -> str:
    """The ``printf ... > "$NELL_BAT"`` block that writes the wrapper."""
    text = (REPO_ROOT / "app" / "build_python_runtime.sh").read_text(encoding="utf-8")
    match = re.search(
        r"^  (printf '%s\\r\\n' \\\n(?:    '.*' \\\n)+    > \"\$NELL_BAT\")$",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise RuntimeError(
            "could not find the nell.bat printf block in app/build_python_runtime.sh - "
            "refactor changed the marker; update _extract_bat_generator."
        )
    return match.group(1)


def _generate(tmp_path: Path) -> bytes:
    target = tmp_path / "Scripts" / "nell.bat"
    target.parent.mkdir()
    subprocess.run(
        ["bash", "-e", "-c", _extract_bat_generator()],
        env={"NELL_BAT": str(target), "PATH": "/usr/bin:/bin"},
        check=True,
        timeout=30,
    )
    return target.read_bytes()


def test_bat_wrapper_is_pure_ascii(tmp_path: Path) -> None:
    raw = _generate(tmp_path)
    non_ascii = [b for b in raw if b > 0x7F]
    assert not non_ascii, f"nell.bat carries non-ASCII bytes: {non_ascii[:8]!r}"


def test_bat_wrapper_lines_are_crlf_terminated(tmp_path: Path) -> None:
    raw = _generate(tmp_path)
    assert raw.endswith(b"\r\n")
    lines = raw.split(b"\r\n")[:-1]
    assert lines, "wrapper is empty"
    for line in lines:
        assert b"\n" not in line and b"\r" not in line, f"bare LF/CR inside {line!r}"


def test_bat_wrapper_launches_the_relative_python(tmp_path: Path) -> None:
    text = _generate(tmp_path).decode("ascii")
    assert text.startswith("@echo off\r\n")
    assert (
        '"%~dp0..\\python.exe" -c "import sys; from brain.cli import main; sys.exit(main())" %*'
        in text
    )
    for line in text.split("\r\n"):
        if line.startswith("rem "):
            # A comment must stay a single token on any codepage.
            assert line.isascii()
