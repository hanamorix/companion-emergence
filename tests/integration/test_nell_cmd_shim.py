"""Executes the Windows ``nell.cmd`` shim for real under ``cmd.exe`` (#408, C3x).

The template is the file the Rust app embeds (``include_str!``), so this test and the app cannot
drift. The fake ``nell.bat`` echoes its arguments and exits 7. Cases: a path with spaces, a path
containing ``%``, and the ``%LOCALAPPDATA%`` spelling used when the profile path is not ASCII
(the directory name holds a character inside cp437/cp850 and one outside every OEM code page).

Windows only: cmd.exe is the oracle. The same text rules are covered on every OS by the Rust
tests in ``app/src-tauri/nell-shortcut/tests/text.rs``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="needs cmd.exe")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATE = REPO_ROOT / "app" / "src-tauri" / "nell-shortcut" / "src" / "nell_cmd.tmpl"

FAKE_BAT = "@echo off\r\necho ARGS=%*\r\nexit /b 7\r\n"


def _render(bat_expr: str) -> bytes:
    text = TEMPLATE.read_text(encoding="utf-8").replace("\r\n", "\n").replace("{NELL_BAT}", bat_expr)
    text = text.replace("\n", "\r\n")
    assert text.isascii(), "nell.cmd must stay ASCII (cmd reads it in the OEM code page, #261)"
    return text.encode("ascii")


def _run(cmd_file: Path, *args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(extra_env or {})}
    return subprocess.run(
        ["cmd", "/c", str(cmd_file), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cmd_file.parent),
    )


def _bat(dir_: Path) -> Path:
    bat = dir_ / "python-runtime" / "Scripts" / "nell.bat"
    bat.parent.mkdir(parents=True)
    bat.write_bytes(FAKE_BAT.encode("ascii"))
    return bat


def test_spaced_ascii_path_passes_args_and_exit_code(tmp_path: Path) -> None:
    bat = _bat(tmp_path / "Companion Emergence")
    shim = tmp_path / "nell.cmd"
    shim.write_bytes(_render(str(bat)))
    cp = _run(shim, "a", "b c")
    assert cp.returncode == 7, cp.stderr
    assert 'ARGS=a "b c"' in cp.stdout


def test_percent_in_the_path_is_doubled(tmp_path: Path) -> None:
    bat = _bat(tmp_path / "100%x")
    shim = tmp_path / "nell.cmd"
    shim.write_bytes(_render(str(bat).replace("%", "%%")))
    cp = _run(shim, "one")
    assert cp.returncode == 7, cp.stderr
    assert "ARGS=one" in cp.stdout


def test_local_app_data_spelling_works_with_a_non_ascii_profile_path(tmp_path: Path) -> None:
    profile = tmp_path / "José日"  # e-acute (inside cp437/cp850) + U+65E5 (outside every OEM page)
    local = profile / "AppData" / "Local"
    bat = _bat(local / "Companion Emergence")
    assert bat.exists()
    shim = tmp_path / "nell.cmd"
    shim.write_bytes(_render("%LOCALAPPDATA%\\Companion Emergence\\python-runtime\\Scripts\\nell.bat"))
    cp = _run(shim, "x", "", extra_env={"LOCALAPPDATA": str(local)})
    assert cp.returncode == 7, cp.stderr
    assert "ARGS=x" in cp.stdout
