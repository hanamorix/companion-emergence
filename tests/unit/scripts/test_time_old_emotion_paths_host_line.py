"""scripts/time_old_emotion_paths.py - the CPU model on the host line.

On Linux the script reads the ``model name`` field of /proc/cpuinfo itself; on any other system, or when the field is
missing or unreadable, it prints "unknown". It starts no process. The tests point the script's ``CPUINFO`` constant at a
file in ``tmp_path`` and use sentinel model names, never a real CPU's.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import platform
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "time_old_emotion_paths.py"

_spec = importlib.util.spec_from_file_location("time_old_emotion_paths_under_test", SCRIPT)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


def _cpu_model_on(monkeypatch: pytest.MonkeyPatch, cpuinfo: Path, platform_name: str = "linux") -> str:
    """Call ``_cpu_model`` with CPUINFO pointed at ``cpuinfo`` and sys.platform patched for that call only."""
    monkeypatch.setattr(mod, "CPUINFO", cpuinfo)
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "platform", platform_name)
        return mod._cpu_model()


def test_cpu_model_reads_first_model_name_collapsed(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_bytes(
        b"processor\t: 0\nmodel name\t:  Sentinel   CPU\t9000 \nflags\t\t: fpu avx2\n\n"
        b"processor\t: 1\nmodel name\t: Second Sentinel\n"
    )
    assert _cpu_model_on(monkeypatch, cpuinfo) == "Sentinel CPU 9000"


def test_cpu_model_matches_the_field_name_exactly(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_bytes(b"model\t: A\nmodel name2\t: B\nno colon on this line\nmodel name\t: C\n")
    assert _cpu_model_on(monkeypatch, cpuinfo) == "C"


@pytest.mark.parametrize(
    "content",
    [
        None,
        b"processor\t: 0\nflags\t: fpu\n",
        b"model name\t:   \n",
        b"no colon anywhere\n",
        b"flags\t: \xff\xfe\n",
    ],
    ids=["file-absent", "no-field", "empty-value", "no-colon", "bad-bytes-no-field"],
)
def test_cpu_model_unknown_when_missing(tmp_path, monkeypatch, content):
    cpuinfo = tmp_path / "cpuinfo"
    if content is not None:
        cpuinfo.write_bytes(content)
    assert _cpu_model_on(monkeypatch, cpuinfo) == "unknown"


def test_cpu_model_survives_undecodable_bytes(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_bytes(b"flags\t: \xff\xfe\nmodel name\t: Sentinel CPU\n")
    assert _cpu_model_on(monkeypatch, cpuinfo) == "Sentinel CPU"


def test_cpu_model_unknown_when_unreadable(tmp_path, monkeypatch):
    # A directory cannot be read as text: the read raises OSError.
    assert _cpu_model_on(monkeypatch, tmp_path) == "unknown"


@pytest.mark.parametrize("platform_name", ["darwin", "win32"])
def test_cpu_model_unknown_off_linux(tmp_path, monkeypatch, platform_name):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_bytes(b"model name\t: Sentinel CPU\n")
    assert _cpu_model_on(monkeypatch, cpuinfo, platform_name) == "unknown"


def test_host_line_prints_cpu_model_and_arch(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_bytes(b"model name\t: Sentinel CPU 9000\n")
    monkeypatch.setattr(mod, "CPUINFO", cpuinfo)
    # Only the platform string is patched, and only for the linux case, so the app's AVX2 probe is never
    # called under a non-linux value. The avx2 and ram values come from the app's probes (real /proc files) and
    # are not asserted.
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "platform", "linux")
        line = mod._host_line()
    assert line.startswith("host: ")
    assert "cpu Sentinel CPU 9000" in line
    assert f"arch {platform.machine() or 'unknown'}" in line
    assert "avx2=" in line
    assert "cores " in line
    assert "ram " in line


def test_cpu_model_starts_no_process(tmp_path, monkeypatch):
    started: list[object] = []

    class SpyPopen:
        def __init__(self, *args, **kwargs):
            started.append((args, kwargs))
            raise AssertionError("a process was started")

    monkeypatch.setattr(subprocess, "Popen", SpyPopen)
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_bytes(b"model name\t: Sentinel CPU\n")
    assert _cpu_model_on(monkeypatch, cpuinfo) == "Sentinel CPU"  # also points mod.CPUINFO at the file
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "platform", "linux")
        mod._host_line()
    assert started == []

    # The same property read from the source: no call to subprocess, platform.processor, os.popen or os.system
    # inside _cpu_model itself (a walk of that one function, so a comment elsewhere is not a hit).
    tree = ast.parse(inspect.getsource(mod._cpu_model))
    banned = {"subprocess", "processor", "popen", "system"}
    called = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in (
            [node.func.attr, ast.unparse(node.func.value)]
            if isinstance(node.func, ast.Attribute)
            else [getattr(node.func, "id", "")]
        )
    }
    assert called.isdisjoint(banned), called & banned
