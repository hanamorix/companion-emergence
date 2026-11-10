"""scripts/measure_feels_high_thresholds.py - it measures the engine's rule alone, and its host line.

Two groups. The host line: on Linux the script reads the ``model name`` field of /proc/cpuinfo itself; on any other
system, or when the field is missing or unreadable, it prints "unknown". It starts no process. Those tests point the
script's ``CPUINFO`` constant at a file in ``tmp_path`` and use sentinel model names, never a real CPU's.

The run: a small synthetic database (one counted row a day for 100 days, two registered feelings whose strength swings
between 2 and 9) goes through the whole script, so the script runs on this branch end to end: it must exit 0 (its own
invariants hold), print one column per floor and no column for the removed old aggregate reader, and fire a reader row.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect
import platform
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "measure_feels_high_thresholds.py"

_spec = importlib.util.spec_from_file_location("measure_feels_high_thresholds_under_test", SCRIPT)
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


def _synthetic_db(path: Path) -> None:
    """One counted row a day for 100 days; love and joy swing between 2 and 9 so the range part of the rule has a spread."""
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(path)
    try:
        start = datetime(2026, 1, 1, 12, tzinfo=UTC)
        for day in range(100):
            swing = 2.0 + 7.0 * ((day % 10) / 9.0)
            row = Memory.create_new(
                content=f"a made thing on day {day}",
                memory_type="making",
                domain="creative",
                emotions={"love": swing, "joy": 9.0 - swing + 2.0},
            )
            store.create(dataclasses.replace(row, created_at=start + timedelta(days=day)))
    finally:
        store.close()


def test_the_script_runs_end_to_end_on_the_engine_rule_alone(tmp_path):
    db = tmp_path / "memories.db"
    _synthetic_db(db)
    floors = (0.0, 0.1, 1.0)

    lines, code = mod.run(str(db), floors)
    text = "\n".join(lines)

    assert code == 0, text[-2000:]  # exit 0 means its own invariants held
    assert "INVARIANT FAILED" not in text
    assert "| reader row | floor 0 | floor 0.1 | floor 1 |" in text
    assert "| old " not in text and "old fixed" not in text
    assert "### R and the proposal" in text
    # The rule fires on this history: the love temperature row has floor-0 firings after day 30.
    after_30 = text.split("Firing counts, after day 30")[1].split("Firing counts, all ticks")[0]
    love_row = next(line for line in after_30.splitlines() if line.startswith("| temperature: love |"))
    first_cell = love_row.split("|")[2].strip()
    assert int(first_cell.split("/")[0]) > 0, love_row


def test_main_prints_the_report_and_exits_zero(tmp_path, capsys):
    db = tmp_path / "memories.db"
    _synthetic_db(db)

    code = mod.main([str(db), "--floors", "0,0.5"])

    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("# Feels-high threshold measurement")
    assert "host: " in out
    assert "| reader row | floor 0 | floor 0.5 |" in out  # one column per floor ...
    assert "| old" not in out  # ... and no column for the removed old reader


@pytest.mark.no_base_failure(
    reason="pins the one-line abort of the source step; that step moved into this script unchanged, so the old script gave the same line"
)
def test_a_path_that_is_not_a_file_aborts_with_the_one_line_message(tmp_path, capsys):
    code = mod.main([str(tmp_path / "absent.db")])

    err = capsys.readouterr().err
    assert code == 1
    assert err.startswith("measure_feels_high_thresholds: FAILED at source: not a regular file")
