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
    """Call ``_cpu_model`` with CPUINFO pointed at ``cpuinfo`` and the script's platform seam (`mod._platform`) faked."""
    monkeypatch.setattr(mod, "CPUINFO", cpuinfo)
    monkeypatch.setattr(mod, "_platform", lambda: platform_name)
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
    monkeypatch.setattr(mod, "_platform", lambda: "linux")
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
    monkeypatch.setattr(mod, "_platform", lambda: "linux")
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
    # Every section of the report is still printed (a silent loss of one would otherwise pass).
    for section in (
        "By part, per floor",
        "Near-zero firings after day 30",
        "Present feelings whose highest short average",
        "body_strain mean:",
        "maker discharges:",
        "host: ",
    ):
        assert section in text, section
    # The rule fires on this history: the love temperature row has floor-0 firings after day 30.
    after_30 = text.split("Firing counts, after day 30")[1].split("Firing counts, all ticks")[0]
    love_row = next(line for line in after_30.splitlines() if line.startswith("| temperature: love |"))
    first_cell = love_row.split("|")[2].strip()
    assert int(first_cell.split("/")[0]) > 0, love_row


def test_a_row_whose_only_emotion_is_non_finite_counts_as_having_emotions_and_logs_no_drop(tmp_path, caplog):
    """The "with emotions" count asks whether a row names an emotion, not whether it has a finite value.

    The extra row is an "event" row, which the engine does not count, so the report's only read of it is that count.
    `create_new` drops non-finite values, so the row is built the way a loaded row is.
    """
    import logging

    from brain.memory.store import Memory, MemoryStore

    db = tmp_path / "memories.db"
    _synthetic_db(db)
    store = MemoryStore(db)
    try:
        store.create(
            Memory.from_stored_row(
                id="nan-valued-row",
                content="a thing the engine does not count",
                memory_type="event",
                domain="creative",
                created_at=datetime(2026, 2, 1, 6, tzinfo=UTC),
                emotions={"nan_ache": float("nan")},
            ),
            fold=False,
        )
    finally:
        store.close()

    with caplog.at_level(logging.DEBUG):
        lines, code = mod.run(str(db), (0.0, 0.1, 1.0))

    text = "\n".join(lines)
    assert code == 0, text[-2000:]
    assert "101 active rows, 101 with emotions" in text
    assert not [r for r in caplog.records if "dropping emotion" in r.getMessage()]


RESEARCH_ROW = "research gate: feeling half (opposing curve)"


def _research_row():
    return mod.reader_conditions([], 3)[RESEARCH_ROW]


def _high(*names: str):
    return set(names).__contains__


def test_research_row_an_opposing_feeling_never_opens_it():
    row = _research_row()
    assert not row({"grief": 9.0}, _high("grief"))
    assert not row({"shame": 9.0, "comfort_seeking": 9.0}, _high("shame", "comfort_seeking"))
    assert not row({"rest_need": 9.0}, _high("rest_need"))


def test_research_row_needs_both_high_and_strictly_above_the_threshold():
    from brain.engines.research import opposing_threshold

    row = _research_row()
    opp = 2.0
    t = opposing_threshold(opp)
    assert row({"grief": opp, "love": t + 0.01}, _high("love"))
    assert not row({"grief": opp, "love": t}, _high("love"))  # at the threshold
    assert not row({"grief": opp, "love": t - 0.01}, _high("love"))
    assert not row({"grief": opp, "love": t + 1.0}, _high())  # above T but not high


def test_research_row_with_no_opposing_feeling_todays_test_alone_decides():
    row = _research_row()
    assert row({"love": 0.5}, _high("love"))
    assert not row({"love": 9.0}, _high())


def test_research_row_opp_is_the_max_over_opposing_names_present_not_their_sum_or_the_high_ones():
    from brain.engines.research import opposing_threshold

    row = _research_row()
    # max(2, 2) = 2, T = 5.0; a sum (4) would give a higher T
    assert opposing_threshold(2.0) < 5.2 < opposing_threshold(4.0)
    assert row({"shame": 2.0, "grief": 2.0, "love": 5.2}, _high("love"))
    # unequal values: max 7 gives T 6.5 (closed at 5.6); the mean of the two (4) would give T 5.55 (open)
    assert opposing_threshold(7.0) > 5.6 > opposing_threshold(4.0)
    assert not row({"grief": 1.0, "shame": 7.0, "love": 5.6}, _high("love"))
    # an opposing feeling that is present but not high still sets opp
    assert opposing_threshold(7.0) > 6.0
    assert not row({"grief": 7.0, "love": 6.0}, _high("love"))


def test_research_row_follows_the_engine_constants_it_copies_nothing(monkeypatch):
    from brain import dev_constants

    row = _research_row()
    assert row({"grief": 0.0, "love": 5.0}, _high("love"))
    monkeypatch.setattr(dev_constants, "RESEARCH_GATE_OPPOSING_CURVE", ((0.0, 9.0), (10.0, 9.0)))
    assert not row({"grief": 0.0, "love": 5.0}, _high("love"))
    monkeypatch.undo()
    monkeypatch.setattr(dev_constants, "RESEARCH_GATE_OPPOSING_FEELINGS", frozenset({"love"}))
    assert not row({"love": 9.0, "joy": 4.0}, _high("love", "joy"))  # love is now opposing: opp 9, T 7.4, joy 4 below


def _grief_db(path: Path) -> None:
    """The same history with a constant grief of 5 (an opposing feeling) beside love swinging between 2 and 9."""
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
                emotions={"love": swing, "grief": 5.0},
            )
            store.create(dataclasses.replace(row, created_at=start + timedelta(days=day)))
    finally:
        store.close()


def test_the_research_row_end_to_end_is_stricter_than_any_feeling_high(tmp_path):
    db = tmp_path / "memories.db"
    _grief_db(db)

    lines, code = mod.run(str(db), (0.0, 1.0))
    text = "\n".join(lines)

    assert code == 0, text[-2000:]
    assert "research gate: any feeling high" not in text
    after_30 = text.split("Firing counts, after day 30")[1].split("Firing counts, all ticks")[0]
    research = next(line for line in after_30.splitlines() if line.startswith(f"| {RESEARCH_ROW} |"))
    fired, total = (int(x) for x in research.split("|")[3].strip().split("/"))  # the floor 1 column
    # grief, an opposing feeling, never opens the gate and raises the threshold for love: some ticks fire, not all.
    assert 0 < fired < total, research


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
