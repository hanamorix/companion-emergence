"""Unit tests for cmd_restart and cmd_tail_log — the two new daemon handlers
added with the `nell supervisor` rename. The four existing handlers
(cmd_start/stop/status/tail) are covered by tests/bridge/test_lifecycle.py
at the integration layer; we don't duplicate that here."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from brain.bridge import daemon


def _args(persona: str, **kw) -> argparse.Namespace:
    ns = argparse.Namespace(persona=persona, idle_shutdown=30, client_origin="cli", timeout=180.0)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


# ---------- cmd_restart ----------


def test_cmd_restart_calls_stop_then_start_when_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: stop returns 0, start returns 0, restart returns 0 with two-phase output."""
    calls: list[str] = []

    def fake_stop(args):
        calls.append("stop")
        return 0

    def fake_start(args):
        calls.append("start")
        return 0

    monkeypatch.setattr(daemon, "cmd_stop", fake_stop)
    monkeypatch.setattr(daemon, "cmd_start", fake_start)

    rc = daemon.cmd_restart(_args("nell"))
    assert rc == 0
    assert calls == ["stop", "start"]
    out = capsys.readouterr().out
    assert "stopping bridge" in out
    assert "starting bridge" in out


def test_cmd_restart_bails_when_stop_timed_out_on_wedged_bridge(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stop returns 1 (SIGTERM timeout / wedged bridge) — restart must NOT call start.

    cmd_stop returns 0 for clean-stop AND no-bridge-running, so the only way
    to see stop=1 is a wedge. Spawning a second bridge over a wedged first
    one is the exact footgun the supervisor exists to prevent.
    """
    calls: list[str] = []

    def fake_stop(args):
        calls.append("stop")
        return 1

    def fake_start(args):
        calls.append("start")
        return 0

    monkeypatch.setattr(daemon, "cmd_stop", fake_stop)
    monkeypatch.setattr(daemon, "cmd_start", fake_start)

    rc = daemon.cmd_restart(_args("nell"))
    assert rc == 1
    assert calls == ["stop"]
    err = capsys.readouterr().err
    assert "restart aborted" in err
    assert "stop failed" in err


def test_cmd_restart_bails_when_stop_returns_unexpected_high_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stop returns 2 (lock held) — restart must NOT call start; returns stop's code."""
    calls: list[str] = []

    def fake_stop(args):
        calls.append("stop")
        return 2

    def fake_start(args):
        calls.append("start")
        return 0

    monkeypatch.setattr(daemon, "cmd_stop", fake_stop)
    monkeypatch.setattr(daemon, "cmd_start", fake_start)

    rc = daemon.cmd_restart(_args("nell"))
    assert rc == 2
    assert calls == ["stop"]
    err = capsys.readouterr().err
    assert "restart aborted" in err
    assert "stop failed" in err


def test_cmd_restart_propagates_start_failure_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop succeeds but start returns 1 — restart returns 1, not coerced."""
    monkeypatch.setattr(daemon, "cmd_stop", lambda a: 0)
    monkeypatch.setattr(daemon, "cmd_start", lambda a: 1)
    assert daemon.cmd_restart(_args("nell")) == 1


def test_cmd_restart_propagates_start_already_running_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop succeeds but start returns 2 (race: another process spawned) — restart returns 2."""
    monkeypatch.setattr(daemon, "cmd_stop", lambda a: 0)
    monkeypatch.setattr(daemon, "cmd_start", lambda a: 2)
    assert daemon.cmd_restart(_args("nell")) == 2


# ---------- cmd_tail_log ----------


def _make_persona(tmp_path: Path, name: str = "nell") -> Path:
    home = tmp_path / "home"
    persona_dir = home / "personas" / name
    persona_dir.mkdir(parents=True)
    return persona_dir


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, persona: str = "nell") -> Path:
    """Wire NELLBRAIN_HOME so get_persona_dir AND get_log_dir resolve under tmp_path/home.

    get_log_dir() honors NELLBRAIN_HOME (returning <HOME>/logs) since the
    paths.py override fix; setting the env var is sufficient.
    """
    home = tmp_path / "home"
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _make_persona(tmp_path, persona)
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))
    return log_dir


def _tail_args(persona: str, **kw) -> argparse.Namespace:
    ns = argparse.Namespace(persona=persona, lines=50, follow=False)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_cmd_tail_log_prints_last_n_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log_dir = _patch_paths(monkeypatch, tmp_path)
    log_path = log_dir / "bridge-nell.log"
    log_path.write_text("\n".join(f"line {i}" for i in range(1, 101)) + "\n")

    rc = daemon.cmd_tail_log(_tail_args("nell", lines=3))
    assert rc == 0
    out = capsys.readouterr().out
    assert "line 98" in out
    assert "line 99" in out
    assert "line 100" in out
    assert "line 97" not in out


def test_cmd_tail_log_default_50_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log_dir = _patch_paths(monkeypatch, tmp_path)
    log_path = log_dir / "bridge-nell.log"
    log_path.write_text("\n".join(f"line {i}" for i in range(1, 101)) + "\n")

    rc = daemon.cmd_tail_log(_tail_args("nell"))  # lines=50 default
    assert rc == 0
    out = capsys.readouterr().out
    assert "line 51" in out
    assert "line 50" not in out


def test_cmd_tail_log_n_zero_prints_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log_dir = _patch_paths(monkeypatch, tmp_path)
    log_path = log_dir / "bridge-nell.log"
    log_path.write_text("alpha\nbeta\n")

    rc = daemon.cmd_tail_log(_tail_args("nell", lines=0))
    assert rc == 0
    out = capsys.readouterr().out
    assert out == ""


def test_cmd_tail_log_missing_file_returns_1_with_helpful_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_paths(monkeypatch, tmp_path)  # log dir exists; log file does not

    rc = daemon.cmd_tail_log(_tail_args("nell"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "bridge log not found" in err
    assert "supervisor ever started" in err


def test_cmd_tail_log_persona_not_found_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))

    rc = daemon.cmd_tail_log(_tail_args("ghost"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "persona directory not found" in err


def test_cmd_tail_log_follow_mode_emits_new_lines_then_exits_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Follow mode polls the log; KeyboardInterrupt exits cleanly with code 0.

    Timing: the writer thread sleeps 150ms, appends two lines, sleeps another
    150ms, then sets the stop_event. The daemon polls every 200ms. If this
    test ever flakes on a heavily loaded Windows CI runner where the test
    thread is starved between the writer flush and stop_event.set(), widen
    the writer's first sleep (200-300ms) rather than lengthening t.join's
    timeout — the issue would be the chunk-read race, not the join.
    """
    import threading
    import time

    log_dir = _patch_paths(monkeypatch, tmp_path)
    log_path = log_dir / "bridge-nell.log"
    log_path.write_text("seed\n")

    stop_event = threading.Event()
    monkeypatch.setattr(daemon, "_follow_should_stop", stop_event)

    # writer thread appends two lines after a short delay, then signals interrupt
    def writer():
        time.sleep(0.15)
        with log_path.open("a") as f:
            f.write("new1\n")
            f.write("new2\n")
            f.flush()
        time.sleep(0.15)
        stop_event.set()

    t = threading.Thread(target=writer, daemon=True)
    t.start()

    rc = daemon.cmd_tail_log(_tail_args("nell", lines=1, follow=True))
    t.join(timeout=2.0)
    assert rc == 0
    out = capsys.readouterr().out
    assert "seed" in out
    assert "new1" in out
    assert "new2" in out
