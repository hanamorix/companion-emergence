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


def test_cmd_restart_proceeds_when_nothing_was_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stop returns 1 with no-bridge-running semantics — restart should still try start."""
    calls: list[str] = []

    def fake_stop(args):
        calls.append("stop")
        print("no bridge running")
        return 1

    def fake_start(args):
        calls.append("start")
        return 0

    monkeypatch.setattr(daemon, "cmd_stop", fake_stop)
    monkeypatch.setattr(daemon, "cmd_start", fake_start)

    rc = daemon.cmd_restart(_args("nell"))
    assert rc == 0
    assert calls == ["stop", "start"]


def test_cmd_restart_bails_when_stop_fails_with_lock_held(
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
