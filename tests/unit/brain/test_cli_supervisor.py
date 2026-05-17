"""Argparse wiring tests for `nell supervisor` — verifies dispatch + flags
without invoking the real daemon. Behaviour of cmd_restart and cmd_tail_log
is covered by tests/unit/brain/bridge/test_daemon_extras.py."""

from __future__ import annotations

import pytest

from brain import cli

_STUB_HANDLERS = {
    "cmd_start": lambda args, **kw: 0,
    "cmd_run": lambda args: 0,
    "cmd_stop": lambda args: 0,
    "cmd_status": lambda args: 0,
    "cmd_tail": lambda args: 0,
    "cmd_restart": lambda args: 0,
    "cmd_tail_log": lambda args: 0,
}


@pytest.fixture(autouse=True)
def stub_daemon_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real daemon handlers with no-op stubs for argparse-only tests."""
    from brain.bridge import daemon

    for name, fn in _STUB_HANDLERS.items():
        monkeypatch.setattr(daemon, name, fn)


@pytest.mark.parametrize(
    "action",
    ["start", "run", "stop", "status", "restart", "tail-events", "tail-log"],
)
def test_supervisor_action_parses_with_required_persona(action: str) -> None:
    """Each action accepts --persona NAME and returns the stub's exit code (0)."""
    rc = cli.main(["supervisor", action, "--persona", "nell"])
    assert rc == 0


@pytest.mark.parametrize(
    "action",
    ["start", "run", "stop", "status", "restart", "tail-events", "tail-log"],
)
def test_supervisor_action_requires_persona(
    action: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing --persona is an argparse error (SystemExit code 2)."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["supervisor", action])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--persona" in err


def test_supervisor_start_accepts_idle_shutdown_and_client_origin() -> None:
    rc = cli.main(
        [
            "supervisor",
            "start",
            "--persona",
            "nell",
            "--idle-shutdown",
            "5",
            "--client-origin",
            "tauri",
        ]
    )
    assert rc == 0


def test_supervisor_run_defaults_to_launchd_and_no_idle_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreground service mode defaults to launchd origin and no idle shutdown."""
    from brain.bridge import daemon

    seen = {}

    def fake_run(args):
        seen["persona"] = args.persona
        seen["idle_shutdown"] = args.idle_shutdown
        seen["client_origin"] = args.client_origin
        return 0

    monkeypatch.setattr(daemon, "cmd_run", fake_run)

    rc = cli.main(["supervisor", "run", "--persona", "nell"])
    assert rc == 0
    assert seen == {
        "persona": "nell",
        "idle_shutdown": 0,
        "client_origin": "launchd",
    }


def test_supervisor_stop_accepts_timeout() -> None:
    rc = cli.main(["supervisor", "stop", "--persona", "nell", "--timeout", "10"])
    assert rc == 0


def test_supervisor_restart_accepts_start_and_stop_flags() -> None:
    rc = cli.main(
        [
            "supervisor",
            "restart",
            "--persona",
            "nell",
            "--idle-shutdown",
            "5",
            "--client-origin",
            "tests",
            "--timeout",
            "10",
        ]
    )
    assert rc == 0


def test_supervisor_tail_log_accepts_n_and_follow() -> None:
    rc = cli.main(["supervisor", "tail-log", "--persona", "nell", "-n", "20"])
    assert rc == 0
    rc = cli.main(["supervisor", "tail-log", "--persona", "nell", "-n", "0", "-f"])
    assert rc == 0


def test_supervisor_tail_log_rejects_negative_n(capsys: pytest.CaptureFixture[str]) -> None:
    """Argparse-level validation: -n must be a non-negative integer."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["supervisor", "tail-log", "--persona", "nell", "-n", "-3"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "-n" in err or "lines" in err


def test_supervisor_help_lists_all_seven_actions(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["supervisor", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for action in ("start", "run", "stop", "status", "restart", "tail-events", "tail-log"):
        assert action in out
