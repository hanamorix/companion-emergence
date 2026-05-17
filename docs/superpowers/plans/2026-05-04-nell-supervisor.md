# nell supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `nell supervisor` stub with a canonical bridge lifecycle command (start/stop/status/restart/tail-events/tail-log), keep `nell bridge` as a deprecating alias until v0.1.

**Architecture:** Two new daemon handlers (`cmd_restart`, `cmd_tail_log`) wrap existing primitives in `brain/bridge/daemon.py`. A new argparse subparser tree in `brain/cli.py` wires all six actions. The existing `nell bridge` subparser tree is rerouted through a one-line `_deprecated_bridge` wrapper that prints a stderr warning before forwarding to the same handlers.

**Tech Stack:** Python 3, argparse, pytest, monkeypatch/capsys/tmp_path fixtures. Pure-Python tail (no shell `tail` — Windows CI lacks it).

---

## Spec reference

`docs/superpowers/specs/2026-05-04-nell-supervisor-design.md` (committed as `c72d4fd`).

## File structure

| Action | Path | Purpose |
|---|---|---|
| Modify | `brain/bridge/daemon.py` | Add `cmd_restart` (~30 lines) + `cmd_tail_log` (~40 lines) next to existing handlers |
| Modify | `brain/cli.py` | Add `supervisor` subparser tree; add `_deprecated_bridge` wrapper; rewrite `bridge` subparser to route through wrapper; remove `"supervisor"` from `_STUB_NAMES` |
| Create | `tests/unit/brain/bridge/test_daemon_extras.py` | Unit tests for `cmd_restart` + `cmd_tail_log` |
| Create | `tests/unit/brain/test_cli_supervisor.py` | Argparse wiring tests for the new subparser tree |
| Modify | `tests/unit/brain/test_cli.py` | Update `STUB_COMMANDS` parametrize list (remove `"supervisor"`); replace the `supervisor --help` stub-test with a deprecation-alias test |
| Modify | `CHANGELOG.md` | Added entry + new Deprecated subsection |
| Modify | `docs/roadmap.md` | Strike supervisor in §2; v0.1 removal in §3; Done-recently prepend |

**Why a new `test_daemon_extras.py` file** — there's no `test_daemon.py` today. `cmd_start/stop/status/tail` are exercised only via FastAPI integration in `tests/bridge/test_lifecycle.py`. Adding unit-level coverage for the two new handlers is the right move; naming it `_extras` keeps it scoped (we're not extending coverage to all four existing handlers — YAGNI).

## TDD ordering

Phase 1 (Tasks 1–2): handlers, tests-first.
Phase 2 (Task 3): argparse wiring for `nell supervisor`.
Phase 3 (Task 4): deprecation alias for `nell bridge`.
Phase 4 (Tasks 5–6): code comments, docs.

Each task ends with a commit. Run the full test suite (`pytest`) at the end of each task to confirm cross-platform suite stays green — the CI matrix (macOS/Linux/Windows) gates the merge.

---

### Task 1: `cmd_restart` handler with TDD

**Files:**
- Create: `tests/unit/brain/bridge/test_daemon_extras.py`
- Modify: `brain/bridge/daemon.py` (add `cmd_restart` after existing `cmd_tail`)

- [ ] **Step 1: Write the failing tests for `cmd_restart`**

Create `tests/unit/brain/bridge/test_daemon_extras.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/brain/bridge/test_daemon_extras.py -v`
Expected: 5 failures with `AttributeError: module 'brain.bridge.daemon' has no attribute 'cmd_restart'`

- [ ] **Step 3: Implement `cmd_restart` in `brain/bridge/daemon.py`**

Append after the existing `cmd_tail` function (around line 270):

```python
def cmd_restart(args) -> int:
    """Stop the bridge, then start it again. Two-phase, gated on stop success.

    Stop is allowed to return 0 (stopped cleanly) or 1 (no bridge was running) —
    both proceed to start. Any other non-zero stop code (lock held, timeout)
    bails before start so we don't spawn a second bridge while the first is wedged.
    Restart's exit code is whatever cmd_start returned (0/1/2) when stop succeeded,
    or stop's exit code when stop failed.
    """
    print("stopping bridge…")
    stop_rc = cmd_stop(args)
    if stop_rc not in (0, 1):
        print(f"restart aborted: stop failed (exit {stop_rc})", file=sys.stderr)
        return stop_rc
    print("starting bridge…")
    return cmd_start(args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/brain/bridge/test_daemon_extras.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the full suite**

Run: `pytest`
Expected: 1160 passed (1155 baseline + 5 new), no regressions.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/brain/bridge/test_daemon_extras.py brain/bridge/daemon.py
git commit -m "$(cat <<'EOF'
feat(bridge): add cmd_restart handler — sequential stop-then-start

Two-phase restart gated on stop success. Stop codes 0 and 1 (no bridge
running) proceed to start; anything else bails with stop's exit code so
we don't spawn a second bridge while the first is wedged. Start's exit
code (0/1/2) propagates unchanged.

Part of the nell supervisor rename. See
docs/superpowers/specs/2026-05-04-nell-supervisor-design.md.
EOF
)"
```

---

### Task 2: `cmd_tail_log` handler with TDD

**Files:**
- Modify: `tests/unit/brain/bridge/test_daemon_extras.py` (extend)
- Modify: `brain/bridge/daemon.py` (add `cmd_tail_log` after `cmd_restart`)

- [ ] **Step 1: Append failing tests for `cmd_tail_log`**

Append to `tests/unit/brain/bridge/test_daemon_extras.py`:

```python
# ---------- cmd_tail_log ----------


def _make_persona(tmp_path: Path, name: str = "nell") -> Path:
    home = tmp_path / "home"
    persona_dir = home / "personas" / name
    persona_dir.mkdir(parents=True)
    return persona_dir


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, persona: str = "nell") -> Path:
    """Wire NELLBRAIN_HOME so get_persona_dir + get_log_dir resolve under tmp_path."""
    home = tmp_path / "home"
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    persona_dir = _make_persona(tmp_path, persona)
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
    """Follow mode polls the log; KeyboardInterrupt exits cleanly with code 0."""
    import threading
    import time

    log_dir = _patch_paths(monkeypatch, tmp_path)
    log_path = log_dir / "bridge-nell.log"
    log_path.write_text("seed\n")

    # writer thread appends two lines after a short delay, then signals interrupt
    def writer():
        time.sleep(0.15)
        with log_path.open("a") as f:
            f.write("new1\n")
            f.write("new2\n")
            f.flush()
        time.sleep(0.15)
        # raise KeyboardInterrupt in main via _follow_should_stop hook
        daemon._follow_should_stop.set()  # type: ignore[attr-defined]

    daemon._follow_should_stop = threading.Event()  # type: ignore[attr-defined]
    t = threading.Thread(target=writer, daemon=True)
    t.start()

    rc = daemon.cmd_tail_log(_tail_args("nell", lines=1, follow=True))
    t.join(timeout=2.0)
    assert rc == 0
    out = capsys.readouterr().out
    assert "seed" in out
    assert "new1" in out
    assert "new2" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/brain/bridge/test_daemon_extras.py -v -k tail_log`
Expected: 6 failures with `AttributeError: module 'brain.bridge.daemon' has no attribute 'cmd_tail_log'`

- [ ] **Step 3: Implement `cmd_tail_log` in `brain/bridge/daemon.py`**

Add these imports near the top of `brain/bridge/daemon.py` (only `threading` and `time` if not already imported — `time` already is):

```python
import threading
```

Append after `cmd_restart`:

```python
# Hook used by tests + ctrl-c handler in follow mode. Tests can set this
# Event to exit follow mode cleanly without raising KeyboardInterrupt.
_follow_should_stop: threading.Event | None = None


def cmd_tail_log(args) -> int:
    """Print the last N lines of the bridge log; -f to follow.

    Cross-platform: pure Python loop, no shell `tail` (Windows CI lacks it).
    Follow mode polls every 200ms. KeyboardInterrupt is treated as a clean
    exit (returns 0). Tests can interrupt the loop by setting the module-level
    `_follow_should_stop` Event before the call.
    """
    from brain.paths import get_log_dir, get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        print(f"persona directory not found: {persona_dir}", file=sys.stderr)
        return 1

    log_path = get_log_dir() / f"bridge-{persona_dir.name}.log"
    if not log_path.exists():
        print(
            f"bridge log not found at {log_path} — has the supervisor ever started?",
            file=sys.stderr,
        )
        return 1

    n = max(0, int(args.lines))
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            tail = lines[-n:] if n > 0 else []
            for line in tail:
                print(line, end="")
            if not getattr(args, "follow", False):
                return 0
            # Follow mode: seek to end, poll for new content
            f.seek(0, 2)  # SEEK_END
            stop = _follow_should_stop or threading.Event()
            try:
                while not stop.is_set():
                    chunk = f.read()
                    if chunk:
                        print(chunk, end="")
                    else:
                        time.sleep(0.2)
            except KeyboardInterrupt:
                return 0
            return 0
    except OSError as e:
        print(f"error reading {log_path}: {e}", file=sys.stderr)
        return 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/brain/bridge/test_daemon_extras.py -v`
Expected: 11 passed (5 from Task 1 + 6 new).

- [ ] **Step 5: Run the full suite — verify cross-platform readiness**

Run: `pytest`
Expected: 1166 passed (1160 + 6 new), no regressions. The follow-mode test uses `threading` + `time.sleep` — no platform-specific calls.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/brain/bridge/test_daemon_extras.py brain/bridge/daemon.py
git commit -m "$(cat <<'EOF'
feat(bridge): add cmd_tail_log handler — pure-Python cross-platform tail

Reads the persona's bridge log file (-n LINES, -f follow). Pure Python so
Windows CI doesn't need `tail`. Follow mode polls every 200ms; clean
exit on KeyboardInterrupt or via the module-level _follow_should_stop
Event used by tests.

Part of the nell supervisor rename.
EOF
)"
```

---

### Task 3: Wire `nell supervisor` argparse subparser tree

**Files:**
- Create: `tests/unit/brain/test_cli_supervisor.py`
- Modify: `brain/cli.py`
- Modify: `tests/unit/brain/test_cli.py` (drop `"supervisor"` from `STUB_COMMANDS`, repoint the `--help` test)

- [ ] **Step 1: Write the failing tests for the `nell supervisor` argparse surface**

Create `tests/unit/brain/test_cli_supervisor.py`:

```python
"""Argparse wiring tests for `nell supervisor` — verifies dispatch + flags
without invoking the real daemon. Behaviour of cmd_restart and cmd_tail_log
is covered by tests/unit/brain/bridge/test_daemon_extras.py."""
from __future__ import annotations

import pytest

from brain import cli


_STUB_HANDLERS = {
    "cmd_start": lambda args: 0,
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


@pytest.mark.parametrize("action", ["start", "stop", "status", "restart", "tail-events", "tail-log"])
def test_supervisor_action_parses_with_required_persona(action: str) -> None:
    """Each action accepts --persona NAME and returns the stub's exit code (0)."""
    rc = cli.main(["supervisor", action, "--persona", "nell"])
    assert rc == 0


@pytest.mark.parametrize("action", ["start", "stop", "status", "restart", "tail-events", "tail-log"])
def test_supervisor_action_requires_persona(action: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Missing --persona is an argparse error (SystemExit code 2)."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["supervisor", action])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--persona" in err


def test_supervisor_start_accepts_idle_shutdown_and_client_origin() -> None:
    rc = cli.main(["supervisor", "start", "--persona", "nell", "--idle-shutdown", "5", "--client-origin", "tauri"])
    assert rc == 0


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


def test_supervisor_help_lists_all_six_actions(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["supervisor", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for action in ("start", "stop", "status", "restart", "tail-events", "tail-log"):
        assert action in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/brain/test_cli_supervisor.py -v`
Expected: most fail. Some may fail with "supervisor: not implemented yet" (the current stub still triggers); the action-specific tests fail with argparse "invalid choice" because no `supervisor` subparser exists yet.

- [ ] **Step 3: Update `STUB_COMMANDS` in `tests/unit/brain/test_cli.py`**

The existing test file parametrizes a stub-not-implemented test on `STUB_COMMANDS = ["supervisor", "rest", "works"]`. Once we unstub `supervisor`, that test must no longer include it. Edit `tests/unit/brain/test_cli.py` lines 35–39:

```python
STUB_COMMANDS = [
    "rest",
    "works",
]
```

Replace the existing `test_stub_subcommand_help_works` test (which hardcodes `"supervisor"`) with one that uses the first remaining stub:

```python
def test_stub_subcommand_help_works(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each stub subcommand supports --help without crashing."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main([STUB_COMMANDS[0], "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert STUB_COMMANDS[0] in captured.out.lower()
```

- [ ] **Step 4: Add the `supervisor` subparser tree in `brain/cli.py`**

Find the `# nell bridge — SP-7 daemon control` block (around line 1385). Insert the new `supervisor` subparser block immediately before it:

```python
# nell supervisor — canonical bridge lifecycle (replaces nell bridge in v0.1).
from brain.bridge.daemon import (
    cmd_restart,
    cmd_start,
    cmd_status,
    cmd_stop,
    cmd_tail,
    cmd_tail_log,
)

s_sub = subparsers.add_parser(
    "supervisor",
    help="Manage the per-persona bridge daemon — canonical lifecycle command.",
)
s_actions = s_sub.add_subparsers(dest="action", required=True)


def _add_persona_arg(p):
    p.add_argument("--persona", required=True)


def _nonneg_int(v: str) -> int:
    iv = int(v)
    if iv < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return iv


s_start = s_actions.add_parser("start", help="Start the bridge daemon.")
_add_persona_arg(s_start)
s_start.add_argument(
    "--idle-shutdown", type=float, default=30,
    help="Idle-shutdown threshold in minutes (0 = never).",
)
s_start.add_argument("--client-origin", default="cli", choices=["cli", "tauri", "tests"])
s_start.set_defaults(func=cmd_start)

s_stop = s_actions.add_parser("stop", help="Stop the bridge daemon.")
_add_persona_arg(s_stop)
s_stop.add_argument("--timeout", type=float, default=180.0)
s_stop.set_defaults(func=cmd_stop)

s_status = s_actions.add_parser("status", help="Show bridge daemon status.")
_add_persona_arg(s_status)
s_status.set_defaults(func=cmd_status)

s_restart = s_actions.add_parser(
    "restart", help="Stop and start — gated on stop success.",
)
_add_persona_arg(s_restart)
s_restart.add_argument("--idle-shutdown", type=float, default=30)
s_restart.add_argument("--client-origin", default="cli", choices=["cli", "tauri", "tests"])
s_restart.add_argument("--timeout", type=float, default=180.0)
s_restart.set_defaults(func=cmd_restart)

s_tail_events = s_actions.add_parser("tail-events", help="Tail /events as JSON lines.")
_add_persona_arg(s_tail_events)
s_tail_events.set_defaults(func=cmd_tail)

s_tail_log = s_actions.add_parser(
    "tail-log", help="Tail the bridge log file (cross-platform).",
)
_add_persona_arg(s_tail_log)
s_tail_log.add_argument(
    "-n", "--lines", dest="lines", type=_nonneg_int, default=50,
    help="Print the last N lines (default 50; 0 = none, useful with -f).",
)
s_tail_log.add_argument(
    "-f", "--follow", dest="follow", action="store_true", default=False,
    help="Follow the log: emit new lines as written. Ctrl-c to exit.",
)
s_tail_log.set_defaults(func=cmd_tail_log)
```

- [ ] **Step 5: Remove `"supervisor"` from the stub list in `brain/cli.py`**

Find lines 49–55. Edit:

```python
# Subcommands the framework plans to ship. Each is a stub in Week 1;
# `nell supervisor` was unstubbed on 2026-05-04 (see docs/superpowers/specs/2026-05-04-nell-supervisor-design.md).
_STUB_NAMES = (
    "rest",
    "works",
)
```

(Verify the constant is named `_STUB_NAMES` — adjust the name if the file uses a different one. The current code references it via `_make_stub` factory at line 58.)

- [ ] **Step 6: Run the new + existing CLI tests**

Run: `pytest tests/unit/brain/test_cli_supervisor.py tests/unit/brain/test_cli.py -v`
Expected: all `test_supervisor_*` pass; `STUB_COMMANDS` parametrize tests pass for `rest` and `works`; the rewritten `test_stub_subcommand_help_works` passes.

- [ ] **Step 7: Run the full suite**

Run: `pytest`
Expected: ~1175 passed (1166 + ~10 new wiring tests). Zero regressions.

- [ ] **Step 8: Commit**

```bash
git add brain/cli.py tests/unit/brain/test_cli_supervisor.py tests/unit/brain/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): implement nell supervisor — canonical bridge lifecycle

Six actions: start, stop, status, restart, tail-events, tail-log.
Same args + exit codes as `nell bridge`, plus restart and tail-log.
Removes 'supervisor' from the stub list.

Closes the April-30 audit P2 finding about operationally important
names being stubbed as success. See
docs/superpowers/specs/2026-05-04-nell-supervisor-design.md.
EOF
)"
```

---

### Task 4: Deprecation alias for `nell bridge`

**Files:**
- Modify: `brain/cli.py` (add `_deprecated_bridge` wrapper; rewrite `b_*.set_defaults` calls)
- Modify: `tests/unit/brain/test_cli.py` (add deprecation-alias tests)

- [ ] **Step 1: Write the failing tests for the deprecation alias**

Append to `tests/unit/brain/test_cli.py`:

```python
# ---------- nell bridge deprecation alias (removed in v0.1) ----------


@pytest.fixture
def stub_daemon(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Replace daemon handlers with stubs that record calls; return the call log."""
    from brain.bridge import daemon

    calls: dict[str, list] = {"start": [], "stop": [], "status": [], "tail": []}

    monkeypatch.setattr(daemon, "cmd_start", lambda a: (calls["start"].append(a), 0)[1])
    monkeypatch.setattr(daemon, "cmd_stop", lambda a: (calls["stop"].append(a), 0)[1])
    monkeypatch.setattr(daemon, "cmd_status", lambda a: (calls["status"].append(a), 0)[1])
    monkeypatch.setattr(daemon, "cmd_tail", lambda a: (calls["tail"].append(a), 0)[1])
    return calls


@pytest.mark.parametrize(
    "action,handler_key",
    [("start", "start"), ("stop", "stop"), ("status", "status"), ("tail-events", "tail")],
)
def test_bridge_alias_still_dispatches_to_real_handler(
    action: str, handler_key: str, stub_daemon: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    """`nell bridge X` keeps working exactly like before — the real daemon handler runs."""
    rc = cli.main(["bridge", action, "--persona", "nell"])
    assert rc == 0
    assert len(stub_daemon[handler_key]) == 1


def test_bridge_alias_prints_deprecation_warning_to_stderr(
    stub_daemon: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["bridge", "status", "--persona", "nell"])
    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower()
    assert "nell supervisor" in captured.err
    assert "v0.1" in captured.err


def test_bridge_alias_does_not_print_warning_to_stdout(
    stub_daemon: dict, capsys: pytest.CaptureFixture[str]
) -> None:
    """Warning must NOT pollute stdout — scripts piping output through jq/grep depend on this."""
    cli.main(["bridge", "status", "--persona", "nell"])
    captured = capsys.readouterr()
    assert "deprecated" not in captured.out.lower()


def test_bridge_alias_preserves_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the real handler returns 2, the alias must also return 2 (not coerce)."""
    from brain.bridge import daemon

    monkeypatch.setattr(daemon, "cmd_start", lambda a: 2)
    rc = cli.main(["bridge", "start", "--persona", "nell"])
    assert rc == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/brain/test_cli.py -v -k bridge_alias`
Expected: tests fail because no warning is emitted yet (the alias dispatches but silently).

- [ ] **Step 3: Add `_deprecated_bridge` wrapper in `brain/cli.py`**

Add this helper near the top of `brain/cli.py` (after the imports, before the existing handler functions):

```python
# Deprecation alias for `nell bridge` — to be removed in v0.1.
# Note: `nell chat` auto-spawn (see `_chat_handler`) imports
# brain.bridge.daemon directly and does NOT use this CLI surface, so
# removing the alias does not break chat. See docs/roadmap.md §3.
def _deprecated_bridge(real_handler):
    def wrapped(args):
        print(
            "warning: 'nell bridge' is deprecated; use 'nell supervisor' instead. "
            "This alias will be removed in v0.1.",
            file=sys.stderr,
        )
        return real_handler(args)
    wrapped.__name__ = f"deprecated_{real_handler.__name__}"
    return wrapped
```

Then find the existing bridge subparser block (around line 1385) and rewrite the four `set_defaults` calls to route through `_deprecated_bridge`:

```python
b_start.set_defaults(func=_deprecated_bridge(cmd_start))
# (same args block as before — only the func= line changes)

b_stop.set_defaults(func=_deprecated_bridge(cmd_stop))
b_status.set_defaults(func=_deprecated_bridge(cmd_status))
b_tail.set_defaults(func=_deprecated_bridge(cmd_tail))
```

Leave the help text `"Manage the per-persona bridge daemon (SP-7)."` unchanged. Discoverable warning beats hidden command.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/brain/test_cli.py -v -k bridge_alias`
Expected: 7 passed (4 parametrized dispatch tests + warning-to-stderr + warning-not-on-stdout + exit-code-preserved).

- [ ] **Step 5: Run the full suite**

Run: `pytest`
Expected: ~1182 passed (1175 + 7 new). Zero regressions.

- [ ] **Step 6: Commit**

```bash
git add brain/cli.py tests/unit/brain/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): deprecate `nell bridge` — alias forwards to `nell supervisor`

`nell bridge X` still works (start/stop/status/tail-events) but prints
a one-line warning to stderr before dispatching. Exit codes unchanged.
Will be removed in v0.1.

The `nell chat` auto-spawn path is unaffected — it uses
brain.bridge.daemon internals directly, not the CLI.
EOF
)"
```

---

### Task 5: Code-comment cross-reference at `_chat_handler` auto-spawn

**Files:**
- Modify: `brain/cli.py` (one comment block near the chat auto-spawn inside `_chat_handler`)

This task is the second half of the "non-impact note" — Task 4 added the comment above `_deprecated_bridge`; this one adds the breadcrumb at the chat side so the cross-reference is bidirectional.

- [ ] **Step 1: Find the chat auto-spawn block**

Run: `grep -n "Dispatch \`nell chat\`" brain/cli.py`
Expected: a single match — the `_chat_handler` function. The auto-spawn import + call lives just below the docstring.

- [ ] **Step 2: Add the breadcrumb comment**

Above the line `from brain.bridge import daemon, state_file` inside `_chat_handler`:

```python
# Note: this auto-spawn imports brain.bridge.daemon directly. It does
# NOT shell out to `nell bridge`/`nell supervisor`, so the deprecated
# bridge alias (removed in v0.1) does not affect this path. See
# _deprecated_bridge above and docs/roadmap.md §3.
from brain.bridge import daemon, state_file
```

- [ ] **Step 3: Run the full suite (sanity)**

Run: `pytest`
Expected: same as Task 4 — comments don't change behaviour.

- [ ] **Step 4: Commit**

```bash
git add brain/cli.py
git commit -m "$(cat <<'EOF'
docs(cli): cross-reference comment at chat auto-spawn → bridge alias

Breadcrumb so whoever removes the `nell bridge` alias in v0.1 sees that
chat auto-spawn uses brain.bridge.daemon internals directly and won't
break. Pairs with the comment above _deprecated_bridge.
EOF
)"
```

---

### Task 6: CHANGELOG + roadmap deltas

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/roadmap.md`

- [ ] **Step 1: Update `CHANGELOG.md`**

Open `CHANGELOG.md`. Under `## 0.0.1 - Unreleased` → `### Added`, append:

```
- `nell supervisor` lifecycle command — canonical operator surface for the per-persona bridge daemon. Actions: `start`, `stop`, `status`, `restart`, `tail-events`, `tail-log`. Wraps the existing bridge daemon implementation; same args, same exit codes, plus sequential `restart` (stop-then-start, gated on stop success) and cross-platform `tail-log`.
```

After the existing `### Fixed` section (or wherever the section ordering naturally lands — match the current file's ordering), insert a new `### Deprecated` subsection:

```markdown
### Deprecated

- `nell bridge` — use `nell supervisor` instead. The alias still works and forwards to the new command, but prints a deprecation warning to stderr. Will be removed in v0.1.
```

- [ ] **Step 2: Update `docs/roadmap.md` §2 — Replace stubs**

Find the suggested-order list (around line 51–55):

```markdown
1. `nell supervisor` — expose bridge/supervisor lifecycle in one operator-facing place.
2. `nell rest` — clarify whether this is sleep/rest cadence, bridge rest, or old-plan residue before implementing.
3. `nell works` — define the user story before building; the name is currently ambiguous.
```

Replace with:

```markdown
1. ~~`nell supervisor` — expose bridge/supervisor lifecycle in one operator-facing place.~~ *(shipped 2026-05-04 — see `docs/superpowers/specs/2026-05-04-nell-supervisor-design.md`)*
2. `nell rest` — clarify whether this is sleep/rest cadence, bridge rest, or old-plan residue before implementing.
3. `nell works` — define the user story before building; the name is currently ambiguous.
```

Also update the "Current intentional stubs" list (lines ~38–42). Remove `- nell supervisor`:

```markdown
Current intentional stubs:

- `nell rest`
- `nell works`
```

- [ ] **Step 3: Update `docs/roadmap.md` §3 — Public release blockers**

Under "Before public/tagged release" (around line 65), add a new bullet:

```markdown
- Remove the deprecated `nell bridge` alias. Removing it does not affect `nell chat` auto-spawn — chat uses `brain.bridge.daemon` internals directly (inside `_chat_handler`), not the CLI surface.
```

- [ ] **Step 4: Update `docs/roadmap.md` "Done recently"**

Prepend (so newest is first):

```markdown
- Implemented `nell supervisor` as the canonical bridge lifecycle command (start/stop/status/restart/tail-events/tail-log), with `nell bridge` kept as a deprecating alias until v0.1.
```

- [ ] **Step 5: Sanity-check the docs**

Run: `pytest`
Expected: same passing count as Task 5.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md docs/roadmap.md
git commit -m "$(cat <<'EOF'
docs(roadmap,changelog): record nell supervisor + bridge deprecation

CHANGELOG: new Added entry for `nell supervisor`, new Deprecated
subsection for `nell bridge`.

Roadmap §2: strike supervisor stub with shipped date; remove from
intentional-stubs list. Roadmap §3: add v0.1 blocker to remove the
bridge alias, with the `nell chat` non-impact note.
EOF
)"
```

---

## Self-review pass

**Spec coverage:**

| Spec section | Plan task |
|---|---|
| CLI surface (six actions) | Task 3 |
| `restart` semantics | Task 1 |
| `tail-log` semantics | Task 2 |
| `tail-events` identical | Task 3 (wires to existing `cmd_tail`) |
| Defaults preserved | Task 3 (argparse args mirror existing) |
| Architecture / file table | Tasks 1–3 (touched files match) |
| Deprecation alias | Task 4 |
| Error handling exit codes | Tasks 1, 2, 4 (asserted in tests) |
| Persona-not-found uniform | Task 2 (`cmd_tail_log`); Task 1 inherits via stop/start mocks; existing `cmd_start` already covers it |
| Testing approach | Tasks 1–4 (every TDD pair) |
| Documentation deltas | Tasks 5, 6 |
| Non-impact notes | Task 4 (comment above wrapper) + Task 5 (comment at chat handler) + Task 6 (roadmap §3 line) |
| Out-of-scope items | None — explicitly not implemented |

**Placeholder scan:** none. Every step has either exact code or an exact command + expected output.

**Type consistency:**

- `cmd_restart`, `cmd_tail_log` — defined in Task 1/Task 2, imported by name in Task 3, no rename anywhere.
- `_deprecated_bridge` — defined in Task 4, used in Task 4 and referenced by name in Task 5's comment.
- `_follow_should_stop` — module-level Event in Task 2; tests in Task 2 set it directly via `daemon._follow_should_stop`. Naming consistent.
- `STUB_COMMANDS` (test) and `_STUB_NAMES` (cli) — both exist, both correctly updated to drop `"supervisor"`. Verified during Task 3 step 5 (the file may use `_STUB_NAMES` or another name; the step instructs verifying before editing).
- Argparse dest names: `lines` for `-n`, `follow` for `-f`, `persona`, `idle_shutdown`, `client_origin`, `timeout` — all match the `_args`/`_tail_args` test factories in Task 1/2.

**Cross-platform note:** Task 2's `cmd_tail_log` uses pure Python (`pathlib`, `time.sleep`, no shell `tail`, no inotify/fsevents). The follow-mode test uses `threading` + a daemon thread to write new lines and a module-level Event to signal exit — no platform-specific calls. CI matrix (macOS/Linux/Windows in `.github/workflows/`) should remain green.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-04-nell-supervisor.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
