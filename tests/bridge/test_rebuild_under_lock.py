"""Engine 6, S107 step (c), HS5: the rebuild under the start lock (`daemon._rebuild_under_lock`), its progress lines, and the lifted
`runner._runtime_log_attached` helper.

The step ends every other running command (`kill_command`), then runs `nell supervisor rebuild --persona <name>` as a child
process, waits for it with no timeout, and shows progress lines when stderr is a terminal. A failed rebuild (a non-zero exit, an
OSError from the spawn) is one logged, shown failure and the step ends normally. Every test here carries `real_rebuild_step`, so the
conftest replaces the daemon's `Popen` with a recorder: no rebuild child is spawned for real (the one real child is in
test_rebuild_real_child.py). Synthetic personas only, under the bridge conftest's tmp home.
"""

from __future__ import annotations

import io
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from brain import process_identity
from brain.bridge import daemon, kill, runner, stop_watch
from brain.bridge.kill import KillResult
from brain.emotion import engine_files
from brain.paths import get_log_dir, get_persona_dir
from brain.utils import file_ops
from tests.conftest import FakeRebuildChild, RebuildPopenRecorder

STARTED = "Updating your memories. This can take a few minutes."
RESCALING = "Updating the scale of your memories."
PREPARING = "Getting ready to update your memories."


def _could_not_end(pid: int) -> str:
    return f"Another command (process {pid}) is still running and could not be stopped. Nothing was updated. Try again when it has finished."


@pytest.fixture
def persona() -> Path:
    path = get_persona_dir("nell")
    path.mkdir(parents=True)
    return path


@pytest.fixture(autouse=True)
def _root_logger_is_left_as_found() -> Any:
    """The transient runtime-log handler sets the root level to INFO and does not put it back (as `cmd_run`'s use of it)."""
    root = logging.getLogger()
    level = root.level
    yield
    root.setLevel(level)


class _Clock:
    """Replaces `stop_watch._clock`: records each sleep and never sleeps."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def now(self) -> float:
        return 0.0

    def sleep(self, dt: float) -> None:
        self.sleeps.append(dt)


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(stop_watch, "_clock", fake)
    return fake


class _Terminal(io.StringIO):
    """A stderr that is a terminal."""

    def isatty(self) -> bool:
        return True


def _terminal(monkeypatch: pytest.MonkeyPatch) -> _Terminal:
    term = _Terminal()
    monkeypatch.setattr(sys, "stderr", term)
    return term


def _progress(monkeypatch: pytest.MonkeyPatch, script: list[Any]) -> list[int]:
    """`engine_files.read_engine_progress` answers one entry of `script` per call (an exception instance is raised, the last
    entry repeats). Returns the list that counts the calls."""
    calls: list[int] = []

    def read(persona_dir: Path) -> Any:
        index = min(len(calls), len(script) - 1)
        calls.append(1)
        item = script[index]
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(engine_files, "read_engine_progress", read)
    return calls


def _runs(monkeypatch: pytest.MonkeyPatch, records: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(engine_files, "live_command_runs", lambda persona_dir: list(records))


def _kills(monkeypatch: pytest.MonkeyPatch, answers: dict[int, Any]) -> list[tuple[int, Any]]:
    asked: list[tuple[int, Any]] = []

    def kill_command(pid: int, token: Any, **kwargs: Any) -> KillResult:
        asked.append((pid, token))
        answer = answers[pid]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(kill, "kill_command", kill_command)
    return asked


def _handlers() -> list[logging.Handler]:
    return list(logging.getLogger().handlers)


# --- the spawn -------------------------------------------------------------------------------------------------------------


@pytest.mark.real_rebuild_step(reason="the step under test")
@pytest.mark.parametrize("env", [None, {"A_SENTINEL": "1"}])
def test_the_child_is_spawned_with_the_production_argv_three_devnull_streams_no_timeout_and_bridge_pythons_env(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, env: dict[str, str] | None
) -> None:
    monkeypatch.setattr(daemon, "bridge_python", lambda: ("a-python", env))
    assert daemon._rebuild_under_lock(persona) is None
    assert rebuild_spawns.calls == [
        (
            ["a-python", "-P", "-m", "brain.cli", "supervisor", "rebuild", "--persona", "nell"],
            {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "env": env},
        )
    ]


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_the_step_waits_for_the_child_with_no_limit(persona: Path, rebuild_spawns: RebuildPopenRecorder, clock: _Clock) -> None:
    rebuild_spawns.running_polls = 500  # far more ticks than any timeout would allow
    assert daemon._rebuild_under_lock(persona) is None
    assert len(clock.sleeps) == 500 and set(clock.sleeps) == {stop_watch.POLL_S}
    assert not rebuild_spawns.children[0].killed


# --- a failed rebuild ------------------------------------------------------------------------------------------------------


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_non_zero_exit_is_one_failed_rebuild_one_line_one_log_entry_and_the_step_ends_normally(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    rebuild_spawns.exit_code = 3
    with caplog.at_level(logging.WARNING, logger="brain.bridge.daemon"):
        assert daemon._rebuild_under_lock(persona) is None
    assert capsys.readouterr().err == "Updating your memories did not finish (exit code 3). Going on with your memories as they were saved.\n"
    assert [r.getMessage() for r in caplog.records if r.name == "brain.bridge.daemon"] == [
        "the rebuild child for nell exited with code 3; going on with the saved state"
    ]


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_an_oserror_from_the_spawn_is_one_failed_rebuild_with_its_type_and_reason(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    rebuild_spawns.error = FileNotFoundError("no such interpreter")
    with caplog.at_level(logging.WARNING, logger="brain.bridge.daemon"):
        assert daemon._rebuild_under_lock(persona) is None
    expected = "could not be started (FileNotFoundError: no such interpreter)"
    assert capsys.readouterr().err == f"Updating your memories {expected}. Going on with your memories as they were saved.\n"
    assert [r.getMessage() for r in caplog.records if r.name == "brain.bridge.daemon"] == [
        "the rebuild child for nell could not be started (FileNotFoundError: no such interpreter); going on with the saved state"
    ]


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_child_that_has_already_failed_at_the_first_poll_gives_the_failure_line_and_never_sleeps(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    term = _terminal(monkeypatch)
    reads = _progress(monkeypatch, [None])
    rebuild_spawns.exit_code = 1
    assert daemon._rebuild_under_lock(persona) is None
    assert term.getvalue().splitlines() == [
        STARTED,
        "Updating your memories did not finish (exit code 1). Going on with your memories as they were saved.",
    ]
    assert reads == [] and clock.sleeps == []


# --- the progress lines ----------------------------------------------------------------------------------------------------


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_with_a_terminal_the_spawn_line_and_each_phase_line_show_once_as_the_record_changes(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    term = _terminal(monkeypatch)
    script: list[Any] = [
        None,  # no record yet: nothing
        {"phase": "rescaling", "done": None, "total": None},
        {"phase": "rescaling", "done": None, "total": None},  # unchanged: nothing
        {"phase": "rebuilding", "done": None, "total": None},
        {"phase": "rebuilding", "done": 3, "total": 9},
        {"phase": "rebuilding", "done": 3, "total": 9},  # unchanged: nothing
        {"phase": "rebuilding", "done": 12, "total": 9},  # clamped to the total
        {"phase": "rebuilding", "done": -3, "total": 9},  # clamped to zero
        {"phase": "rescaling", "done": 2, "total": 9},  # the scale update wins over a count
    ]
    _progress(monkeypatch, script)
    rebuild_spawns.running_polls = len(script)
    assert daemon._rebuild_under_lock(persona) is None
    assert term.getvalue().splitlines() == [
        STARTED,
        RESCALING,
        PREPARING,
        "Updating your memories: 3 of 9 (33%).",
        "Updating your memories: 9 of 9 (100%).",
        "Updating your memories: 0 of 9 (0%).",
        RESCALING,
    ]
    assert clock.sleeps == [stop_watch.POLL_S] * len(script)
    assert not rebuild_spawns.children[0].killed


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_child_that_exits_at_the_first_poll_shows_only_the_spawn_line_and_reads_no_record(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    term = _terminal(monkeypatch)
    reads = _progress(monkeypatch, [{"phase": "rescaling"}])
    assert daemon._rebuild_under_lock(persona) is None
    assert term.getvalue().splitlines() == [STARTED]
    assert reads == [] and clock.sleeps == []


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_without_a_terminal_nothing_is_printed(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _progress(monkeypatch, [{"phase": "rescaling"}, {"phase": "rebuilding", "done": 1, "total": 2}])
    rebuild_spawns.running_polls = 2
    assert daemon._rebuild_under_lock(persona) is None
    assert capsys.readouterr().err == ""


class _BrokenTerminal(_Terminal):
    def write(self, text: str) -> int:
        raise BrokenPipeError("the reader went away")


class _TerminalThatCannotSay(io.StringIO):
    def __init__(self, error: type[BaseException]) -> None:
        super().__init__()
        self.error = error

    def isatty(self) -> bool:
        raise self.error("no answer")


@pytest.mark.real_rebuild_step(reason="the step under test")
@pytest.mark.parametrize("error", [AttributeError, ValueError, OSError])
def test_a_stderr_that_cannot_say_whether_it_is_a_terminal_is_not_one(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    stream = _TerminalThatCannotSay(error)
    monkeypatch.setattr(sys, "stderr", stream)
    rebuild_spawns.running_polls = 1
    _progress(monkeypatch, [{"phase": "rescaling"}])
    assert daemon._rebuild_under_lock(persona) is None
    assert stream.getvalue() == ""


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_stderr_with_no_isatty_is_not_a_terminal(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Bare:
        def __init__(self) -> None:
            self.written: list[str] = []

        def write(self, text: str) -> int:
            self.written.append(text)
            return len(text)

        def flush(self) -> None:
            pass

    bare = Bare()
    monkeypatch.setattr(sys, "stderr", bare)
    assert daemon._rebuild_under_lock(persona) is None
    assert bare.written == []


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_print_that_raises_is_ignored_and_the_child_is_not_killed(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "stderr", _BrokenTerminal())
    _progress(monkeypatch, [{"phase": "rescaling"}, {"phase": "rebuilding", "done": 1, "total": 2}])
    rebuild_spawns.running_polls = 2
    assert daemon._rebuild_under_lock(persona) is None
    child = rebuild_spawns.children[0]
    assert not child.killed and child.poll_calls == 3


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_malformed_record_or_a_failing_read_never_kills_the_child_and_never_stops_the_loop(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    term = _terminal(monkeypatch)
    script: list[Any] = [
        {"phase": 5, "done": True, "total": "x"},
        {"phase": "rebuilding", "done": 1, "total": math.nan},
        {"phase": "rebuilding", "done": 1, "total": math.inf},
        {"phase": "rebuilding", "done": 1, "total": 0},
        {"phase": "rebuilding", "done": True, "total": 4},
        OSError("the record could not be read"),
        {"phase": "rebuilding", "done": 1, "total": 4},
    ]
    _progress(monkeypatch, script)
    rebuild_spawns.running_polls = len(script)
    assert daemon._rebuild_under_lock(persona) is None
    assert term.getvalue().splitlines() == [STARTED, PREPARING, "Updating your memories: 1 of 4 (25%)."]
    child = rebuild_spawns.children[0]
    assert not child.killed and child.poll_calls == len(script) + 1


@pytest.mark.real_rebuild_step(reason="the step under test")
@pytest.mark.parametrize("raised", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("where", ["sleep", "read"])
def test_any_baseexception_while_the_child_runs_kills_and_waits_for_it_and_propagates(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, clock: _Clock, raised: type[BaseException], where: str
) -> None:
    rebuild_spawns.running_polls = 5
    if where == "sleep":

        def boom(dt: float) -> None:
            raise raised()

        monkeypatch.setattr(clock, "sleep", boom)
    else:
        _terminal(monkeypatch)  # the record is read only when stderr is a terminal
        _progress(monkeypatch, [raised()])
    before = _handlers()
    with pytest.raises(raised):
        daemon._rebuild_under_lock(persona)
    child = rebuild_spawns.children[0]
    assert child.killed and child.waited
    assert _handlers() == before


# --- the rebuild notice (Hana's 11-18 review of #633, point 5; owner S112) --------------------------------------------------


def _notice(persona: Path) -> str | None:
    record = engine_files.read_rebuild_notice(persona)
    return None if record is None else record["reason"]


CHILD_TOKEN = "linux:boot:424242:777"  # the token of the fake rebuild child (pid 424242) while it runs


def _plant(persona: Path, reason: str = "an earlier attempt's", *, pid: int = 4242, pid_token: str | None = "linux:boot:4242:99") -> None:
    engine_files.rebuild_notice_path(persona).write_text(json.dumps({"pid": pid, "pid_token": pid_token, "reason": reason}), encoding="utf-8")


def _child_token(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    """The fake child has no real process: `process_identity.token_of` answers `answer` for its pid (424242)."""
    real = process_identity.token_of
    monkeypatch.setattr(process_identity, "token_of", lambda pid: answer if pid == 424242 else real(pid))


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_case_6_a_child_that_cannot_be_started_leaves_a_notice_with_the_error(persona: Path, rebuild_spawns: RebuildPopenRecorder) -> None:
    rebuild_spawns.error = FileNotFoundError("no such interpreter")
    assert daemon._rebuild_under_lock(persona) is None
    assert _notice(persona) == "FileNotFoundError: no such interpreter"


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_child_that_exits_non_zero_with_no_notice_of_its_own_leaves_one_with_the_exit_code(
    persona: Path, rebuild_spawns: RebuildPopenRecorder
) -> None:
    rebuild_spawns.exit_code = 3
    assert daemon._rebuild_under_lock(persona) is None
    assert _notice(persona) == "the update of your memories stopped (exit code 3)"


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_child_that_exits_non_zero_after_writing_its_own_notice_keeps_it(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    rebuild_spawns.running_polls = 1
    rebuild_spawns.exit_code = 1
    _child_token(monkeypatch, CHILD_TOKEN)
    monkeypatch.setattr(
        clock, "sleep", lambda dt: _plant(persona, "the first rebuild failed (OSError: its own)", pid=424242, pid_token=CHILD_TOKEN)
    )
    assert daemon._rebuild_under_lock(persona) is None
    assert _notice(persona) == "the first rebuild failed (OSError: its own)"  # the launched child's own record is more exact


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_notice_of_another_process_is_replaced_by_a_non_zero_exit(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    _child_token(monkeypatch, CHILD_TOKEN)
    _plant(persona)  # an earlier attempt's: another pid and token
    rebuild_spawns.exit_code = 9
    assert daemon._rebuild_under_lock(persona) is None
    assert _notice(persona) == "the update of your memories stopped (exit code 9)"


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_notice_of_a_recycled_pid_with_another_token_is_replaced(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    _child_token(monkeypatch, CHILD_TOKEN)
    _plant(persona, pid=424242, pid_token="linux:boot:424242:1")  # the same pid number, an earlier process
    rebuild_spawns.exit_code = 9
    assert daemon._rebuild_under_lock(persona) is None
    assert _notice(persona) == "the update of your memories stopped (exit code 9)"


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_notice_without_a_writer_identity_is_replaced_by_a_non_zero_exit(persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
    _child_token(monkeypatch, CHILD_TOKEN)
    engine_files.rebuild_notice_path(persona).write_text('{"at": "2999-01-01T00:00:00Z", "reason": "from before the identity"}', encoding="utf-8")
    rebuild_spawns.exit_code = 9
    assert daemon._rebuild_under_lock(persona) is None
    assert _notice(persona) == "the update of your memories stopped (exit code 9)"


@pytest.mark.real_rebuild_step(reason="the step under test")
@pytest.mark.parametrize("unreadable", [process_identity.GONE, process_identity.UNKNOWN])
def test_a_child_token_that_cannot_be_read_means_the_generic_reason_is_written(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, clock: _Clock, unreadable: str
) -> None:
    _child_token(monkeypatch, unreadable)
    rebuild_spawns.running_polls = 1
    rebuild_spawns.exit_code = 1
    monkeypatch.setattr(clock, "sleep", lambda d: _plant(persona, "its own", pid=424242, pid_token=CHILD_TOKEN))
    assert daemon._rebuild_under_lock(persona) is None
    assert _notice(persona) == "the update of your memories stopped (exit code 1)"


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_child_that_completes_clears_the_notice_once_the_persona_is_switched_over(persona: Path, rebuild_spawns: RebuildPopenRecorder) -> None:
    from datetime import UTC, datetime

    from tests.unit.brain.emotion._engine_support import make_live

    make_live(persona, datetime(2030, 1, 1, tzinfo=UTC))  # what the real child's completed switch-over leaves
    _plant(persona)
    assert daemon._rebuild_under_lock(persona) is None
    assert not engine_files.rebuild_notice_path(persona).exists()


@pytest.mark.real_rebuild_step(reason="the step under test")
@pytest.mark.no_base_failure(reason="a pin of the rule's other half: on base nothing writes or deletes the notice, so it trivially stays")
def test_a_child_that_exits_zero_on_a_persona_still_not_switched_keeps_the_notice(persona: Path, rebuild_spawns: RebuildPopenRecorder) -> None:
    _plant(persona)  # an exit-0 skip (S61): the persona stays unswitched, so what the notice says still holds
    assert daemon._rebuild_under_lock(persona) is None
    assert _notice(persona) == "an earlier attempt's"


@pytest.mark.real_rebuild_step(reason="the step under test")
@pytest.mark.no_base_failure(reason="a pin of a negative: on base nothing writes the notice, so an interrupted step trivially writes none")
def test_a_step_interrupted_while_the_child_runs_writes_no_notice(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    rebuild_spawns.running_polls = 5

    def boom(dt: float) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(clock, "sleep", boom)
    with pytest.raises(KeyboardInterrupt):
        daemon._rebuild_under_lock(persona)
    assert not engine_files.rebuild_notice_path(persona).exists()


# --- ending the other commands first ---------------------------------------------------------------------------------------


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_each_running_command_is_ended_first_in_order_with_its_pid_and_token_and_none_when_the_token_is_not_a_string(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runs(
        monkeypatch,
        [
            {"pid": 11, "pid_token": "tok-11"},
            {"pid": 12, "pid_token": None},
            {"pid": 13, "pid_token": 7},
            {"pid": 14},
        ],
    )
    order: list[str] = []
    asked = _kills(
        monkeypatch,
        {
            11: KillResult("killed", "command killed", [], True),
            12: KillResult("not_running", "command not running"),
            13: KillResult("not_running", "command not running"),
            14: KillResult("not_running", "command not running"),
        },
    )
    rebuild_spawns.running_polls = 0
    real_call = rebuild_spawns.__class__.__call__

    def watching(self: RebuildPopenRecorder, argv: Any, *args: Any, **kwargs: Any) -> Any:
        order.append(f"spawn after {len(asked)} kills")
        return real_call(self, argv, *args, **kwargs)

    monkeypatch.setattr(rebuild_spawns.__class__, "__call__", watching)
    assert daemon._rebuild_under_lock(persona) is None
    assert asked == [(11, "tok-11"), (12, None), (13, None), (14, None)]
    assert order == ["spawn after 4 kills"]


@pytest.mark.real_rebuild_step(reason="the step under test")
@pytest.mark.parametrize(
    "answer",
    [
        KillResult("killed", "processes still running after the kill: [900]", [900], True),
        KillResult("unknown_identity", "identity of command 77 could not be confirmed; not killing"),
        KillResult("other_bridge", "something else"),
        RuntimeError("psutil fell over"),
    ],
    ids=["survivors", "unknown_identity", "other_outcome", "exception"],
)
def test_a_command_that_cannot_be_ended_stops_the_step_with_one_line_naming_its_pid_and_runs_nothing_else(
    persona: Path,
    rebuild_spawns: RebuildPopenRecorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    answer: Any,
) -> None:
    _runs(monkeypatch, [{"pid": 77, "pid_token": "tok"}, {"pid": 78, "pid_token": "tok"}])
    asked = _kills(monkeypatch, {77: answer, 78: KillResult("killed", "command killed", [], True)})
    with caplog.at_level(logging.WARNING, logger="brain.bridge.daemon"):
        assert daemon._rebuild_under_lock(persona) == 1
    assert capsys.readouterr().err == _could_not_end(77) + "\n"
    assert asked == [(77, "tok")]  # the next command was never asked
    assert rebuild_spawns.calls == []
    assert any("77" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_record_with_no_token_is_not_killed_by_the_real_kill_command_and_stops_the_step(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, capsys: pytest.CaptureFixture[str]
) -> None:
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        path = engine_files.command_run_path(persona, other.pid)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 1, "pid": other.pid, "pid_token": None, "persona": "nell"}), encoding="utf-8")
        assert daemon._rebuild_under_lock(persona) == 1
        assert other.poll() is None  # still running: nothing was killed without an identity to check
        assert capsys.readouterr().err == _token_less_line(other.pid, path) + "\n"
        assert rebuild_spawns.calls == []
    finally:
        other.kill()
        other.wait()


def _token_less_line(pid: int, path: Path) -> str:
    """B60 HQT: the not-ended line of a record with no token that `kill_command` answers `unknown_identity`."""
    return (
        f"Another command (process {pid}) is still running and could not be stopped: {path}. Nothing was updated. "
        "Try again when it has finished, or, if that process is not a nell command, delete that file and try again."
    )


@pytest.mark.real_rebuild_step(reason="the step under test")
@pytest.mark.parametrize("token", [None, "", 7], ids=["none", "empty", "not a string"])
def test_a_record_with_no_usable_token_that_cannot_be_confirmed_gives_its_path_and_the_delete_advice(
    persona: Path,
    rebuild_spawns: RebuildPopenRecorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    token: Any,
) -> None:
    _runs(monkeypatch, [{"pid": 77, "pid_token": token}, {"pid": 78, "pid_token": "tok"}])
    asked = _kills(monkeypatch, {77: KillResult("unknown_identity", "identity of command 77 could not be confirmed; not killing")})
    assert daemon._rebuild_under_lock(persona) == 1
    assert capsys.readouterr().err == _token_less_line(77, engine_files.command_run_path(persona, 77)) + "\n"
    assert [pid for pid, _ in asked] == [77]  # the next record was never reached
    assert rebuild_spawns.calls == []


@pytest.mark.real_rebuild_step(reason="the step under test")
@pytest.mark.no_base_failure(reason="pins the unchanged plain line: only a token-less record that cannot be confirmed gets the path")
def test_a_record_with_no_token_whose_kill_raises_keeps_the_plain_line(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _runs(monkeypatch, [{"pid": 77, "pid_token": None}])
    _kills(monkeypatch, {77: RuntimeError("psutil fell over")})
    assert daemon._rebuild_under_lock(persona) == 1
    assert capsys.readouterr().err == _could_not_end(77) + "\n"


@pytest.mark.real_rebuild_step(reason="the step under test")
@pytest.mark.no_base_failure(reason="pins the unchanged plain line: a real token that cannot be confirmed keeps it, also after a token-less record was ended")
def test_a_real_token_that_cannot_be_confirmed_keeps_the_plain_line_after_an_earlier_token_less_command_was_gone(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _runs(monkeypatch, [{"pid": 76, "pid_token": None}, {"pid": 77, "pid_token": "tok"}])
    _kills(
        monkeypatch,
        {
            76: KillResult("not_running", "command not running"),
            77: KillResult("unknown_identity", "identity of command 77 could not be confirmed; not killing"),
        },
    )
    assert daemon._rebuild_under_lock(persona) == 1
    assert capsys.readouterr().err == _could_not_end(77) + "\n"


def _unreadable_line(pid: int, path: Path) -> str:
    return (
        f"The record of another command (process {pid}) is unreadable, so that command was not stopped: {path}. "
        "Nothing was updated. If it is not running, delete that file and try again."
    )


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_an_unreadable_record_of_a_live_process_stops_the_step_without_asking_kill_command_and_the_line_gives_its_path(
    persona: Path,
    rebuild_spawns: RebuildPopenRecorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = engine_files.command_run_path(persona, 77)
    _runs(monkeypatch, [{engine_files.UNREADABLE_RUN_KEY: True, "pid": 77, "path": str(path)}, {"pid": 78, "pid_token": "tok"}])
    asked = _kills(monkeypatch, {78: KillResult("killed", "command killed", [], True)})
    with caplog.at_level(logging.WARNING, logger="brain.bridge.daemon"):
        assert daemon._rebuild_under_lock(persona) == 1
    assert capsys.readouterr().err == _unreadable_line(77, path) + "\n"
    assert asked == []  # nothing is asked of kill_command for it, and the next record was never reached
    assert rebuild_spawns.calls == []
    assert any(str(path) in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_folder_of_records_that_cannot_be_listed_stops_the_step_with_a_line_naming_the_folder(
    persona: Path,
    rebuild_spawns: RebuildPopenRecorder,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    def refuse(folder: object) -> list[str]:
        raise PermissionError("no access")

    monkeypatch.setattr(file_ops, "list_dir", refuse)
    folder = engine_files.command_runs_dir(persona)
    with caplog.at_level(logging.WARNING, logger="brain.bridge.daemon"):
        assert daemon._rebuild_under_lock(persona) == 1
    assert capsys.readouterr().err == (
        f"The commands that may be running could not be checked: the folder {folder} cannot be listed (PermissionError: no access). "
        "Nothing was updated. Check that folder and try again.\n"
    )
    assert rebuild_spawns.calls == []
    assert any(str(folder) in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_real_unreadable_record_of_a_live_process_stops_the_step_and_the_process_is_untouched(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, capsys: pytest.CaptureFixture[str]
) -> None:
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        path = engine_files.command_run_path(persona, other.pid)
        path.parent.mkdir(parents=True)
        path.write_text("this is not json {", encoding="utf-8")
        assert daemon._rebuild_under_lock(persona) == 1
        assert other.poll() is None and path.exists()  # nothing killed, the record kept
        assert capsys.readouterr().err == _unreadable_line(other.pid, path) + "\n"
        assert rebuild_spawns.calls == []
    finally:
        other.kill()
        other.wait()


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_real_running_command_with_its_token_is_ended_before_the_child_is_spawned_and_the_end_is_logged(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, caplog: pytest.LogCaptureFixture
) -> None:
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        token = process_identity.token_of(other.pid)
        path = engine_files.command_run_path(persona, other.pid)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 1, "pid": other.pid, "pid_token": token, "persona": "nell"}), encoding="utf-8")
        with caplog.at_level(logging.INFO, logger="brain.bridge.daemon"):
            assert daemon._rebuild_under_lock(persona) is None
        other.wait(timeout=20)  # ended
        assert len(rebuild_spawns.calls) == 1
        assert any(str(other.pid) in r.getMessage() and r.levelno == logging.INFO for r in caplog.records)
    finally:
        if other.poll() is None:
            other.kill()
        other.wait()


# --- the runtime log -------------------------------------------------------------------------------------------------------


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_the_ended_command_line_reaches_the_runtime_log_through_the_handler_the_step_attaches(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runs(monkeypatch, [{"pid": 31337, "pid_token": "tok"}])
    _kills(monkeypatch, {31337: KillResult("killed", "command killed", [], True)})
    before = _handlers()
    assert daemon._rebuild_under_lock(persona) is None
    text = (get_log_dir() / "runtime-nell.log").read_text(encoding="utf-8")
    assert "31337" in text and "INFO brain.bridge.daemon" in text
    assert _handlers() == before


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_the_transient_handler_is_attached_once_at_the_entry_and_only_when_none_is_attached(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    attached: list[Path] = []
    detached: list[Any] = []
    real_attach, real_detach = runner._attach_transient_runtime_log, runner._detach_runtime_log

    def attach(persona_dir: Path) -> Any:
        attached.append(persona_dir)
        return real_attach(persona_dir)

    def detach(handler: Any) -> None:
        detached.append(handler)
        real_detach(handler)

    monkeypatch.setattr(runner, "_attach_transient_runtime_log", attach)
    monkeypatch.setattr(runner, "_detach_runtime_log", detach)
    seen_during: list[bool] = []
    real_call = rebuild_spawns.__class__.__call__

    def watching(self: RebuildPopenRecorder, argv: Any, *args: Any, **kwargs: Any) -> Any:
        seen_during.append(any(isinstance(h, runner._AppendThenCloseHandler) for h in _handlers()))
        return real_call(self, argv, *args, **kwargs)

    monkeypatch.setattr(rebuild_spawns.__class__, "__call__", watching)
    assert daemon._rebuild_under_lock(persona) is None
    assert attached == [persona] and len(detached) == 1 and detached[0] is not None
    assert seen_during == [True]
    # with the persistent handler already attached (a guarded `cmd_run`), the step attaches nothing
    attached.clear()
    detached.clear()
    runner._setup_runtime_logging(persona)
    assert daemon._rebuild_under_lock(persona) is None
    assert attached == [] and detached == [None]


@pytest.mark.real_rebuild_step(reason="the step under test")
def test_a_runtime_log_that_cannot_be_set_up_does_not_stop_the_step(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log lines are best effort: when the bridge stack cannot be imported the command is still ended and the child still runs."""
    import brain.bridge as bridge_package

    _runs(monkeypatch, [{"pid": 31337, "pid_token": "tok"}])
    asked = _kills(monkeypatch, {31337: KillResult("killed", "command killed", [], True)})
    monkeypatch.delattr(bridge_package, "runner", raising=False)
    monkeypatch.setitem(sys.modules, "brain.bridge.runner", None)
    assert daemon._rebuild_under_lock(persona) is None
    assert asked == [(31337, "tok")]
    assert len(rebuild_spawns.calls) == 1
    assert sys.modules["brain.bridge.runner"] is None  # nothing in the step imported it behind the guard


@pytest.mark.real_rebuild_step(reason="the step under test")
@pytest.mark.parametrize("scenario", ["success", "failed child", "failed kill step", "interrupt"])
def test_after_the_step_the_root_loggers_handlers_are_the_list_before(
    persona: Path, rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch, clock: _Clock, scenario: str
) -> None:
    before = _handlers()
    if scenario == "failed child":
        rebuild_spawns.exit_code = 9
    elif scenario == "failed kill step":
        _runs(monkeypatch, [{"pid": 5, "pid_token": "t"}])
        _kills(monkeypatch, {5: KillResult("unknown_identity", "no")})
    elif scenario == "interrupt":
        rebuild_spawns.running_polls = 3

        def boom(dt: float) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(clock, "sleep", boom)
    if scenario == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            daemon._rebuild_under_lock(persona)
    else:
        daemon._rebuild_under_lock(persona)
    assert _handlers() == before


def test_runtime_log_attached_finds_only_the_rotating_handler_for_that_personas_log(persona: Path) -> None:
    other = get_persona_dir("other")
    other.mkdir(parents=True)
    assert runner._runtime_log_attached(persona) is False
    transient = runner._attach_transient_runtime_log(persona)
    try:
        assert runner._runtime_log_attached(persona) is False  # the append-and-close handler is not the persistent one
    finally:
        runner._detach_runtime_log(transient)
    runner._setup_runtime_logging(persona)
    assert runner._runtime_log_attached(persona) is True
    assert runner._runtime_log_attached(other) is False
    before = _handlers()
    runner._setup_runtime_logging(persona)  # still idempotent
    assert _handlers() == before


def test_runtime_log_attached_is_false_when_the_lookup_itself_fails(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> Path:
        raise OSError("no log folder")

    monkeypatch.setattr("brain.paths.get_log_dir", broken)
    assert runner._runtime_log_attached(persona) is False


# --- the conftest fixture itself (test infrastructure) ---------------------------------------------------------------------


@pytest.mark.no_base_failure(reason="test infrastructure: the conftest fixture, which exists on base too")
def test_a_test_without_the_marker_sees_the_real_subprocess_and_a_no_op_register(tmp_path: Path) -> None:
    assert daemon.subprocess is subprocess
    assert getattr(daemon, "_rebuild_needed")(tmp_path, False) is False  # noqa: B009
    engine_files.register_command_run(tmp_path)
    assert not engine_files.command_runs_dir(tmp_path).exists()


@pytest.mark.real_rebuild_step(reason="the recorder under test")
@pytest.mark.no_base_failure(reason="test infrastructure: the conftest recorder, which exists on base too")
def test_the_recorder_answers_only_the_rebuild_argv_and_passes_any_other_spawn_to_the_real_popen(
    rebuild_spawns: RebuildPopenRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert daemon.subprocess is not subprocess and daemon.subprocess.Popen is rebuild_spawns
    assert daemon.subprocess.DEVNULL == subprocess.DEVNULL
    child = daemon.subprocess.Popen([sys.executable, "-c", "pass"])  # a real spawn
    assert not isinstance(child, FakeRebuildChild) and child.wait(timeout=60) == 0
    assert rebuild_spawns.calls == []
    seen: list[Any] = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, *a, **k: seen.append(argv) or "patched")
    assert daemon.subprocess.Popen(["x", "y"]) == "patched" and seen == [["x", "y"]]  # a test's own patch takes effect
    fake = daemon.subprocess.Popen(["py", "-m", "brain.cli", "supervisor", "rebuild", "--persona", "p"])
    assert isinstance(fake, FakeRebuildChild) and seen == [["x", "y"]]  # the rebuild argv never reaches the real Popen
