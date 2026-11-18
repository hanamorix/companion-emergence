"""Engine 6, S107 step (a), HS1: `nell supervisor rebuild --persona <name>`, the one-shot rebuild as its own process.

The subcommand is internal (a later step has the supervisor's start spawn it before the bridge). In order it: writes the first
progress record (phase "rebuilding") before anything slow, including the import of the bridge stack; sets up the runtime log;
opens no store and loads no vocabulary itself; and calls `ensure_engine_ready(persona_dir)`, whose failures nothing catches
(`main()` logs the crash and the process exits 1). The record is removed in a `finally`. Success, S61's skip and a logged
roll-forward failure all exit 0. A bridge recorded running is refused first (exit 2, one stderr line, nothing written or run). The bridge conftest points `KINDLED_HOME` at a tmp home; personas are synthetic.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from brain import cli as brain_cli
from brain.bridge import runner, state_file
from brain.cli import main
from brain.emotion import engine, engine_files, engine_rebuild, nudge_scale_fix
from brain.emotion.engine import EngineStartupError
from brain.emotion.engine_utils import day_of
from brain.health.anomaly import BrainIntegrityError
from brain.paths import get_log_dir, get_persona_dir, runtime_log_path
from tests.unit.brain.emotion._engine_support import (
    NOW,
    OtherHolder,
    install,
    local_day,
    local_noon,
    make_persona,
    row,
    ruin_the_state,
    save,
    simple_state,
)

T20 = local_noon(local_day(20))
T34 = local_noon(local_day(34))
ARGV = ["supervisor", "rebuild", "--persona", "nell"]
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def persona(tmp_path: Path) -> Path:
    """The persona `nell` under the test's home, with a database of one counted row and an empty vocabulary file."""
    path = get_persona_dir("nell")
    make_persona(path, [row(T20, emotions={"joy": 6.0}, mid="seed")])
    (path / "emotion_vocabulary.json").write_text('{"version": 1, "emotions": []}', encoding="utf-8")
    return path


def _record(persona_dir: Path) -> dict | None:
    return engine_files.read_progress(persona_dir)


class _Recorder:
    """Stands in for `ensure_engine_ready` and the runtime-logging setup; keeps what the progress record held at each call."""

    def __init__(self, persona_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.persona_dir = persona_dir
        self.events: list[tuple[str, dict | None]] = []
        self.ensure_args: list[tuple[object, ...]] = []
        self.ensure_error: BaseException | None = None
        monkeypatch.setattr(runner, "_setup_runtime_logging", self._logging)
        monkeypatch.setattr(brain_cli.emotion_engine, "ensure_engine_ready", self._ensure)

    def _logging(self, persona_dir: Path) -> None:
        self.events.append(("logging", _record(self.persona_dir)))

    def _ensure(self, *args: object, **kwargs: object) -> None:
        assert not kwargs
        self.ensure_args.append(args)
        self.events.append(("ensure", _record(self.persona_dir)))
        if self.ensure_error is not None:
            raise self.ensure_error


# --- the parser -------------------------------------------------------------------------------------------------------


def test_the_parser_has_the_rebuild_action_with_a_required_persona_and_a_one_line_internal_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = brain_cli._build_parser()
    args = parser.parse_args(ARGV)
    assert args.func is brain_cli._supervisor_rebuild_handler and args.persona == "nell"
    with pytest.raises(SystemExit):
        parser.parse_args(["supervisor", "rebuild"])  # --persona is required
    capsys.readouterr()
    with pytest.raises(SystemExit):
        parser.parse_args(["supervisor", "--help"])
    listing = [line for line in capsys.readouterr().out.splitlines() if "rebuild" in line]
    assert listing and any("internal" in line.lower() for line in listing)


# --- the order, the record and its removal ----------------------------------------------------------------------------


def test_the_first_record_is_written_before_the_runtime_log_is_set_up_and_before_the_engine_call_and_is_gone_after(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _Recorder(persona, monkeypatch)
    assert main(ARGV) == 0
    assert [name for name, _ in recorder.events] == ["logging", "ensure"]
    for _, seen in recorder.events:
        assert seen is not None and seen["phase"] == "rebuilding" and seen["persona"] == "nell"
    assert recorder.ensure_args == [(persona,)]  # the existing call, one argument
    assert _record(persona) is None


def test_the_first_record_is_written_before_the_bridge_stack_runner_module_is_imported(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict | None] = []

    class _Finder:
        """First on the path: reads the record when the import machinery asks for the runner, then refuses the import."""

        @staticmethod
        def find_spec(name: str, path: object = None, target: object = None) -> None:
            if name == "brain.bridge.runner":
                seen.append(_record(persona))
                raise ImportError("stopped at the runner import")

    import brain.bridge

    monkeypatch.delitem(sys.modules, "brain.bridge.runner")
    monkeypatch.delattr(brain.bridge, "runner")
    monkeypatch.setattr(sys, "meta_path", [_Finder, *sys.meta_path])
    with pytest.raises(ImportError):
        main(ARGV)
    assert len(seen) == 1 and seen[0] is not None and seen[0]["phase"] == "rebuilding"
    assert _record(persona) is None  # removed by the finally even though the import failed


def test_the_record_is_removed_in_a_finally_when_the_engine_call_fails_and_the_failure_reaches_the_crash_log(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _Recorder(persona, monkeypatch)
    recorder.ensure_error = EngineStartupError("the first rebuild", OSError("disk full"))
    with pytest.raises(EngineStartupError):
        main(ARGV)  # nothing catches it: main() logs it and raises, so the process exits 1
    assert _record(persona) is None
    crash_log = (get_log_dir() / "cli-crash.log").read_text(encoding="utf-8")
    assert "EngineStartupError" in crash_log and "the first rebuild" in crash_log and "supervisor" in crash_log


def test_the_record_is_removed_when_the_runtime_log_setup_raises_too(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(persona_dir: Path) -> None:
        raise RuntimeError("log setup exploded")

    monkeypatch.setattr(runner, "_setup_runtime_logging", boom)
    with pytest.raises(RuntimeError):
        main(ARGV)
    assert _record(persona) is None


def test_the_handler_opens_no_store_and_loads_no_vocabulary_itself(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.emotion import persona_loader
    from brain.memory import store as store_module

    _Recorder(persona, monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the rebuild process opens no store and loads no vocabulary of its own")

    monkeypatch.setattr(store_module, "MemoryStore", forbidden)
    monkeypatch.setattr(brain_cli, "MemoryStore", forbidden, raising=False)
    monkeypatch.setattr(brain_cli, "load_persona_vocabulary", forbidden)
    monkeypatch.setattr(persona_loader, "load_persona_vocabulary", forbidden)
    assert main(ARGV) == 0


# --- the exit code ----------------------------------------------------------------------------------------------------


def test_s61_skip_exits_zero_and_leaves_no_record(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable state and a database that fails its integrity check: this start is skipped with a warning, exit 0."""
    save(persona, simple_state(saved_at=T20))
    ruin_the_state(persona)

    def broken(*args: object, **kwargs: object) -> None:
        raise BrainIntegrityError(str(persona / "memories.db"), "page 7 is damaged")

    monkeypatch.setattr(nudge_scale_fix, "MemoryStore", broken)
    install(persona, NOW)
    assert main(ARGV) == 0
    assert not engine.engine_for(persona).is_live()  # skipped: nothing was switched over
    assert _record(persona) is None


def test_a_roll_forward_failure_that_the_engine_logs_exits_zero(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved = local_noon(local_day(30))
    state = simple_state(saved_at=saved, day=day_of(saved), switch_over_at=T20)
    state.rows_through = 1
    save(persona, state)
    install(persona, NOW)

    def broken(*args: object, **kwargs: object) -> object:
        raise OSError("disk I/O error")

    monkeypatch.setattr(engine_rebuild, "_read_window", broken)
    assert main(ARGV) == 0  # ensure_ready logs the failed check and goes on
    assert _record(persona) is None


def test_a_persona_that_does_not_exist_is_refused_and_nothing_is_made_or_run(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = _Recorder(persona, monkeypatch)
    missing = get_persona_dir("nel")  # a mistyped name
    assert not missing.exists()
    assert main(["supervisor", "rebuild", "--persona", "nel"]) == 1
    assert f"persona directory not found: {missing}" in capsys.readouterr().err
    assert not missing.exists()  # no folder was made, so no record and no empty persona
    assert recorder.events == []  # neither the runtime log nor the engine was reached


RUNNING_BRIDGE_LINE = "A bridge is running for this persona, so the rebuild was not run."


def _tree(root: Path) -> list[tuple[str, int]]:
    return sorted((str(p.relative_to(root)), p.stat().st_size if p.is_file() else -1) for p in root.rglob("*"))


def test_a_hand_run_beside_a_running_bridge_is_refused_with_one_line_and_runs_and_writes_nothing(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hana's 11-18 review of #633, point 3 (Planning's HRV answer): the bridge's committed-but-unapplied rows would read as a roll-forward."""
    recorder = _Recorder(persona, monkeypatch)
    monkeypatch.setattr(state_file, "is_running", lambda persona_dir: True)
    before = _tree(persona)
    assert main(ARGV) == 2
    assert capsys.readouterr().err == RUNNING_BRIDGE_LINE + "\n"
    assert recorder.events == []  # neither the runtime log nor the engine was reached, and no record was written before the refusal
    assert _record(persona) is None
    assert _tree(persona) == before  # nothing was written anywhere under the persona


@pytest.mark.no_base_failure(reason="a control: with no bridge recorded running the child runs as it always did, on base too")
def test_with_no_bridge_running_the_same_hand_run_goes_on_to_the_engine(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = _Recorder(persona, monkeypatch)
    monkeypatch.setattr(state_file, "is_running", lambda persona_dir: False)
    assert main(ARGV) == 0
    assert [name for name, _ in recorder.events] == ["logging", "ensure"]
    assert RUNNING_BRIDGE_LINE not in capsys.readouterr().err


# --- the real process: the entry point and the exit code ------------------------------------------------------------------


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")]))
    return env


def _run_child() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-P", "-m", "brain.cli", *ARGV],
        capture_output=True,
        text=True,
        env=_child_env(),
        cwd=str(REPO_ROOT),
        timeout=300,
        check=False,
    )


def test_a_real_child_on_a_small_unswitched_persona_exits_zero_and_leaves_it_switched_over(persona: Path) -> None:
    result = _run_child()
    assert result.returncode == 0, result.stderr
    assert engine_files.read_state(persona).state is not None
    assert engine_files.switch_over_state(engine_files.read_state(persona)) is True
    assert _record(persona) is None


def test_a_real_child_waits_for_a_held_engine_lock_past_one_wait_step_and_finishes_when_it_frees(persona: Path) -> None:
    """S101, S107 (HS8): the child no longer gives up after one `EMOTION_LOCK_WAIT_S` step (5 s) on a lock another process holds.

    The wait is observed inside a step: the child's own log line in the runtime log ("start-up waits for it") appears only after
    a first step ran out, and only the waiting code writes it. The child is then still running; after the holder lets go it
    takes the lock, rebuilds and exits 0. Bounded: the log is polled for at most 60 s, the exit waited for at most 120 s, and the
    lock is released and the child killed in a `finally` (a child that waits for ever must not outlive a failing test).
    """
    log = runtime_log_path("nell")
    holder = OtherHolder(persona)
    child = subprocess.Popen(
        [sys.executable, "-P", "-m", "brain.cli", *ARGV],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_child_env(),
        cwd=str(REPO_ROOT),
    )
    try:
        waited = False
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and child.poll() is None:
            if log.exists() and "start-up waits for it" in log.read_text(encoding="utf-8", errors="replace"):
                waited = True
                break
            time.sleep(0.1)
        assert waited, f"the child never waited for the held lock (exit code {child.poll()})"
        assert child.poll() is None  # still running after a whole step ran out
        holder.release()
        assert child.wait(timeout=120) == 0
        assert engine_files.switch_over_state(engine_files.read_state(persona)) is True
        assert _record(persona) is None
    finally:
        holder.release()
        if child.poll() is None:
            child.kill()
            child.wait()


def test_a_real_child_on_a_database_that_is_not_a_database_exits_one_and_the_crash_log_holds_the_error(persona: Path) -> None:
    (persona / "memories.db").write_bytes(b"this is not a sqlite database" * 40)
    result = _run_child()
    assert result.returncode == 1
    crash_log = (get_log_dir() / "cli-crash.log").read_text(encoding="utf-8")
    assert "EngineStartupError" in crash_log
    assert _record(persona) is None
