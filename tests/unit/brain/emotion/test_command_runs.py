"""Engine 6, S107 step (b), HS3 (a) and (b): the run record of a command and the reader that finds the live ones.

`register_command_run(persona_dir)` writes `<persona_dir>/command_runs/<pid>.json` and removes it at exit with `atexit`, only
when the record names this process; a failed write is logged and the command goes on. `live_command_runs(persona_dir)` lists the
records of other live commands: only `<pid>.json` names count, a record that is unreadable, has no usable pid or has a dead
writer is deleted, this process's own record is left out, a failed delete is logged and that record skipped, and an unlistable
folder reads as no records. Nothing calls either function in production yet (the caller is a later step). Synthetic personas only.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from brain import dev_constants, process_identity
from brain.emotion import engine_files
from brain.utils import file_ops

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def persona(tmp_path: Path) -> Path:
    path = tmp_path / "nell"
    path.mkdir()
    return path


@pytest.fixture
def exit_steps(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Callable[..., object], tuple[object, ...]]]:
    """The `atexit` registrations `engine_files` makes, kept instead of left on the test run's exit."""
    steps: list[tuple[Callable[..., object], tuple[object, ...]]] = []
    monkeypatch.setattr(engine_files.atexit, "register", lambda func, *args, **kwargs: steps.append((func, args)))
    return steps


@pytest.fixture
def live_other() -> Iterator[subprocess.Popen]:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield child
    finally:
        child.kill()
        child.wait()


def _gone_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _put(persona: Path, pid: int, record: object, *, name: str | None = None) -> Path:
    path = engine_files.command_runs_dir(persona) / (name or f"{pid}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record if isinstance(record, str) else json.dumps(record), encoding="utf-8")
    return path


def _record_of(proc: subprocess.Popen, persona: Path) -> dict:
    return engine_files.command_run_record(pid=proc.pid, pid_token=process_identity.token_of(proc.pid), persona=persona.name)


def _logged(caplog: pytest.LogCaptureFixture, text: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if text in r.getMessage()]


# --- the record and register_command_run ---------------------------------------------------------------------------------


def test_the_folder_name_is_the_dev_constant_beside_the_progress_filename(persona: Path) -> None:
    assert dev_constants.COMMAND_RUNS_DIRNAME == "command_runs"
    assert engine_files.command_runs_dir(persona) == persona / "command_runs"
    assert engine_files.command_run_path(persona, 77) == persona / "command_runs" / "77.json"


@pytest.mark.real_rebuild_step(reason="calls the real register_command_run, which the conftest otherwise makes a no-op")
def test_register_writes_this_processs_record_and_registers_one_exit_step(
    persona: Path, exit_steps: list[tuple[Callable[..., object], tuple[object, ...]]]
) -> None:
    engine_files.register_command_run(persona)
    path = engine_files.command_run_path(persona, os.getpid())
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 1,
        "pid": os.getpid(),
        "pid_token": process_identity.own_token(),
        "persona": "nell",
    }
    assert exit_steps == [(engine_files._remove_own_command_run, (persona,))]


@pytest.mark.real_rebuild_step(reason="calls the real register_command_run, which the conftest otherwise makes a no-op")
def test_the_exit_step_removes_only_a_record_that_names_this_process(
    persona: Path, live_other: subprocess.Popen, exit_steps: list[tuple[Callable[..., object], tuple[object, ...]]]
) -> None:
    other = _put(persona, live_other.pid, _record_of(live_other, persona))
    engine_files.register_command_run(persona)
    own = engine_files.command_run_path(persona, os.getpid())
    assert own.exists()

    # The record at this process's path now names another process's identity (a pid reused after a crash): it is not ours.
    own.write_text(
        json.dumps(engine_files.command_run_record(pid=os.getpid(), pid_token="linux:other:1:1", persona="nell")), encoding="utf-8"
    )
    func, args = exit_steps[0]
    func(*args)
    assert own.exists() and other.exists()

    engine_files.register_command_run(persona)  # a fresh record of ours
    func, args = exit_steps[1]
    func(*args)
    assert not own.exists()  # removed
    assert other.exists()  # another process's record is never touched


def test_the_exit_step_with_no_record_does_nothing_and_never_raises(persona: Path) -> None:
    engine_files._remove_own_command_run(persona)
    assert not engine_files.command_runs_dir(persona).exists()


@pytest.mark.real_rebuild_step(reason="calls the real register_command_run, which the conftest otherwise makes a no-op")
def test_a_failed_register_goes_on_unregistered_and_is_logged_once(
    persona: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exit_steps: list[tuple[Callable[..., object], tuple[object, ...]]],
) -> None:
    def refuse(*args: object, **kwargs: object) -> bool:
        raise OSError("disk full")

    monkeypatch.setattr(engine_files, "write_json", refuse)
    with caplog.at_level(logging.WARNING, logger=engine_files.logger.name):
        engine_files.register_command_run(persona)
        engine_files.register_command_run(persona)  # the second failure is throttled
    assert exit_steps == []  # nothing was written, so nothing is removed at exit
    assert len(_logged(caplog, "could not write this command's run record")) == 1
    assert engine_files.live_command_runs(persona) == []


# --- live_command_runs ------------------------------------------------------------------------------------------------------


def test_a_missing_folder_reads_as_no_records_and_is_not_created(persona: Path) -> None:
    assert engine_files.live_command_runs(persona) == []
    assert not engine_files.command_runs_dir(persona).exists()


@pytest.mark.real_rebuild_step(reason="calls the real register_command_run, which the conftest otherwise makes a no-op")
def test_a_live_other_command_is_listed_and_this_processs_own_record_is_left_out(
    persona: Path, live_other: subprocess.Popen, exit_steps: list[tuple[Callable[..., object], tuple[object, ...]]]
) -> None:
    record = _record_of(live_other, persona)
    path = _put(persona, live_other.pid, record)
    engine_files.register_command_run(persona)
    assert engine_files.live_command_runs(persona) == [record]
    assert path.exists() and engine_files.command_run_path(persona, os.getpid()).exists()  # neither was deleted


def test_a_record_whose_writer_is_dead_is_deleted_and_not_listed(persona: Path) -> None:
    pid = _gone_pid()
    record = engine_files.command_run_record(pid=pid, pid_token=process_identity.own_token(), persona="nell")
    path = _put(persona, pid, record)
    assert engine_files.live_command_runs(persona) == []
    assert not path.exists()


def test_a_record_whose_pid_now_carries_another_token_is_deleted(persona: Path, live_other: subprocess.Popen) -> None:
    record = engine_files.command_run_record(pid=live_other.pid, pid_token="linux:not-this-one:1:1", persona="nell")
    path = _put(persona, live_other.pid, record)
    if not process_identity.is_token(process_identity.token_of(live_other.pid)):
        pytest.skip("this system gives no readable identity token, so a reused pid cannot be told apart")
    assert engine_files.live_command_runs(persona) == []
    assert not path.exists()


@pytest.mark.parametrize(
    "content",
    [
        "this is not json {",
        "",
        "[1, 2, 3]",
        '"a string"',
        {"version": 1, "pid_token": "t", "persona": "nell"},
        {"version": 1, "pid": "123", "pid_token": "t"},
        {"version": 1, "pid": True, "pid_token": "t"},
        {"version": 1, "pid": 0, "pid_token": "t"},
        {"version": 1, "pid": -5, "pid_token": "t"},
        {"version": 1, "pid": None},
    ],
)
def test_an_unreadable_record_or_one_with_no_usable_pid_is_deleted(persona: Path, content: object) -> None:
    path = _put(persona, 4001, content)
    assert engine_files.live_command_runs(persona) == []
    assert not path.exists()


def test_only_names_of_the_form_pid_dot_json_are_records_and_other_entries_are_left_alone(
    persona: Path, live_other: subprocess.Popen
) -> None:
    record = _record_of(live_other, persona)
    kept = [
        _put(persona, 0, record, name="notes.txt"),
        _put(persona, 0, record, name="abc.json"),
        _put(persona, 0, record, name=f"{live_other.pid}.json.tmp"),
        _put(persona, 0, record, name=f"{live_other.pid}.json.bak1"),
        _put(persona, 0, record, name=f"x{live_other.pid}.json"),
        _put(persona, 0, "garbage", name=".json"),
        _put(persona, 0, "garbage", name="1.2.json"),
    ]
    (engine_files.command_runs_dir(persona) / "5.json").mkdir()  # a folder with a record's name
    assert engine_files.live_command_runs(persona) == []
    assert all(path.exists() for path in kept)
    assert (engine_files.command_runs_dir(persona) / "5.json").is_dir()


def test_a_failed_delete_is_logged_and_that_record_skipped_without_failing_the_read(
    persona: Path, live_other: subprocess.Popen, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dead = _put(persona, _gone_pid(), "this is not json {", name="4002.json")
    live = _record_of(live_other, persona)
    _put(persona, live_other.pid, live)

    def refuse(path: object) -> None:
        raise PermissionError("held open")

    monkeypatch.setattr(file_ops, "delete_file", refuse)
    with caplog.at_level(logging.WARNING, logger=engine_files.logger.name):
        runs = engine_files.live_command_runs(persona)
    assert runs == [live]  # the live record is still returned
    assert dead.exists()
    assert len(_logged(caplog, "could not read or delete the command run record")) == 1


def test_a_record_that_cannot_be_read_because_of_an_io_fault_is_skipped_logged_and_kept(
    persona: Path, live_other: subprocess.Popen, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = _put(persona, live_other.pid, _record_of(live_other, persona))

    def broken(*args: object, **kwargs: object) -> object:
        raise OSError("input/output error")

    monkeypatch.setattr(engine_files, "read_json", broken)
    with caplog.at_level(logging.WARNING, logger=engine_files.logger.name):
        assert engine_files.live_command_runs(persona) == []
    assert path.exists()  # it may name a live command: never deleted for a read fault
    assert len(_logged(caplog, "could not read or delete the command run record")) == 1


def test_a_list_dir_error_is_logged_and_read_as_no_records(
    persona: Path, live_other: subprocess.Popen, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = _put(persona, live_other.pid, _record_of(live_other, persona))

    def refuse(folder: object) -> list[str]:
        raise PermissionError("no access")

    monkeypatch.setattr(file_ops, "list_dir", refuse)
    with caplog.at_level(logging.WARNING, logger=engine_files.logger.name):
        assert engine_files.live_command_runs(persona) == []
    assert path.exists()
    assert len(_logged(caplog, "could not list the command run records")) == 1


def test_records_come_back_in_file_name_order(persona: Path, live_other: subprocess.Popen) -> None:
    second = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        records = [_record_of(live_other, persona), _record_of(second, persona)]
        for record in records:
            _put(persona, record["pid"], record)
        names = sorted(str(r["pid"]) + ".json" for r in records)
        assert [f"{r['pid']}.json" for r in engine_files.live_command_runs(persona)] == names
    finally:
        second.kill()
        second.wait()


# --- the real process: atexit runs for a normal exit and a killed one leaves the record for the next reader ----------------


def _run_child(persona: Path, exit_with: str) -> None:
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from brain.emotion import engine_files\n"
        "engine_files.register_command_run(Path(sys.argv[1]))\n"
        "assert engine_files.command_run_path(Path(sys.argv[1]), __import__('os').getpid()).exists()\n"
        + exit_with
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")]))
    result = subprocess.run(
        [sys.executable, "-P", "-c", code, str(persona)], capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr


def test_a_command_that_exits_normally_removes_its_record_at_exit(persona: Path) -> None:
    _run_child(persona, "sys.exit(0)\n")
    assert list(engine_files.command_runs_dir(persona).iterdir()) == []


def test_a_command_that_dies_without_running_exit_steps_leaves_a_record_the_next_read_deletes(persona: Path) -> None:
    _run_child(persona, "__import__('os')._exit(0)\n")
    left = list(engine_files.command_runs_dir(persona).iterdir())
    assert len(left) == 1  # no atexit ran
    assert engine_files.live_command_runs(persona) == []  # its writer is dead
    assert not left[0].exists()
