"""``locked_within`` (#529, C42 a-d, f): a bounded-wait cross-process lock that never enters its body on timeout."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from brain import dev_constants
from brain.utils.file_lock import file_lock, locked_within


def test_timeout_raises_and_the_body_never_runs(tmp_path: Path, wedged_lock) -> None:
    """C42 (a): against a wedged holder in ANOTHER process the helper raises TimeoutError and the body is not entered."""
    target = tmp_path / "data.json"
    ran: list[str] = []
    with wedged_lock(target), pytest.raises(TimeoutError), locked_within(target, 0.3):
        ran.append("body")  # pragma: no cover - must never run
    assert ran == []


def test_success_enters_the_body_exactly_once(tmp_path: Path) -> None:
    """C42 (b)."""
    target = tmp_path / "data.json"
    entered: list[int] = []
    with locked_within(target, 1.0):
        entered.append(1)
    assert entered == [1]


def test_a_body_exception_propagates_once_and_releases_the_lock(tmp_path: Path) -> None:
    """C42 (c): no retry of the body, and a second acquirer gets the lock at once afterwards."""
    target = tmp_path / "data.json"
    calls: list[int] = []

    class BoomError(Exception):
        pass

    with pytest.raises(BoomError), locked_within(target, 2.0):
        calls.append(1)
        raise BoomError
    assert calls == [1]
    started = time.monotonic()
    with locked_within(target, 2.0):
        pass
    assert time.monotonic() - started < 1.0


def test_a_false_flag_from_the_nonblocking_form_never_runs_the_body(tmp_path: Path, monkeypatch) -> None:
    """C42 (d): ``file_lock(blocking=False)`` yields False and still runs its own body; the helper must not follow it."""
    from contextlib import contextmanager

    import brain.utils.file_lock as fl

    @contextmanager
    def never_acquires(path, *, blocking=True):
        yield False

    monkeypatch.setattr(fl, "file_lock", never_acquires)
    ran: list[int] = []
    with pytest.raises(TimeoutError), fl.locked_within(tmp_path / "x.json", 0.2):
        ran.append(1)  # pragma: no cover
    assert ran == []


def test_the_real_nonblocking_lock_is_released_between_polls(tmp_path: Path) -> None:
    """A lock held briefly by another thread is acquired within the wait, so the poll loop does not give up early."""
    import threading

    target = tmp_path / "data.json"
    released = threading.Event()

    def holder() -> None:
        with file_lock(target):
            time.sleep(0.3)
        released.set()

    t = threading.Thread(target=holder)
    t.start()
    time.sleep(0.05)
    started = time.monotonic()
    with locked_within(target, 5.0):
        assert time.monotonic() - started >= 0.1  # it waited for the holder, then got the lock
    t.join()
    assert released.is_set()


def test_wait_constant_covers_the_live_retry_constants_and_stays_under_the_webview_cut() -> None:
    """C42 (f): the constant is at least R_read + (K+1) * R_replace computed from the LIVE constants.

    K is the most replaces one ``write_defaults_section`` heal performs while it holds the tunables lock: the heal's own
    install (1), the corrupt backups it quarantines (3 at most: ``.bak1`` to ``.bak3``) and the notice file it writes
    (1); the payload write after it is the "+ 1". Changing a retry constant without re-deriving the wait fails here.
    """
    from brain.health import attempt_heal, json_io

    k_heal_replaces = 1 + 3 + 1
    step = attempt_heal._REPLACE_RETRY_SLEEP_S
    r_replace = step * sum(range(1, attempt_heal._REPLACE_RETRIES))
    r_read = step * sum(range(1, json_io._READ_ATTEMPTS))
    needed = r_read + (k_heal_replaces + 1) * r_replace
    assert dev_constants.SETTINGS_FILE_LOCK_WAIT_S >= needed
    # The picker request spends the 30 s probe budget first, then the write; the webview cuts a request at about 60 s.
    from brain.bridge import server

    probe_budget = server._MODEL_CHECK_TIMEOUT_S
    webview_cut = 60.0
    assert probe_budget + dev_constants.SETTINGS_FILE_LOCK_WAIT_S < webview_cut
    assert dev_constants.SETTINGS_FILE_LOCK_WAIT_S < probe_budget


def test_the_wait_and_poll_constants_are_plain_json_safe_numbers() -> None:
    json.dumps([dev_constants.SETTINGS_FILE_LOCK_WAIT_S, dev_constants.SETTINGS_FILE_LOCK_POLL_S])
    assert 0 < dev_constants.SETTINGS_FILE_LOCK_POLL_S < dev_constants.SETTINGS_FILE_LOCK_WAIT_S
