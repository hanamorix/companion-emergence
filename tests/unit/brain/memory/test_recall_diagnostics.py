"""Tests for brain.memory.recall_diagnostics (name-recall fix, increment D1).

Criteria: C10 (record shape, path, prune), CONC-3 (prune vs append), INV-I13a
(`os.replace` failure). Every test is deterministic: clocks are injected, the
CONC-3 window is opened by a seam on `os.replace` (no sleeps that decide the
outcome), and all data is synthetic in `tmp_path`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from brain import dev_constants
from brain.memory import recall_diagnostics as rd

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clean_pending_buffer():
    """The diagnostics buffer is module state (S189): no test inherits another's."""
    rd._pending.clear()
    yield
    rd._pending.clear()


def _para(
    path: str = "reranked",
    width: int = 5,
    pass_mark: float | None = 0.4,
    scale: str | None = "normalized",
):
    return rd.ParagraphDiagnostic(path=path, width=width, pass_mark=pass_mark, scale=scale)


def _kwargs(**over):
    base = {
        "source": "passive",
        "paragraph_count": 2,
        "whole_message_fallback": False,
        "total_width": 12,
        "budget": 4.5,
        "paragraphs": [_para(), _para("cosine", 7, 0.31, "cosine")],
        "now": NOW,
    }
    base.update(over)
    return base


def _rec_line(ts: datetime, n: int = 0) -> str:
    return json.dumps({"ts": ts.isoformat(), "source": "passive", "n": n})


def _write(path: Path, lines: list[str], *, eol: str = "\n") -> bytes:
    data = "".join(line + eol for line in lines).encode()
    path.write_bytes(data)
    return data


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- path (S59) --------------------------------------------------------------


def test_diagnostics_path_is_the_persona_directory_file(tmp_path):
    assert dev_constants.RECALL_DIAGNOSTICS_LOG_FILENAME == "recall_diagnostics.log.jsonl"
    assert rd.diagnostics_path(tmp_path) == tmp_path / "recall_diagnostics.log.jsonl"
    # a str persona dir works too
    assert rd.diagnostics_path(str(tmp_path)) == tmp_path / "recall_diagnostics.log.jsonl"


# --- record shape (C10) ------------------------------------------------------


def test_build_record_passive_shape():
    rec = rd.build_record(**_kwargs())
    assert list(rec) == [
        "ts",
        "source",
        "paragraph_count",
        "whole_message_fallback",
        "total_width",
        "budget",
        "paragraphs",
    ]
    assert rec["ts"] == "2026-09-29T12:00:00+00:00"
    assert rec["source"] == "passive"
    assert "mode" not in rec
    assert rec["paragraph_count"] == 2
    assert rec["whole_message_fallback"] is False
    assert rec["total_width"] == 12
    assert rec["budget"] == 4.5
    assert rec["paragraphs"] == [
        {"path": "reranked", "width": 5, "pass_mark": 0.4, "scale": "normalized"},
        {"path": "cosine", "width": 7, "pass_mark": 0.31, "scale": "cosine"},
    ]


def test_build_record_tool_carries_mode_and_fallback_flag():
    rec = rd.build_record(**_kwargs(source="tool", mode="semantic", whole_message_fallback=True))
    assert rec["source"] == "tool"
    assert rec["mode"] == "semantic"
    assert rec["whole_message_fallback"] is True


def test_build_record_clock_is_injected_and_normalised_to_utc():
    eastern = datetime(2026, 9, 29, 4, 0, 0, tzinfo=timezone(timedelta(hours=-4)))
    assert rd.build_record(**_kwargs(now=eastern))["ts"] == "2026-09-29T08:00:00+00:00"
    naive = datetime(2026, 1, 2, 3, 4, 5)
    assert rd.build_record(**_kwargs(now=naive))["ts"] == "2026-01-02T03:04:05+00:00"


def test_build_record_non_finite_numbers_become_null_so_the_line_is_strict_json():
    rec = rd.build_record(
        **_kwargs(budget=float("inf"), paragraphs=[_para(pass_mark=float("nan"))])
    )
    assert rec["budget"] is None
    assert rec["paragraphs"][0]["pass_mark"] is None
    # strict JSON: no NaN/Infinity tokens
    json.dumps(rec, allow_nan=False)


def test_build_record_allows_null_budget_pass_mark_and_scale():
    rec = rd.build_record(**_kwargs(budget=None, paragraphs=[_para(pass_mark=None, scale=None)]))
    assert rec["budget"] is None
    assert rec["paragraphs"][0]["pass_mark"] is None
    assert rec["paragraphs"][0]["scale"] is None


def test_build_record_empty_paragraph_list_is_valid():
    rec = rd.build_record(**_kwargs(paragraph_count=0, total_width=0, paragraphs=[]))
    assert rec["paragraphs"] == []


def test_build_record_rejects_unknown_source_and_path():
    with pytest.raises(ValueError):
        rd.build_record(**_kwargs(source="heartbeat"))
    with pytest.raises(ValueError):
        rd.build_record(**_kwargs(paragraphs=[_para(path="lexical")]))


# --- append (C10) ------------------------------------------------------------


def test_log_recall_appends_exactly_one_line_per_call_to_the_named_file(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    assert rd.log_recall(path, **_kwargs()) is True
    assert rd.log_recall(path, **_kwargs(source="tool", mode="lexical", paragraphs=[])) is True
    raw = path.read_bytes()
    assert raw.count(b"\n") == 2 and raw.endswith(b"\n")
    recs = _read_records(path)
    assert [r["source"] for r in recs] == ["passive", "tool"]
    assert recs[1]["mode"] == "lexical"
    # the only files in the persona dir: the log and the lock sidecar file_lock keeps beside it
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "recall_diagnostics.log.jsonl",
        "recall_diagnostics.log.jsonl.lock",
    ]


def test_append_round_trips_non_ascii_text_and_writes_pure_ascii_lines(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    assert rd.append_record(path, {"ts": NOW.isoformat(), "scale": "échelle\u2028"}) is True
    raw = path.read_bytes()
    assert raw.isascii()  # no U+2028 for a str.splitlines() reader to split on
    assert len(raw.decode().splitlines()) == 1
    assert _read_records(path)[0]["scale"] == "échelle\u2028"


def test_concurrent_appends_all_land_as_intact_lines(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    errors: list[BaseException] = []

    def worker(k: int) -> None:
        try:
            for i in range(10):
                assert rd.log_recall(
                    path, **_kwargs(paragraph_count=k * 100 + i, now=datetime.now(UTC))
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    rd.flush_pending(path)  # an append that met a held lock is buffered until the next write or lull
    recs = _read_records(path)
    assert sorted(r["paragraph_count"] for r in recs) == sorted(
        k * 100 + i for k in range(4) for i in range(10)
    )


# --- fail-soft (never raises into recall) -----------------------------------


def test_log_recall_never_raises_when_the_path_cannot_be_opened(tmp_path, caplog):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    bad = blocker / "recall_diagnostics.log.jsonl"  # parent is a regular file
    with caplog.at_level(logging.WARNING, logger=rd.logger.name):
        assert rd.log_recall(bad, **_kwargs()) is False
    assert any("append" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    assert not any(r.exc_info for r in caplog.records)  # no traceback per call


def test_log_recall_never_raises_on_bad_input_and_writes_nothing(tmp_path, caplog):
    path = rd.diagnostics_path(tmp_path)
    with caplog.at_level(logging.ERROR, logger=rd.logger.name):
        assert rd.log_recall(path, **_kwargs(source="bogus")) is False
        assert rd.log_recall(path, **_kwargs(paragraphs=[_para(path="bogus")])) is False
        assert rd.log_recall(path, **_kwargs(total_width="not a number")) is False
    assert not path.exists()
    assert len([r for r in caplog.records if r.exc_info]) == 3


def test_append_never_raises_on_an_unserialisable_record(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    # A NaN used to drop the whole record; json_io writes it as null (BC4: telemetry is never lost to a float).
    assert rd.append_record(path, {"ts": NOW.isoformat(), "x": float("nan")}) is True
    assert _read_records(path)[0]["x"] is None
    other = rd.diagnostics_path(tmp_path / "second")
    cyclic: dict = {"ts": NOW.isoformat()}
    cyclic["self"] = cyclic
    assert rd.append_record(other, cyclic) is False  # a genuinely unencodable record is still dropped, never raised
    assert not other.exists() or other.read_bytes() == b""


def test_append_is_fail_soft_when_the_write_itself_raises(tmp_path, monkeypatch, caplog):
    path = rd.diagnostics_path(tmp_path)

    def boom(*a, **k):
        raise OSError("disk full")

    from brain.health import json_io

    monkeypatch.setattr(json_io, "open", boom, raising=False)  # the module's own `open`, used by the append
    with caplog.at_level(logging.WARNING, logger=rd.logger.name):
        assert rd.log_recall(path, **_kwargs()) is False
    assert any("append" in r.getMessage() for r in caplog.records)


# --- prune (C10) -------------------------------------------------------------


def test_prune_removes_older_keeps_newer_and_returns_the_count(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    lines = [
        _rec_line(NOW - timedelta(days=10), 0),
        _rec_line(NOW - timedelta(days=3), 1),
        _rec_line(NOW - timedelta(days=1), 2),
        _rec_line(NOW - timedelta(minutes=1), 3),
    ]
    _write(path, lines)
    assert rd.prune(path, window_days=2.0, now=NOW) == 2
    assert [r["n"] for r in _read_records(path)] == [2, 3]
    # idempotent
    before = path.read_bytes()
    assert rd.prune(path, window_days=2.0, now=NOW) == 0
    assert path.read_bytes() == before


def test_prune_boundary_record_exactly_at_the_cutoff_is_kept(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    cutoff = NOW - timedelta(days=2)
    _write(path, [_rec_line(cutoff - timedelta(microseconds=1), 0), _rec_line(cutoff, 1)])
    assert rd.prune(path, window_days=2.0, now=NOW) == 1
    assert [r["n"] for r in _read_records(path)] == [1]


def test_prune_with_nothing_to_remove_does_not_rewrite(tmp_path, monkeypatch):
    path = rd.diagnostics_path(tmp_path)
    original = _write(path, [_rec_line(NOW - timedelta(hours=1))])
    monkeypatch.setattr(os, "replace", lambda *a: pytest.fail("rewrote an unchanged file"))
    assert rd.prune(path, window_days=2.0, now=NOW) == 0
    assert path.read_bytes() == original


def test_prune_a_missing_file_returns_zero_and_creates_nothing(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    assert rd.prune(path, window_days=2.0, now=NOW) == 0
    assert list(tmp_path.iterdir()) == []


def test_prune_everything_old_leaves_an_empty_file_that_still_accepts_appends(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    _write(path, [_rec_line(NOW - timedelta(days=9), 0), _rec_line(NOW - timedelta(days=8), 1)])
    assert rd.prune(path, window_days=1.0, now=NOW) == 2
    assert path.read_bytes() == b""
    assert rd.log_recall(path, **_kwargs()) is True
    assert len(_read_records(path)) == 1


def test_prune_removes_lines_it_cannot_age_and_leaves_no_temp_file(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    good = _rec_line(NOW - timedelta(hours=1), 7)
    _write(
        path,
        [
            "{not json",  # torn write
            '{"no_ts": 1}',
            '{"ts": "yesterday-ish"}',
            "[1, 2, 3]",  # valid JSON, not an object
            '{"ts": 5}',  # ts of the wrong type
            good,
        ],
    )
    (tmp_path / "recall_diagnostics.log.jsonl").write_bytes(
        path.read_bytes() + b"\xff\xfe not utf8\n"
    )
    assert rd.prune(path, window_days=2.0, now=NOW) == 6
    assert [r["n"] for r in _read_records(path)] == [7]
    assert not (tmp_path / "recall_diagnostics.log.jsonl.tmp").exists()


def test_prune_removes_lines_whose_ts_or_json_breaks_the_parser_with_an_unexpected_error(tmp_path):
    """An out-of-range but well-formed ts raises OverflowError inside the astimezone
    conversion and deeply nested JSON raises RecursionError; neither may wedge the prune."""
    path = rd.diagnostics_path(tmp_path)
    _write(
        path,
        [
            '{"ts": "0001-01-01T00:00:00+05:00"}',
            '{"ts": "9999-12-31T23:59:59-05:00"}',
            "[" * 100_000,
            _rec_line(NOW - timedelta(hours=1), 5),
        ],
    )
    assert rd.prune(path, window_days=2.0, now=NOW) == 3
    assert [r["n"] for r in _read_records(path)] == [5]


def test_prune_handles_crlf_and_blank_lines(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    data = (
        _rec_line(NOW - timedelta(days=9), 0)
        + "\r\n\r\n"
        + _rec_line(NOW - timedelta(hours=1), 1)
        + "\r\n"
        + _rec_line(NOW - timedelta(hours=2), 2)  # no trailing newline
    )
    path.write_bytes(data.encode())
    assert rd.prune(path, window_days=2.0, now=NOW) == 1
    assert [r["n"] for r in _read_records(path)] == [1, 2]


def test_prune_treats_a_naive_ts_as_utc(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    naive_old = (NOW - timedelta(days=5)).replace(tzinfo=None).isoformat()
    naive_new = (NOW - timedelta(hours=1)).replace(tzinfo=None).isoformat()
    _write(path, [json.dumps({"ts": naive_old, "n": 0}), json.dumps({"ts": naive_new, "n": 1})])
    assert rd.prune(path, window_days=2.0, now=NOW) == 1
    assert [r["n"] for r in _read_records(path)] == [1]


def test_prune_default_window_is_the_calibration_logs_retention_tunable(tmp_path, monkeypatch):
    from brain import tunables
    from brain.memory.store import CALIBRATION_LOG_RETENTION_WINDOW_DAYS

    seen: list[tuple[str, float]] = []

    def fake_get_tunable(key, default):
        seen.append((key, default))
        return 1.0

    monkeypatch.setattr(tunables, "get_tunable", fake_get_tunable)
    path = rd.diagnostics_path(tmp_path)
    _write(path, [_rec_line(NOW - timedelta(days=2), 0), _rec_line(NOW - timedelta(hours=1), 1)])
    assert rd.prune(path, now=NOW) == 1
    assert seen == [("calibration.retention_window_days", CALIBRATION_LOG_RETENTION_WINDOW_DAYS)]
    assert [r["n"] for r in _read_records(path)] == [1]


# --- INV-I13a: os.replace failure is fail-soft, original intact --------------


@pytest.mark.parametrize("exc", [PermissionError("held open (Windows)"), OSError("disk error")])
def test_prune_survives_os_replace_raising_and_leaves_the_original_intact(
    tmp_path, monkeypatch, caplog, exc
):
    path = rd.diagnostics_path(tmp_path)
    original = _write(
        path, [_rec_line(NOW - timedelta(days=9), 0), _rec_line(NOW - timedelta(hours=1), 1)]
    )

    def boom(src, dst):
        raise exc

    monkeypatch.setattr(os, "replace", boom)
    with caplog.at_level(logging.ERROR, logger=rd.logger.name):
        assert rd.prune(path, window_days=2.0, now=NOW) == 0
    assert path.read_bytes() == original
    assert not (tmp_path / "recall_diagnostics.log.jsonl.tmp").exists()
    assert any("prune" in r.getMessage() and r.exc_info for r in caplog.records)
    # the lock was released: a later append succeeds
    monkeypatch.undo()
    assert rd.log_recall(path, **_kwargs()) is True
    assert len(_read_records(path)) == 3


# --- CONC-3: an append inside the prune's read -> rewrite window ------------


def _append_inside_prune_window(monkeypatch, path: Path, point: str, *, settle_s: float) -> dict:
    """Arrange for a concurrent append to be launched from inside a prune, at `point`:
    "after_read" (just after the prune has read the file, before it filters and
    rewrites) or "at_replace" (just before `os.replace`). The append thread gets
    `settle_s` to finish before the prune continues; `state["done_in_window"]` says
    whether it did (a lock that covers the window makes that False)."""
    real_replace = os.replace
    real_record_ts = rd._record_ts
    state: dict = {"launched": False, "done_in_window": None, "thread": None}
    done = threading.Event()

    def body() -> None:
        rd.append_record(path, _fresh_record(999))
        done.set()

    def launch() -> None:
        if state["launched"]:
            return
        state["launched"] = True
        t = threading.Thread(target=body, daemon=True)
        state["thread"] = t
        t.start()
        state["done_in_window"] = done.wait(timeout=settle_s)

    def record_ts_after_read(raw_line):
        launch()
        return real_record_ts(raw_line)

    def replace_after_launch(src, dst):
        launch()
        real_replace(src, dst)

    if point == "after_read":
        monkeypatch.setattr(rd, "_record_ts", record_ts_after_read)
    else:
        monkeypatch.setattr(os, "replace", replace_after_launch)  # the rewrite replaces through os.replace (json_io)
    return state


@pytest.mark.parametrize("point", ["after_read", "at_replace"])
def test_append_attempted_inside_the_prune_window_does_not_wait_and_lands_after_the_rewrite(
    tmp_path, monkeypatch, point
):
    path = rd.diagnostics_path(tmp_path)
    _write(path, [_rec_line(NOW - timedelta(days=9), 0), _rec_line(NOW - timedelta(hours=1), 1)])
    state = _append_inside_prune_window(monkeypatch, path, point, settle_s=0.5)
    assert rd.prune(path, window_days=2.0, now=NOW) == 1
    monkeypatch.undo()
    state["thread"].join(timeout=10)
    assert not state["thread"].is_alive()
    assert state["done_in_window"] is True  # S189: it did not wait for the prune's lock
    assert [r["n"] for r in _read_records(path)] == [1, 999]  # not lost, and after the rewrite


@pytest.mark.parametrize("point", ["after_read", "at_replace"])
def test_the_lock_is_what_protects_the_append_a_lockless_prune_loses_it(
    tmp_path, monkeypatch, point
):
    """Able-to-fail check for CONC-3: with the lock replaced by a no-op the same
    interleaving loses the concurrent append (the wait is on the thread's own
    completion event, so the outcome does not depend on a wall-clock guess)."""

    @contextlib.contextmanager
    def no_lock(path, *, blocking=True):
        yield True

    monkeypatch.setattr(rd, "file_lock", no_lock)
    path = rd.diagnostics_path(tmp_path)
    _write(path, [_rec_line(NOW - timedelta(days=9), 0), _rec_line(NOW - timedelta(hours=1), 1)])
    state = _append_inside_prune_window(monkeypatch, path, point, settle_s=30.0)
    assert rd.prune(path, window_days=2.0, now=NOW) == 1
    monkeypatch.undo()
    state["thread"].join(timeout=10)
    assert state["done_in_window"] is True  # not blocked
    assert [r["n"] for r in _read_records(path)] == [1]  # the append (999) was overwritten


# --- the append never waits on the reply path (S189) -------------------------


def _run_with_timeout(fn, timeout: float = 5.0):
    """Run ``fn`` in a thread; fail the test (rather than hang it) if it waits."""
    box: dict = {}

    def body() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(timeout=timeout)
    assert not t.is_alive(), "the call waited on the held lock"
    assert "error" not in box, box.get("error")
    return box["value"]


def _fresh_record(n: int, *, age: timedelta = timedelta(0)) -> dict:
    return {"ts": (datetime.now(UTC) - age).isoformat(), "source": "passive", "n": n}


def test_append_with_the_lock_held_returns_at_once_and_writes_nothing_yet(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    with rd.file_lock(path):
        assert _run_with_timeout(lambda: rd.append_record(path, _fresh_record(1))) is True
        assert not path.exists() or _read_records(path) == []


def test_the_next_append_writes_the_buffered_records_first_in_order(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    with rd.file_lock(path):
        assert _run_with_timeout(lambda: rd.append_record(path, _fresh_record(1)))
        assert _run_with_timeout(lambda: rd.append_record(path, _fresh_record(2)))
    assert rd.append_record(path, _fresh_record(3)) is True
    assert [r["n"] for r in _read_records(path)] == [1, 2, 3]
    assert rd.append_record(path, _fresh_record(4)) is True
    assert [r["n"] for r in _read_records(path)] == [1, 2, 3, 4]  # written once, not repeated


def test_the_lull_flush_writes_the_buffer_when_the_lock_is_free(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    with rd.file_lock(path):
        assert _run_with_timeout(lambda: rd.append_record(path, _fresh_record(1)))
        assert _run_with_timeout(lambda: rd.flush_pending(path)) == 0  # still held: no wait, no write
    assert rd.flush_pending(path) == 1
    assert [r["n"] for r in _read_records(path)] == [1]
    assert rd.flush_pending(path) == 0  # nothing buffered any more


def test_flush_with_nothing_buffered_creates_no_lock_file(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    assert rd.flush_pending(path) == 0
    assert list(tmp_path.iterdir()) == []


def test_a_buffered_record_older_than_the_window_is_dropped_not_written(tmp_path, monkeypatch):
    path = rd.diagnostics_path(tmp_path)
    monkeypatch.setattr(rd, "_default_window_days", lambda: 2.0)
    with rd.file_lock(path):
        assert _run_with_timeout(
            lambda: rd.append_record(path, _fresh_record(1, age=timedelta(days=3)))
        )
        assert _run_with_timeout(lambda: rd.append_record(path, _fresh_record(2)))
    assert rd.append_record(path, _fresh_record(3)) is True
    assert [r["n"] for r in _read_records(path)] == [2, 3]  # the 3-day-old one was dropped


def test_a_record_that_goes_stale_while_buffered_is_dropped_at_the_flush(tmp_path, monkeypatch):
    path = rd.diagnostics_path(tmp_path)
    monkeypatch.setattr(rd, "_default_window_days", lambda: 2.0)
    with rd.file_lock(path):
        # fresh when buffered (1 day old), stale by the time the lock is free (window shrinks)
        assert _run_with_timeout(
            lambda: rd.append_record(path, _fresh_record(1, age=timedelta(days=1)))
        )
    monkeypatch.setattr(rd, "_default_window_days", lambda: 0.5)
    assert rd.flush_pending(path) == 0
    assert not path.exists() or _read_records(path) == []


def test_prune_flushes_the_records_buffered_while_it_held_the_lock(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    _write(path, [_rec_line(NOW - timedelta(days=9), 0), _rec_line(NOW - timedelta(hours=1), 1)])
    with rd.file_lock(path):
        assert _run_with_timeout(lambda: rd.append_record(path, _fresh_record(7)))
    assert rd.prune(path, window_days=2.0, now=NOW) == 1
    assert [r["n"] for r in _read_records(path)] == [1, 7]  # the pruned file, then the buffered one


def test_buffered_records_are_in_memory_only(tmp_path):
    """A crash with the lock still held loses the buffered record (accepted): nothing
    of it is on disk, and nothing but the buffer holds it."""
    path = rd.diagnostics_path(tmp_path)
    with rd.file_lock(path):
        assert _run_with_timeout(lambda: rd.append_record(path, _fresh_record(1)))
        assert [p.name for p in tmp_path.iterdir()] == [path.name + ".lock"]
        assert len(rd._pending[str(path)]) == 1


def test_a_failing_write_is_still_fail_soft_and_drops_the_record(tmp_path, monkeypatch, caplog):
    path = rd.diagnostics_path(tmp_path)

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(rd, "_write_lines", boom)
    with caplog.at_level(logging.WARNING, logger=rd.logger.name):
        assert rd.append_record(path, _fresh_record(1)) is False
    assert "record dropped" in caplog.text
    assert str(path) not in rd._pending


def test_a_stale_record_is_not_kept_in_the_buffer_even_before_any_flush(tmp_path, monkeypatch):
    """The window bounds the buffer itself (S189), not only what a flush writes."""
    path = rd.diagnostics_path(tmp_path)
    monkeypatch.setattr(rd, "_default_window_days", lambda: 2.0)
    with rd.file_lock(path):
        assert _run_with_timeout(
            lambda: rd.append_record(path, _fresh_record(1, age=timedelta(days=3)))
        )
        assert not rd.has_pending(path)
        assert _run_with_timeout(lambda: rd.append_record(path, _fresh_record(2)))
        assert len(rd._pending[str(path)]) == 1


def test_a_lock_released_between_the_failed_try_and_the_buffering_does_not_strand_the_record(
    tmp_path, monkeypatch
):
    """One more non-blocking try after buffering: if the holder let go in the gap, the
    record is written now, not left waiting for an append that may never come."""
    path = rd.diagnostics_path(tmp_path)
    real = rd.file_lock
    calls = {"n": 0}

    @contextlib.contextmanager
    def first_try_reports_held(p, *, blocking=True):
        calls["n"] += 1
        if calls["n"] == 1:
            yield False
        else:
            with real(p, blocking=blocking) as acquired:
                yield acquired

    monkeypatch.setattr(rd, "file_lock", first_try_reports_held)
    assert rd.append_record(path, _fresh_record(1)) is True
    assert [r["n"] for r in _read_records(path)] == [1]
    assert not rd.has_pending(path)


def test_has_pending_is_per_path(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    assert rd.has_pending(path) is False
    with rd.file_lock(path):
        assert _run_with_timeout(lambda: rd.append_record(path, _fresh_record(1)))
        assert rd.has_pending(path) is True
        assert rd.has_pending(rd.diagnostics_path(tmp_path / "other")) is False
