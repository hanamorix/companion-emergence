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


def test_append_keeps_non_ascii_text_as_utf8(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    assert rd.append_record(path, {"ts": NOW.isoformat(), "scale": "échelle"}) is True
    assert _read_records(path)[0]["scale"] == "échelle"


def test_concurrent_appends_all_land_as_intact_lines(tmp_path):
    path = rd.diagnostics_path(tmp_path)
    errors: list[BaseException] = []

    def worker(k: int) -> None:
        try:
            for i in range(10):
                assert rd.log_recall(path, **_kwargs(paragraph_count=k * 100 + i))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    recs = _read_records(path)
    assert sorted(r["paragraph_count"] for r in recs) == sorted(
        k * 100 + i for k in range(4) for i in range(10)
    )


# --- fail-soft (never raises into recall) -----------------------------------


def test_log_recall_never_raises_when_the_path_cannot_be_opened(tmp_path, caplog):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    bad = blocker / "recall_diagnostics.log.jsonl"  # parent is a regular file
    with caplog.at_level(logging.ERROR, logger=rd.logger.name):
        assert rd.log_recall(bad, **_kwargs()) is False
    assert any("append" in r.getMessage() and r.exc_info for r in caplog.records)


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
    assert rd.append_record(path, {"ts": NOW.isoformat(), "x": float("nan")}) is False
    assert not path.exists() or path.read_bytes() == b""


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
    monkeypatch.setattr(rd.os, "replace", lambda *a: pytest.fail("rewrote an unchanged file"))
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

    monkeypatch.setattr(rd.os, "replace", boom)
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


def _run_append_inside_prune_window(path: Path) -> dict:
    """Run a prune whose `os.replace` first launches a concurrent append and
    gives it time to run; return what the append thread saw."""
    real_replace = os.replace
    state: dict = {"finished_before_replace": None}
    appended = {"n": 999}

    def append_thread_body() -> None:
        rd.append_record(path, {"ts": NOW.isoformat(), "source": "passive", **appended})

    def replace_with_concurrent_append(src, dst):
        t = threading.Thread(target=append_thread_body, daemon=True)
        t.start()
        t.join(timeout=0.5)  # window: the append thread gets a fair chance to write
        state["finished_before_replace"] = not t.is_alive()
        state["thread"] = t
        real_replace(src, dst)

    return {"state": state, "hook": replace_with_concurrent_append}


def test_append_attempted_inside_the_prune_window_blocks_and_lands_after_the_rewrite(
    tmp_path, monkeypatch
):
    path = rd.diagnostics_path(tmp_path)
    _write(path, [_rec_line(NOW - timedelta(days=9), 0), _rec_line(NOW - timedelta(hours=1), 1)])
    seam = _run_append_inside_prune_window(path)
    monkeypatch.setattr(rd.os, "replace", seam["hook"])
    assert rd.prune(path, window_days=2.0, now=NOW) == 1
    monkeypatch.undo()
    seam["state"]["thread"].join(timeout=10)
    assert not seam["state"]["thread"].is_alive()
    assert seam["state"]["finished_before_replace"] is False  # it was blocked by the prune's lock
    assert [r["n"] for r in _read_records(path)] == [1, 999]  # not lost, and after the rewrite


def test_the_lock_is_what_protects_the_append_a_lockless_prune_loses_it(tmp_path, monkeypatch):
    """Able-to-fail check for CONC-3: with the lock replaced by a no-op the same
    interleaving loses the concurrent append."""

    @contextlib.contextmanager
    def no_lock(path, *, blocking=True):
        yield True

    monkeypatch.setattr(rd, "file_lock", no_lock)
    path = rd.diagnostics_path(tmp_path)
    _write(path, [_rec_line(NOW - timedelta(days=9), 0), _rec_line(NOW - timedelta(hours=1), 1)])
    seam = _run_append_inside_prune_window(path)
    monkeypatch.setattr(rd.os, "replace", seam["hook"])
    assert rd.prune(path, window_days=2.0, now=NOW) == 1
    monkeypatch.undo()
    seam["state"]["thread"].join(timeout=10)
    assert seam["state"]["finished_before_replace"] is True  # not blocked
    assert [r["n"] for r in _read_records(path)] == [1]  # the append (999) was overwritten
