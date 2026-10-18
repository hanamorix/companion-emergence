"""Group C of the json_io phase-2 migration: modules with JSONL readers and rewrites.

Main's rule: a migration never WIDENS what a caller swallows. Every reader whose site did not catch an I/O
fault before still raises on one (a directory in place of the file is an I/O fault on every platform), and
a reader that feeds a rewrite, a truncate or a persisted value leaves that file byte-identical when the read
fails. Readers whose site swallowed the fault still return their fallback. Each test calls the REAL function
on a ``tmp_path``. Nothing here touches a persona directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.health.json_io import JsonFileError


def _dir_at(path: Path) -> Path:
    """An I/O fault on any platform: a directory where the file should be."""
    path.mkdir(parents=True)
    return path


# ---------------------------------------------------------------------------------------------
# Readers that propagate an I/O fault (RAISE): it still reaches the caller
# ---------------------------------------------------------------------------------------------

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _pd(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


RAISERS = {
    "attunement.store.read_learned_patterns": (
        "attunement/learned_patterns.jsonl",
        lambda pd: __import__("brain.attunement.store", fromlist=["x"]).read_learned_patterns(pd),
    ),
    "behavioral.log.read_behavioral_log": (
        "behavior.jsonl",
        lambda pd: __import__("brain.behavioral.log", fromlist=["x"]).read_behavioral_log(pd / "behavior.jsonl"),
    ),
    "forgetting.graveyard.search": (
        "forgotten_memories.jsonl",
        lambda pd: __import__("brain.forgetting.graveyard", fromlist=["x"]).search(pd, "dream memories", limit=3),
    ),
    "forgetting.graveyard.read_all": (
        "forgotten_memories.jsonl",
        lambda pd: __import__("brain.forgetting.graveyard", fromlist=["x"]).read_all(pd),
    ),
    "growth.arc_storage.read_removed_arcs": (
        "removed_arcs.jsonl",
        lambda pd: __import__("brain.growth.arc_storage", fromlist=["x"]).read_removed_arcs(pd),
    ),
    "growth.log.read_growth_log": (
        "growth.jsonl",
        lambda pd: __import__("brain.growth.log", fromlist=["x"]).read_growth_log(pd / "growth.jsonl"),
    ),
    "health.adaptive.compute_treatment": (
        "heartbeats.log.jsonl",
        lambda pd: __import__("brain.health.adaptive", fromlist=["x"]).compute_treatment(pd, "f.json"),
    ),
    "health.alarm.compute_pending_alarms": (
        "heartbeats.log.jsonl",
        lambda pd: __import__("brain.health.alarm", fromlist=["x"]).compute_pending_alarms(pd),
    ),
    "ingest.buffer.read_session": (
        "active_conversations/s1.jsonl",
        lambda pd: __import__("brain.ingest.buffer", fromlist=["x"]).read_session(pd, "s1"),
    ),
    "ingest.soul_queue.list_soul_candidates": (
        "soul_candidates.jsonl",
        lambda pd: __import__("brain.ingest.soul_queue", fromlist=["x"]).list_soul_candidates(pd),
    ),
    "initiate.emit.read_candidates": (
        "initiate_candidates.jsonl",
        lambda pd: __import__("brain.initiate.emit", fromlist=["x"]).read_candidates(pd),
    ),
    "initiate.audit.read_recent_audit": (
        "initiate_audit.jsonl",
        lambda pd: list(
            __import__("brain.initiate.audit", fromlist=["x"]).read_recent_audit(pd, window_hours=24)
        ),
    ),
    "initiate.audit.read_recent_d_calls": (
        "initiate_d_calls.jsonl",
        lambda pd: list(
            __import__("brain.initiate.audit", fromlist=["x"]).read_recent_d_calls(pd, window_hours=24)
        ),
    ),
    "initiate.audit.iter_initiate_audit_full": (
        "initiate_audit.jsonl",
        lambda pd: list(__import__("brain.initiate.audit", fromlist=["x"]).iter_initiate_audit_full(pd)),
    ),
    "initiate.adaptive.read_recent_calibration_rows": (
        "d_calibration.jsonl",
        lambda pd: list(
            __import__("brain.initiate.adaptive", fromlist=["x"]).read_recent_calibration_rows(pd, limit=5)
        ),
    ),
    "initiate.adaptive._read_all_d_calls": (
        "initiate_d_calls.jsonl",
        lambda pd: __import__("brain.initiate.adaptive", fromlist=["x"])._read_all_d_calls(pd),
    ),
    "initiate.user_pattern._read_valid_reply_lags": (
        "initiate_audit.jsonl",
        lambda pd: __import__("brain.initiate.user_pattern", fromlist=["x"])._read_valid_reply_lags(pd),
    ),
    "initiate.user_pattern._compute_silence_days": (
        "active_conversations/s1.jsonl",
        lambda pd: __import__("brain.initiate.user_pattern", fromlist=["x"])._compute_silence_days(pd),
    ),
    "initiate.user_pattern._build_hour_histogram": (
        "active_conversations/s1.jsonl",
        lambda pd: __import__("brain.initiate.user_pattern", fromlist=["x"])._build_hour_histogram(pd),
    ),
    "initiate.user_pattern._daily_scan_conversations": (
        "active_conversations/s1.jsonl",
        lambda pd: __import__("brain.initiate.user_pattern", fromlist=["x"])._daily_scan_conversations(pd),
    ),
    "attunement.backfill._read_buffer_turns": (
        "active_conversations/s1.jsonl",
        lambda pd: __import__("brain.attunement.backfill", fromlist=["x"])._read_buffer_turns(pd),
    ),
    "kindled_link.feed_source.peer_key_rotated_entries": (
        "kindled_link/transport.jsonl",
        lambda pd: __import__("brain.kindled_link.feed_source", fromlist=["x"]).peer_key_rotated_entries(pd),
    ),
    "memory.pending.read_recent": (
        "pending_candidates.jsonl",
        lambda pd: __import__("brain.memory.pending", fromlist=["x"]).PendingQueue(pd).read_recent("conversation", limit=3),
    ),
    "narrative_memory.state._replay_from_log": (
        "arcs.log.jsonl",
        lambda pd: __import__("brain.narrative_memory.state", fromlist=["x"])._replay_from_log(pd / "arcs.log.jsonl"),
    ),
    "narrative_memory.state._log_newer_than_state": (
        "arcs.log.jsonl",
        lambda pd: (lambda m: m._log_newer_than_state(pd / "arcs.log.jsonl", m.ArcsState(last_pass_ts_iso="2026-01-01T00:00:00Z")))(
            __import__("brain.narrative_memory.state", fromlist=["x"])
        ),
    ),
    "narrative_memory.tool._reconstruct_arcs_from_log": (
        "arcs.log.jsonl",
        lambda pd: __import__("brain.narrative_memory.tool", fromlist=["x"])._reconstruct_arcs_from_log(pd / "arcs.log.jsonl"),
    ),
    "soul.audit.read_audit_log": (
        "soul_audit.jsonl",
        lambda pd: __import__("brain.soul.audit", fromlist=["x"]).read_audit_log(pd),
    ),
    "soul.audit.iter_audit_full": (
        "soul_audit.jsonl",
        lambda pd: list(__import__("brain.soul.audit", fromlist=["x"]).iter_audit_full(pd)),
    ),
    "soul.candidates.list_under_review_memory_ids": (
        "soul_candidates.jsonl",
        lambda pd: __import__("brain.soul.candidates", fromlist=["x"]).list_under_review_memory_ids(pd),
    ),
    "soul.candidates._accepted_candidate_ids": (
        "soul_candidates.jsonl",
        lambda pd: __import__("brain.soul.candidates", fromlist=["x"])._accepted_candidate_ids(pd, None),
    ),
    "soul.review._load_soul_candidates": (
        "soul_candidates.jsonl",
        lambda pd: __import__("brain.soul.review", fromlist=["x"])._load_soul_candidates(pd),
    ),
}


@pytest.mark.parametrize("name", sorted(RAISERS))
def test_group_c_reader_that_propagated_an_io_fault_still_raises(tmp_path: Path, name: str) -> None:
    rel, call = RAISERS[name]
    pd = _pd(tmp_path / "persona")
    _dir_at(pd / rel)
    with pytest.raises(OSError) as exc:
        call(pd)
    assert isinstance(exc.value, JsonFileError) and exc.value.kind == "io"


@pytest.mark.parametrize("name", sorted(RAISERS))
def test_group_c_reader_returns_nothing_for_a_missing_file(tmp_path: Path, name: str) -> None:
    rel, call = RAISERS[name]
    pd = _pd(tmp_path / "persona")
    (pd / rel).parent.mkdir(parents=True, exist_ok=True)
    call(pd)  # no file: an empty result, not an error (it must not raise)


def test_group_c_server_read_jsonl_lines_keeps_its_is_file_guard_and_raises_on_io(tmp_path: Path) -> None:
    from brain.bridge.server import _read_jsonl_lines

    d = _dir_at(tmp_path / "x.jsonl")
    assert list(_read_jsonl_lines(d)) == []  # a directory was never a file: unchanged
    f = tmp_path / "ok.jsonl"
    f.write_bytes(b'{"a": 1}\nnot json\n{"b": 2}\n')
    assert list(_read_jsonl_lines(f)) == [{"a": 1}, {"b": 2}]


def test_group_c_dream_recent_seed_ids_and_log(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from brain.engines.dream import DreamEngine

    log = tmp_path / "dreams.log.jsonl"
    fake = SimpleNamespace(log_path=log, provider=SimpleNamespace(name=lambda: "p"))
    seed = SimpleNamespace(id="s1")
    mem = SimpleNamespace(id="d1")
    DreamEngine._log(fake, seed, [(SimpleNamespace(id="n1"), 0.5)], mem)
    DreamEngine._log(fake, SimpleNamespace(id="s2"), [], mem)
    assert [json.loads(line)["seed_id"] for line in log.read_text().splitlines()] == ["s1", "s2"]
    assert DreamEngine._recent_seed_ids(fake, 5) == ["s1", "s2"]
    assert DreamEngine._recent_seed_ids(fake, 1) == ["s2"]
    log.unlink()
    log.mkdir()
    with pytest.raises(OSError):
        DreamEngine._recent_seed_ids(fake, 5)


# ---------------------------------------------------------------------------------------------
# Readers whose site swallowed the fault (DEFAULT): they still return their fallback
# ---------------------------------------------------------------------------------------------


def test_group_c_default_readers_fall_back_on_an_io_fault(tmp_path: Path) -> None:
    from brain.bridge.feed import _read_initiate_audit
    from brain.chat.prompt import _count_soul_candidates
    from brain.health.parse_failures import count_recent
    from brain.health.soul_candidate_repair import should_run_soul_candidate_repair
    from brain.kindled_link.views import relay_health
    from brain.utils.memory import _latest_user_turn_in_buffers

    pd = tmp_path / "persona"
    _dir_at(pd / "initiate_audit.jsonl")
    _dir_at(pd / "soul_candidates.jsonl")
    _dir_at(pd / "kindled_link" / "transport.jsonl")
    _dir_at(pd / "parse_failures.jsonl")
    _dir_at(pd / "active_conversations" / "s.jsonl")
    assert _read_initiate_audit(pd) == []
    assert _count_soul_candidates(pd) == 0
    assert should_run_soul_candidate_repair(pd) is False
    assert relay_health(pd) == {"relay_ok": None, "last_poll_ts": None, "last_push_ts": None, "degraded_peers": []}
    assert _latest_user_turn_in_buffers(pd) is None
    assert count_recent(pd) == {}


def test_group_c_feed_monologue_and_count_words_survive_bad_lines(tmp_path: Path) -> None:
    from brain.body.words import count_words_in_session
    from brain.bridge.feed import build_monologue_entries

    pd = tmp_path / "persona"
    pd.mkdir()
    (pd / "monologue_digest.jsonl").write_bytes(
        b'{"ts": "2026-01-01T00:00:00Z", "digest": "a b"}\nnot json\n[1]\n{"digest": "no ts"}\n'
        b'{"ts": "2026-01-02T00:00:00Z", "digest": "c", "surfaced": false}\n'
    )
    got = build_monologue_entries(pd, limit=10)
    assert [e.body for e in got] == ["a b"]  # the withheld digest stays withheld
    (pd / "active_conversations").mkdir()
    (pd / "active_conversations" / "s.jsonl").write_bytes(
        f'{{"speaker": "assistant", "ts": "{NOW.isoformat()}", "text": "one two three"}}\n'.encode()
        + b"garbage\n"
    )
    assert count_words_in_session(None, persona_dir=pd, session_hours=24, now=NOW + timedelta(hours=1)) == 3


# ---------------------------------------------------------------------------------------------
# Readers that feed a rewrite, a truncate or a persisted value: a failed read leaves the file alone
# ---------------------------------------------------------------------------------------------


def test_group_c_pending_drain_roundtrip_and_an_unreadable_queue_is_not_drained(tmp_path: Path) -> None:
    from brain.memory.pending import PendingQueue

    q = PendingQueue(tmp_path)
    q.enqueue_reappraisals(["m1", "m2"], source="t")
    q.enqueue_reappraisals(["m3"], source="t")
    rows = q.drain()
    assert [r["memory_id"] for r in rows] == ["m1", "m2", "m3"]
    assert q.path.read_bytes() == b""  # emptied
    assert q.drain() == []
    # An unreadable queue: drain raises and truncates nothing.
    other = tmp_path / "other"
    other.mkdir()
    q2 = PendingQueue(other)
    q2.path.mkdir()
    marker = q2.path / "keep"
    marker.write_text("x")
    with pytest.raises(OSError):
        q2.drain()
    assert marker.read_text() == "x"


def test_group_c_soul_review_rewrite_roundtrip(tmp_path: Path) -> None:
    from brain.soul.review import _load_soul_candidates, _save_soul_candidates

    recs = [{"memory_id": "m1", "status": "pending", "text": "caf\u00e9"}, {"memory_id": "m2", "status": "accepted"}]
    _save_soul_candidates(tmp_path, recs)
    raw = (tmp_path / "soul_candidates.jsonl").read_bytes()
    assert raw.isascii() and _load_soul_candidates(tmp_path) == recs
    _save_soul_candidates(tmp_path, [])
    assert _load_soul_candidates(tmp_path) == []
    assert not [p for p in tmp_path.iterdir() if ".tmp" in p.name or p.name.endswith(".new")]


def test_group_c_soul_candidate_repair_save_and_repair_locked(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from brain.health import soul_candidate_repair as scr

    path = tmp_path / scr._CANDIDATES_FILE
    scr._save_candidates(path, [{"memory_id": "m1", "text": "t", "status": scr._TARGET_STATUS, "session_id": scr._TARGET_SESSION_ID}])
    assert json.loads(path.read_text())["memory_id"] == "m1"
    store = SimpleNamespace(get=lambda mid: SimpleNamespace(content="body text"))
    report = scr._repair_locked(tmp_path, path, store=store)
    assert report.repaired == 1 and "body text" in json.loads(path.read_text())["text"]
    # An unreadable candidates file: skipped through the site's own handler, and left untouched.
    other = tmp_path / "o"
    other.mkdir()
    bad = other / scr._CANDIDATES_FILE
    bad.mkdir()
    report = scr._repair_locked(other, bad, store=store)
    assert report.repaired == 0 and bad.is_dir()


def test_group_c_emit_remove_candidate_rewrites_only_the_target_and_not_on_a_failed_read(tmp_path: Path) -> None:
    from brain.initiate.emit import emit_initiate_candidate, read_candidates, remove_candidate
    from brain.initiate.schemas import SemanticContext

    for i in range(3):
        emit_initiate_candidate(
            tmp_path, kind="message", source="dream", source_id=f"s{i}",
            semantic_context=SemanticContext(topic_tags=["caf\u00e9"]), now=NOW + timedelta(seconds=i),
        )
    ids = [c.candidate_id for c in read_candidates(tmp_path)]
    assert len(ids) == 3
    remove_candidate(tmp_path, ids[1])
    assert [c.candidate_id for c in read_candidates(tmp_path)] == [ids[0], ids[2]]
    assert (tmp_path / "initiate_candidates.jsonl").read_bytes().isascii()
    # emit is idempotent on (kind, source, source_id)
    emit_initiate_candidate(tmp_path, kind="message", source="dream", source_id="s0", semantic_context=SemanticContext())
    assert len(read_candidates(tmp_path)) == 2
    other = tmp_path / "o"
    other.mkdir()
    (other / "initiate_candidates.jsonl").mkdir()
    with pytest.raises(OSError):
        remove_candidate(other, "x")
    assert (other / "initiate_candidates.jsonl").is_dir()


def test_group_c_update_audit_state_rewrites_under_the_lock_and_fails_closed(tmp_path: Path) -> None:
    from brain.initiate.audit import append_audit_row, read_recent_audit, update_audit_state
    from brain.initiate.schemas import AuditRow

    def row(i: str) -> AuditRow:
        return AuditRow(
            audit_id=i, candidate_id="c" + i, ts=datetime.now(UTC).isoformat(), kind="message", subject="s",
            tone_rendered="t", decision="send_quiet", decision_reasoning="r", gate_check={"allowed": True},
        )

    append_audit_row(tmp_path, row("a1"))
    append_audit_row(tmp_path, row("a2"))
    update_audit_state(tmp_path, audit_id="a2", new_state="delivered", at=datetime.now(UTC).isoformat())
    got = {r.audit_id: r for r in read_recent_audit(tmp_path, window_hours=1)}
    assert got["a2"].delivery["current_state"] == "delivered" and got["a1"].delivery is None
    other = tmp_path / "o"
    other.mkdir()
    (other / "initiate_audit.jsonl").mkdir()
    with pytest.raises(OSError):
        update_audit_state(other, audit_id="a", new_state="delivered", at="2026-01-01T00:00:00+00:00")
    assert (other / "initiate_audit.jsonl").is_dir()


def test_group_c_log_rotation_keeps_bytes_and_leaves_the_log_alone_on_a_failed_read(tmp_path: Path) -> None:
    import gzip

    from brain.health.log_rotation import rotate_age_archive_yearly

    log = tmp_path / "soul_audit.jsonl"
    old = '{"ts": "2024-03-01T00:00:00+00:00", "x": "caf\u00e9"}'
    new = '{"ts": "2026-03-01T00:00:00+00:00", "x": 1}'
    log.write_bytes((old + "\nnot json\n" + new + "\n").encode("utf-8"))
    out = rotate_age_archive_yearly(log, now=NOW, timestamp_field="ts")
    assert [p.name for p in out] == ["soul_audit.2024.jsonl.gz"]
    assert gzip.open(out[0], "rt", encoding="utf-8").read() == old + "\n"  # the raw line, not re-serialised
    assert log.read_text(encoding="utf-8") == new + "\n"  # corrupt line dropped as before
    other = tmp_path / "o"
    other.mkdir()
    bad = other / "soul_audit.jsonl"
    bad.mkdir()
    with pytest.raises(OSError):
        rotate_age_archive_yearly(bad, now=NOW, timestamp_field="ts")
    assert bad.is_dir()


def test_group_c_recall_prune_counts_and_removes_unageable_lines_and_leaves_a_failed_read(tmp_path: Path, caplog) -> None:
    from brain.memory import recall_diagnostics as rd

    path = tmp_path / "recall.jsonl"
    old = json.dumps({"ts": (NOW - timedelta(days=400)).isoformat()})
    fresh = json.dumps({"ts": NOW.isoformat(), "n": 1})
    path.write_bytes(("\n".join([old, "not json", "[1]", fresh, ""])).encode())
    with caplog.at_level(logging.WARNING):
        removed = rd.prune(path, window_days=30, now=NOW)
    assert removed == 3  # one aged out, two cannot be aged
    assert path.read_text() == fresh + "\n"
    assert "no readable ts" in caplog.text
    assert rd.prune(path, window_days=30, now=NOW) == 0  # nothing removed: no rewrite
    only_good = path.read_bytes()
    assert rd.prune(path, window_days=30, now=NOW) == 0 and path.read_bytes() == only_good
    d = tmp_path / "dir.jsonl"
    d.mkdir()
    assert rd.prune(d, window_days=30, now=NOW) == 0 and d.is_dir()  # never raises, leaves the "file" alone


def test_group_c_recall_append_record_appends_ascii_under_the_held_lock(tmp_path: Path) -> None:
    from brain.memory import recall_diagnostics as rd

    path = tmp_path / "recall.jsonl"
    assert rd.append_record(path, {"ts": NOW.isoformat(), "x": "caf\u00e9", "f": float("nan")}) is True
    raw = path.read_bytes()
    assert raw.isascii() and json.loads(raw) == {"ts": NOW.isoformat(), "x": "caf\u00e9", "f": None}


def test_group_c_session_buffer_append_rewrite_and_archive(tmp_path: Path) -> None:
    from brain.ingest import buffer as buf

    sid = "sess-1"
    for i in range(3):
        buf.ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": f"hi \u00e9 {i}", "ts": NOW.isoformat()})
    assert [t["text"] for t in buf.read_session(tmp_path, sid)] == ["hi \u00e9 0", "hi \u00e9 1", "hi \u00e9 2"]
    path = tmp_path / "active_conversations" / f"{sid}.jsonl"
    assert path.read_bytes().isascii()
    buf.rewrite_session_atomic(tmp_path, sid, [{"speaker": "summary", "text": " keep  verbatim ", "compaction": {"k": 1}}])
    assert buf.read_session(tmp_path, sid) == [{"speaker": "summary", "text": " keep  verbatim ", "compaction": {"k": 1}}]
    n = buf.append_archive(tmp_path, sid, [{"a": 1}, {"b": "\u00e9"}])
    assert n == len(b'{"a": 1}\n{"b": "\\u00e9"}\n')
    assert buf.read_archive(tmp_path, sid) == [{"a": 1}, {"b": "\u00e9"}]
    assert not [p for p in path.parent.iterdir() if ".tmp" in p.name]


def test_group_c_soul_queue_candidate_roundtrip(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from brain.ingest.soul_queue import list_soul_candidates, queue_soul_candidate

    item = SimpleNamespace(text="t \u00e9", label="l", importance=7)
    assert queue_soul_candidate(tmp_path, memory_id="m1", item=item, session_id="s") is True
    assert queue_soul_candidate(tmp_path, memory_id="m2", item=item, session_id="s") is True
    assert [r["memory_id"] for r in list_soul_candidates(tmp_path)] == ["m1", "m2"]
    blocker = tmp_path / "b"
    blocker.write_text("file")
    assert queue_soul_candidate(blocker, memory_id="m", item=item, session_id="s") is False  # never raises


def test_group_c_narrative_state_event_log_roundtrip_and_replay(tmp_path: Path) -> None:
    from brain.narrative_memory import state as ns

    ns.append_event(tmp_path, {"event": "arc_opened", "arc_id": "a1", "title": "T \u00e9", "ts_iso": NOW.isoformat()})
    ns.append_event(tmp_path, {"event": "member_added", "arc_id": "a1", "memory_id": "m1", "ts_iso": NOW.isoformat()})
    log = tmp_path / ns.LOG_FILENAME
    assert log.read_bytes().isascii()
    replayed = ns._replay_from_log(log)
    assert replayed.replayed and "a1" in replayed.open and replayed.open["a1"].title == "T \u00e9"
    assert ns._log_newer_than_state(log, ns.ArcsState(last_pass_ts_iso=(NOW - timedelta(days=1)).isoformat())) is True


def test_group_c_utils_memory_latest_user_turn(tmp_path: Path) -> None:
    from brain.utils.memory import _latest_user_turn_in_buffers

    d = tmp_path / "active_conversations"
    d.mkdir()
    (d / "a.jsonl").write_bytes(
        b'{"speaker": "user", "ts": "2026-01-01T00:00:00+00:00"}\nbad\n{"speaker": "assistant", "ts": "2026-02-01T00:00:00+00:00"}\n'
    )
    (d / "b.jsonl").write_bytes(b'{"speaker": "user", "ts": "2026-01-05T00:00:00"}\n')
    assert _latest_user_turn_in_buffers(tmp_path) == datetime(2026, 1, 5, tzinfo=UTC)


def test_group_c_attunement_store_rejection_and_pattern_append(tmp_path: Path) -> None:
    from brain.attunement import store

    assert store.read_learned_patterns(tmp_path) == []
    # _log_rejection is exercised with a real candidate by the attunement store tests; here the file shape.
    rej = store._rejections_path(tmp_path)
    store.append_jsonl(rej, {"k": "caf\u00e9"})
    assert rej.read_bytes().isascii()


def test_group_c_health_cli_handlers(tmp_path: Path, monkeypatch, capsys) -> None:
    from brain import cli

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    pd = tmp_path / "personas" / "p"
    pd.mkdir(parents=True)
    ns = argparse.Namespace(persona="p", ack_file=None, all=True, limit=None)
    cli._health_acknowledge_handler(ns)
    rows = [json.loads(line) for line in (pd / "heartbeats.log.jsonl").read_text().splitlines()]
    assert rows and "user_acknowledged" in rows[0]
    assert cli._health_show_handler(argparse.Namespace(persona="p")) == 0
    (pd / "soul_candidates.jsonl").write_bytes(b'{"id": "abcdefghi", "text": "t", "label": "l"}\nnot json\n')
    assert cli._soul_candidates_handler(argparse.Namespace(persona="p", limit=None)) == 0
    assert "abcdefgh" in capsys.readouterr().out
    (pd / "soul_candidates.jsonl").unlink()
    (pd / "soul_candidates.jsonl").mkdir()
    with pytest.raises(OSError):
        cli._soul_candidates_handler(argparse.Namespace(persona="p", limit=None))


def test_group_c_initiate_schemas_to_dict_matches_the_line_codec_including_conditional_keys() -> None:
    from brain.initiate.adaptive import CalibrationRow
    from brain.initiate.d_call_schema import DCallRow
    from brain.initiate.schemas import AuditRow, InitiateCandidate, SemanticContext

    cand = InitiateCandidate(
        candidate_id="c", ts="t", kind="message", source="dream", source_id="s", semantic_context=SemanticContext()
    )
    assert "proposal" not in cand.to_dict()  # conditional key stays omitted
    cand.proposal = {"p": 1}
    assert cand.to_dict()["proposal"] == {"p": 1}
    audit = AuditRow(
        audit_id="a", candidate_id="c", ts="t", kind="message", subject="s", tone_rendered="r", decision="send_quiet",
        decision_reasoning="d", gate_check={},
    )
    assert "diff" not in audit.to_dict() and "user_modified" not in audit.to_dict()
    audit.diff = "x"
    assert audit.to_dict()["diff"] == "x" and "user_modified" in audit.to_dict()
    d = DCallRow(
        d_call_id="d", ts="t", tick_id="k", model_tier_used="haiku", candidates_in=1, promoted_out=0, filtered_out=1,
        latency_ms=5, tokens_input=1, tokens_output=1,
    )
    cal = CalibrationRow(
        ts_decision="a", ts_closed="b", candidate_id="c", source="s", decision="promote", confidence="high",
        model_tier="haiku", promoted_to_state="delivered", filtered_recurred=None, reason_short="r",
    )
    for row in (cand, audit, d, cal):
        assert json.loads(row.to_jsonl()) == row.to_dict()
        assert type(row).from_dict(row.to_dict()) == row == type(row).from_jsonl(row.to_jsonl())
    with pytest.raises(KeyError):
        AuditRow.from_dict({"audit_id": "a"})
