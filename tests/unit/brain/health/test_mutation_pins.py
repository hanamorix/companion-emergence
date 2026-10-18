"""Pins that kill the mutations the stage-6 code reviews found surviving (build record: mutation table).

Each RAISE/CONTENT site is exercised with a read fault that is NOT a directory (a denied open), so a site
demoted to DEFAULT is caught, and a destructive step cannot fail on its own and hide the demotion.
The best-effort/encoder pins call the real writers with a NaN row. Nothing touches a persona directory.
"""

from __future__ import annotations

import ast
import builtins
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.health import json_io as jio

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


@pytest.fixture
def deny(monkeypatch):
    """deny("needle"): any open() by json_io of a path containing the needle raises PermissionError (not a directory)."""
    real_open = builtins.open
    needles: list[str] = []

    def fake_open(file, *a, **k):
        if any(n in str(file) for n in needles):
            raise PermissionError(13, "denied")
        return real_open(file, *a, **k)

    monkeypatch.setattr(jio, "open", fake_open, raising=False)

    def add(needle: str) -> None:
        needles.append(needle)

    return add


# ---------------------------------------------------------------------------------------------
# RAISE and CONTENT readers: a denied (not missing, not a directory) file raises
# ---------------------------------------------------------------------------------------------


def test_mut_pending_drain_does_not_empty_a_queue_it_could_not_read(tmp_path: Path, deny) -> None:
    from brain.memory.pending import PendingQueue

    q = PendingQueue(tmp_path)
    q.enqueue_reappraisals(["m1", "m2"], source="t")
    before = q.path.read_bytes()
    deny("pending_candidates.jsonl")
    with pytest.raises(OSError):
        q.drain()
    assert q.path.read_bytes() == before  # a swallowed read would have emptied it


def test_mut_prune_does_not_shrink_the_log_when_the_read_fails_midway(tmp_path: Path, monkeypatch) -> None:
    from brain.memory import recall_diagnostics as rd

    path = tmp_path / "recall.jsonl"
    old = json.dumps({"ts": (NOW - timedelta(days=400)).isoformat()})
    fresh = [json.dumps({"ts": NOW.isoformat(), "n": i}) for i in range(3)]
    path.write_bytes(("\n".join([old, *fresh]) + "\n").encode())
    before = path.read_bytes()
    real = jio._stream_raw

    def failing(binary, source, strict, bad_lines):
        it = real(binary, source, strict, bad_lines)
        yield next(it)
        yield next(it)
        raise OSError(5, "EIO")

    monkeypatch.setattr(jio, "_stream_raw", failing)
    assert rd.prune(path, window_days=30, now=NOW) == 0  # never raises; leaves the log alone
    assert path.read_bytes() == before


def test_mut_user_preferences_read_raw_keys_raises_on_an_unreadable_file(tmp_path: Path, deny) -> None:
    from brain.user_preferences import read_raw_keys

    p = tmp_path / "user_preferences.json"
    p.write_text('{"dream_every_hours": 3}')
    assert read_raw_keys(p) == {"dream_every_hours"}
    p.write_bytes(b"\xff{")
    assert read_raw_keys(p) == set()  # bad content: the empty set
    deny("user_preferences.json")
    with pytest.raises(OSError):
        read_raw_keys(p)


def test_mut_og_soul_and_reflex_log_and_soul_candidates_raise_on_an_unreadable_source(tmp_path: Path, deny) -> None:
    from brain.migrator.og_reflex_log import migrate_reflex_log
    from brain.migrator.og_soul import extract_crystallizations_from_og
    from brain.migrator.og_soul_candidates import migrate_soul_candidates

    og = tmp_path / "og"
    og.mkdir()
    (og / "nell_soul.json").write_text("{}")
    (og / "nell_reflex_log.json").write_text('{"fires": []}')
    (og / "soul_candidates.jsonl").write_text('{"memory_id": "m"}\n')
    deny("nell_soul.json")
    deny("nell_reflex_log.json")
    deny("soul_candidates.jsonl")
    with pytest.raises(OSError):
        extract_crystallizations_from_og(og)
    with pytest.raises(OSError):
        migrate_reflex_log(og_data_dir=og, persona_dir=tmp_path / "p1")
    with pytest.raises(OSError):
        migrate_soul_candidates(og_data_dir=og, persona_dir=tmp_path / "p2")
    assert not (tmp_path / "p2" / "soul_candidates.jsonl").exists()  # nothing was migrated over the destination


def test_mut_narrative_state_and_current_read_and_backfill_state_raise_when_unreadable(tmp_path: Path, deny) -> None:
    from brain.attunement.store import read_current_read
    from brain.narrative_memory import state as ns

    (tmp_path / "attunement").mkdir()
    (tmp_path / "attunement" / "current_read.json").write_text("{}")
    (tmp_path / ns.STATE_FILENAME).write_text("{}")
    deny("current_read.json")
    deny(ns.STATE_FILENAME)
    with pytest.raises(OSError):
        read_current_read(tmp_path)
    with pytest.raises(OSError):
        ns.load_or_recover(tmp_path)


def test_mut_attunement_backfill_state_raises_when_unreadable(tmp_path: Path, deny) -> None:
    from brain.attunement.backfill import _load_state, _state_path

    p = _state_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")
    deny(p.name)
    with pytest.raises(OSError):
        _load_state(tmp_path)


def test_mut_vocab_repair_raises_when_the_vocabulary_cannot_be_read(tmp_path: Path, deny) -> None:
    from brain.health.vocab_repair import run_vocab_repair, should_run_vocab_repair

    (tmp_path / "emotion_vocabulary.json").write_text('{"emotions": []}')
    deny("emotion_vocabulary.json")
    with pytest.raises(OSError):
        run_vocab_repair(tmp_path, store=SimpleNamespace(), provider=None)
    with pytest.raises(OSError):
        should_run_vocab_repair(tmp_path)
    assert not (tmp_path / "vocab_repair_state.json").exists()  # a failed read does not retire the repair


def test_mut_session_hours_archive_and_session_readers_raise_when_unreadable(tmp_path: Path, deny) -> None:
    from brain.body.session_hours import _entry_timestamps
    from brain.ingest import buffer

    buf = tmp_path / "s.jsonl"
    buf.write_text('{"ts": "2026-01-01T00:00:00+00:00"}\n')
    deny("s.jsonl")
    with pytest.raises(OSError):
        _entry_timestamps(buf)
    buffer.append_archive(tmp_path, "sess", [{"a": 1}])
    deny("sess.000.jsonl")
    with pytest.raises(OSError):
        buffer.read_archive(tmp_path, "sess")


def test_mut_closure_sites_pass_recover_raise() -> None:
    """chat_history (an endpoint closure) and _adapt_anchors (a nested helper) are not callable in isolation: pin
    that their reader call carries recover=Recovery.RAISE, read from the source."""
    for rel, outer in (("brain/bridge/supervisor.py", "_adapt_anchors"), ("brain/bridge/server.py", "chat_history")):
        tree = ast.parse((Path(__file__).resolve().parents[4] / rel).read_text(encoding="utf-8"))
        fn = next(
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == outer
        )
        calls = [
            c
            for c in ast.walk(fn)
            if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "iter_jsonl"
        ]
        assert calls, (rel, outer)
        for c in calls:
            kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
            assert kw.get("recover") == "Recovery.RAISE", (rel, outer, kw)


# ---------------------------------------------------------------------------------------------
# F1: tunables write_defaults_section must not wipe overrides it could not read
# ---------------------------------------------------------------------------------------------


def test_mut_tunables_defaults_refresh_keeps_overrides_when_the_file_is_unreadable(tmp_path: Path, monkeypatch, deny) -> None:
    from brain import tunables

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables._reset_for_tests()
    p = tunables._file_path()
    p.write_text(json.dumps({"overrides": {"a.b": 3}}))
    before = p.read_bytes()
    deny("tunables.json")
    tunables.write_defaults_section()  # never breaks startup; the refresh is skipped
    assert p.read_bytes() == before


def test_mut_tunables_writes_sorted_keys_and_survives_a_hand_edited_nan(tmp_path: Path, monkeypatch) -> None:
    import os

    from brain import tunables

    synced: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables._reset_for_tests()
    p = tunables._file_path()
    p.write_text('{"overrides": {"z.k": NaN, "a.k": 1}}')  # json.loads accepts NaN; a hand edit can contain it
    tunables.write_defaults_section()
    data = json.loads(p.read_text())
    assert data["overrides"] == {"a.k": 1, "z.k": None}
    text = p.read_text()
    assert text == json.dumps(data, indent=2, sort_keys=True) + "\n"  # sorted keys, indent 2, trailing newline
    assert synced == []  # the defaults refresh never synced


# ---------------------------------------------------------------------------------------------
# Best-effort writers: on_error, nonfinite, formats
# ---------------------------------------------------------------------------------------------


def test_mut_best_effort_appenders_do_not_raise_when_the_target_cannot_be_written(tmp_path: Path) -> None:
    from brain.initiate.new_sources import write_gate_rejection
    from brain.notes.write import _audit

    pd = tmp_path / "persona"
    pd.mkdir()
    (pd / "gate_rejections.jsonl").mkdir()  # the append itself fails with an OSError
    (pd / "notes_audit.jsonl").mkdir()
    write_gate_rejection(pd, ts=NOW, source="s", source_id="i", gate_name="g", threshold_value=1.0, observed_value=2.0)
    _audit(pd, path="/n", ok=True)


def test_mut_new_sources_still_raises_when_the_persona_directory_cannot_be_created(tmp_path: Path) -> None:
    """The mkdir sat outside the best-effort try before the migration; it still does (stage-6 AB-F3)."""
    from brain.initiate.new_sources import write_gate_rejection

    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        write_gate_rejection(
            blocker / "persona", ts=NOW, source="s", source_id="i", gate_name="g", threshold_value=1.0, observed_value=2.0
        )


def test_mut_log_transport_writes_a_nan_count_as_null(tmp_path: Path) -> None:
    from brain.kindled_link.audit import log_transport

    log_transport(tmp_path, event="e", count=float("nan"), now=NOW)  # type: ignore[arg-type]
    (row,) = [json.loads(line) for line in (tmp_path / "kindled_link" / "transport.jsonl").read_text().splitlines()]
    assert row["count"] is None


def test_mut_compaction_markers_are_compact_json_with_a_trailing_newline(tmp_path: Path) -> None:
    from brain.chat import compaction_migration as cm

    cm._write_marker_at(tmp_path / "m.json", {"a": 1, "b": [2]})
    assert (tmp_path / "m.json").read_bytes() == b'{"a": 1, "b": [2]}\n'  # indent=None: one line
    cm._write_marker(tmp_path, {"a": 1, "b": [2]})
    assert cm._marker_path(tmp_path).read_bytes() == b'{"a": 1, "b": [2]}\n'


def _mem_nan():
    from brain.memory.store import Memory

    return Memory.create_new(content="x", memory_type="conversation", domain="us", emotions={"love": float("nan")})


def test_mut_archive_and_log_appenders_write_a_nan_as_null_instead_of_raising(tmp_path: Path) -> None:
    from brain.engines import consolidation, dedup_sweep
    from brain.engines.heartbeat import HeartbeatEngine

    consolidation._archive_preimage(tmp_path, _mem_nan(), "s1")
    dedup_sweep._archive_preimage(tmp_path, _mem_nan())
    rows = [json.loads(line) for line in (tmp_path / "consolidation_archive.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(r["target"]["emotions"]["love"] is None for r in rows)
    HeartbeatEngine._append_log(SimpleNamespace(heartbeat_log_path=tmp_path / "hb.jsonl"), {"elapsed_seconds": math.inf})
    assert json.loads((tmp_path / "hb.jsonl").read_text())["elapsed_seconds"] is None


def test_mut_pending_enqueue_of_a_nan_memory_is_queued_not_lost(tmp_path: Path) -> None:
    from brain.memory.pending import PendingQueue

    q = PendingQueue(tmp_path)
    q.enqueue(_mem_nan(), source="t")
    (row,) = q.drain()
    assert row["emotions"]["love"] is None


def test_mut_behavioral_growth_narrative_appenders_write_nan_as_null(tmp_path: Path) -> None:
    from brain.behavioral.log import append_behavioral_event
    from brain.growth.log import GrowthLogEvent, append_growth_event
    from brain.narrative_memory.state import LOG_FILENAME, append_event

    append_behavioral_event(
        tmp_path / "b.jsonl", kind="journal_entry_added", name="n", timestamp=NOW,
        emotional_state={"love": float("nan")},
    )
    assert json.loads((tmp_path / "b.jsonl").read_text())["emotional_state"] == {"love": None}
    append_growth_event(
        tmp_path / "g.jsonl",
        GrowthLogEvent(
            timestamp=NOW, type="emotion_added", name="n", description="d", decay_half_life_days=float("nan"),
            reason="r", evidence_memory_ids=(), score=0.5, relational_context=None,
        ),
    )
    assert json.loads((tmp_path / "g.jsonl").read_text())["decay_half_life_days"] is None
    append_event(tmp_path, {"event": "arc_opened", "lived_age_hours": float("inf")})
    assert json.loads((tmp_path / LOG_FILENAME).read_text())["lived_age_hours"] is None


def test_mut_audit_and_queue_rewrites_survive_a_legacy_nan_row(tmp_path: Path) -> None:
    from brain.initiate.audit import append_audit_row, read_recent_audit, update_audit_state
    from brain.initiate.emit import emit_initiate_candidate, read_candidates, remove_candidate
    from brain.initiate.schemas import AuditRow, SemanticContext

    def row(i: str) -> AuditRow:
        return AuditRow(
            audit_id=i, candidate_id="c" + i, ts=datetime.now(UTC).isoformat(), kind="message", subject="s",
            tone_rendered="t", decision="send_quiet", decision_reasoning="r", gate_check={"allowed": True},
        )

    nan_row = row("n")
    nan_row.gate_check = {"allowed": True, "sim": float("nan")}
    append_audit_row(tmp_path, nan_row)  # a NaN in a new row is written as null, not lost
    assert "NaN" not in (tmp_path / "initiate_audit.jsonl").read_text()
    (tmp_path / "initiate_audit.jsonl").unlink()
    append_audit_row(tmp_path, row("a"))
    append_audit_row(tmp_path, row("b"))
    path = tmp_path / "initiate_audit.jsonl"
    lines = path.read_text().splitlines()
    path.write_text(lines[0].replace('"allowed": true', '"allowed": true, "sim": NaN') + "\n" + lines[1] + "\n")
    update_audit_state(tmp_path, audit_id="b", new_state="delivered", at=datetime.now(UTC).isoformat())
    got = {r.audit_id: r for r in read_recent_audit(tmp_path, window_hours=1)}
    assert got["b"].delivery["current_state"] == "delivered" and got["a"].gate_check["sim"] is None

    for i in range(2):
        emit_initiate_candidate(
            tmp_path, kind="message", source="dream", source_id=f"s{i}",
            semantic_context=SemanticContext(source_meta={"x": float("nan")}), now=NOW + timedelta(seconds=i),
        )
    q = tmp_path / "initiate_candidates.jsonl"
    assert "NaN" not in q.read_text()  # the appender already wrote null
    q.write_text(q.read_text().replace('"x": null', '"x": NaN'))  # a legacy file
    ids = [c.candidate_id for c in read_candidates(tmp_path)]
    remove_candidate(tmp_path, ids[0])
    assert [c.candidate_id for c in read_candidates(tmp_path)] == [ids[1]]


def test_mut_soul_candidate_rewrites_write_nan_as_null(tmp_path: Path) -> None:
    from brain.soul.review import _save_soul_candidates

    _save_soul_candidates(tmp_path, [{"memory_id": "m", "importance": float("nan")}])
    assert json.loads((tmp_path / "soul_candidates.jsonl").read_text())["importance"] is None


def test_mut_maker_charge_with_a_legacy_nan_is_a_cold_start_and_saves_again(tmp_path: Path) -> None:
    from brain.maker.charge import load_charge, save_charge

    (tmp_path / "maker_charge.json").write_text('{"charge": NaN}') if (tmp_path / "x").exists() else None
    import brain.maker.charge as mc

    p = mc._charge_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"charge": NaN, "prior_soul_count": 2}')
    state = load_charge(tmp_path)
    assert state.charge == 0.0  # non-finite is corrupt: cold default
    save_charge(tmp_path, state)  # and the next save does not raise
    assert json.loads(p.read_text())["charge"] == 0.0
