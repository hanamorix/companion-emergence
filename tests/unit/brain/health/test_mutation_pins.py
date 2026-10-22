"""Pins that kill the mutations the stage-6 code reviews found surviving (build record: mutation table).

Each RAISE/CONTENT site is exercised with a read fault that is NOT a directory (a denied open), so a site
demoted to DEFAULT is caught, and a destructive step cannot fail on its own and hide the demotion.
The best-effort/encoder pins call the real writers with a NaN row. Nothing touches a persona directory.
"""

from __future__ import annotations

import ast
import builtins
import json
import logging
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
# F1: tunables sync_file must not touch a settings file it could not read
# ---------------------------------------------------------------------------------------------


def test_mut_tunables_boot_pass_keeps_values_when_the_file_is_unreadable(tmp_path: Path, monkeypatch, deny) -> None:
    from brain import tunables

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables._reset_for_tests()
    tunables.register("a.b", 1, description="A setting.")
    p = tunables._file_path()
    p.write_text('"a.b" = 3\n')
    before = p.read_bytes()
    deny("tunables.toml")
    tunables.sync_file()  # never breaks startup; the pass is skipped and the file is left as it is
    assert p.read_bytes() == before


@pytest.mark.parametrize("os_name", [None, "nt"], ids=["this-os", "simulated-windows"])
def test_mut_tunables_boot_pass_keeps_a_hand_edited_nan_and_makes_the_write_durable(
    tmp_path: Path, monkeypatch, os_name: str | None
) -> None:
    import os

    from brain import tunables
    from brain.utils import atomic_write

    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (events.append("replace"), real_replace(a, b))[1])
    if os_name is not None:
        # Give the writer alone a view of ``os`` that reports another platform name (patching ``os.name`` itself would
        # break pathlib); every other attribute is the real one, including the instrumented fsync and replace above.
        class _OsAs:
            name = os_name

            def __getattr__(self, attr: str):
                return getattr(os, attr)

        monkeypatch.setattr(atomic_write, "os", _OsAs())

    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    tunables._reset_for_tests()
    tunables.register("new.key", 1, description="A setting.")
    p = tunables._file_path()
    p.write_text('"z.k" = nan\n"a.k" = 1\n')  # TOML reads nan; a hand edit can contain it
    tunables.sync_file()
    text = p.read_text()
    assert text.startswith('"z.k" = nan\n"a.k" = 1\n')  # the user's lines are byte for byte, in their order
    assert '"new.key" = 1' in text
    # The pass writes once through the shared atomic writer: the file's data is fsynced BEFORE the one replace (a power
    # loss must not leave a zero-byte file) on every OS. The writer then fsyncs the directory so the rename itself is
    # durable, but only where it can: ``fsync_dir`` returns early unless ``os.name == "posix"`` (Windows has no directory
    # handles), so the trailing fsync is expected exactly when the writer's own check says it runs.
    expected = ["fsync", "replace"] + (["fsync"] if atomic_write.os.name == "posix" else [])
    assert events == expected


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

    # The NaN is set after construction, so the helper works whether or not Memory.create_new itself rejects a
    # non-finite emotion (it may): the tests below are about what the writers do with a record that carries one.
    mem = Memory.create_new(content="x", memory_type="conversation", domain="us", emotions={"love": 0.5})
    mem.emotions["love"] = float("nan")
    mem.score = float("nan")
    return mem


def test_mut_archive_and_log_appenders_write_a_nan_as_null_instead_of_raising(tmp_path: Path) -> None:
    from brain.engines import consolidation, dedup_sweep
    from brain.engines.heartbeat import HeartbeatEngine

    consolidation._archive_preimage(tmp_path, _mem_nan(), "s1")
    dedup_sweep._archive_preimage(tmp_path, _mem_nan())
    rows = [json.loads(line) for line in (tmp_path / "consolidation_archive.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(r["target"]["emotions"]["love"] is None for r in rows)
    HeartbeatEngine._append_log(SimpleNamespace(heartbeat_log_path=tmp_path / "hb.jsonl"), {"elapsed_seconds": math.inf})
    assert json.loads((tmp_path / "hb.jsonl").read_text())["elapsed_seconds"] is None


def test_mut_pending_enqueue_of_a_nan_memory_is_promoted_by_the_gate(tmp_path: Path, caplog) -> None:
    """Behaviour delta of the migration, recorded: before it, a NaN candidate was queued as the NaN token, read
    back by Memory.from_dict, and then crashed the gate at store.create (SQLite NOT NULL on score) after the queue
    had been drained. Now the NaN is clamped to 0.0 at the producer (with one warning) and the candidate is
    promoted. Writing null instead would hand the gate's float() parse something that is not a number."""
    import logging

    from brain.engines.consolidation import run_consolidation
    from brain.memory.pending import PendingQueue
    from brain.memory.store import Memory, MemoryStore

    q = PendingQueue(tmp_path)
    mem = _mem_nan()
    with caplog.at_level(logging.WARNING, logger="brain.utils.finite"):
        q.enqueue(mem, source="t")
    msgs = [r.getMessage() for r in caplog.records if r.name == "brain.utils.finite"]
    assert len(msgs) == 1 and "pending candidate" in msgs[0] and "emotions.love=nan" in msgs[0] and "score=nan" in msgs[0]
    back = Memory.from_dict(json.loads(q.path.read_text().splitlines()[0]))  # raised TypeError with null
    assert back.emotions["love"] == 0.0 and back.score == 0.0

    store = MemoryStore(tmp_path / "memories.db")
    try:
        result = run_consolidation(store, persona_dir=tmp_path)
        assert result.promoted == 1  # reached store.create without IntegrityError
        stored = store.get(mem.id, bump=False)
        assert stored.emotions == {"love": 0.0} and stored.score == 0.0 and stored.content == "x"
    finally:
        store.close()


def test_mut_behavioral_appender_writes_nan_as_null(tmp_path: Path) -> None:
    from brain.behavioral.log import append_behavioral_event, read_behavioral_log

    append_behavioral_event(
        tmp_path / "b.jsonl", kind="journal_entry_added", name="n", timestamp=NOW,
        emotional_state={"love": float("nan")},
    )
    (row,) = read_behavioral_log(tmp_path / "b.jsonl")
    assert row["emotional_state"] == {"love": None}


def test_mut_growth_event_with_nan_is_read_back(tmp_path: Path, caplog) -> None:
    """read_growth_log float()s the score (null would skip the row); a non-finite half-life is stored as the
    reader's own "no decay" null."""
    from brain.growth.log import GrowthLogEvent, append_growth_event, read_growth_log

    ev = GrowthLogEvent(
        timestamp=NOW, type="emotion_added", name="n", description="d", decay_half_life_days=float("nan"),
        reason="r", evidence_memory_ids=("m",), score=float("nan"), relational_context=None,
    )
    with caplog.at_level(logging.WARNING, logger="brain.utils.finite"):
        append_growth_event(tmp_path / "g.jsonl", ev)
    msgs = [r.getMessage() for r in caplog.records if r.name == "brain.utils.finite"]
    assert len(msgs) == 2 and any("score" in m for m in msgs) and any("decay_half_life_days" in m for m in msgs)
    (back,) = read_growth_log(tmp_path / "g.jsonl")
    assert back.score == 0.0 and back.decay_half_life_days is None and back.name == "n"


def test_mut_narrative_event_with_nan_is_replayed(tmp_path: Path, caplog) -> None:
    """_reconstruct_arcs_from_log float()s lived_age_hours; one null used to abort the whole replay."""
    from brain.narrative_memory.state import LOG_FILENAME, append_event
    from brain.narrative_memory.tool import _reconstruct_arcs_from_log

    with caplog.at_level(logging.WARNING, logger="brain.utils.finite"):
        append_event(tmp_path, {"event": "arc_opened", "arc_id": "a1", "title": "t", "ts_iso": NOW.isoformat(),
                                "lived_age_hours": float("nan")})
    assert any("arc log event arc_opened" in r.getMessage() and "lived_age_hours=nan" in r.getMessage()
               for r in caplog.records)
    append_event(tmp_path, {"event": "arc_closed", "arc_id": "a1", "ts_iso": NOW.isoformat(),
                            "lived_age_hours": float("inf")})
    arcs = _reconstruct_arcs_from_log(tmp_path / LOG_FILENAME)
    assert [a.id for a in arcs] == ["a1"]


def test_mut_graveyard_entry_with_nan_weight_is_restorable(tmp_path: Path, caplog) -> None:
    """The restore plan float()s every hebbian neighbour weight; one null used to abort the whole plan."""
    from brain.forgetting.graveyard import append
    from brain.forgetting.salience import SalienceInputs
    from brain.recovery.engine import _build_restore_plan

    mem = _mem_nan()
    inputs = SalienceInputs(emotion=0.1, hebbian=0.1, recall=0.1, soul=0.1, freshness=0.1)
    with caplog.at_level(logging.WARNING, logger="brain.utils.finite"):
        append(tmp_path, memory=mem, salience_at_drop=float("nan"), inputs=inputs, lived_age_hours=1.0,
               reason="fade", hebbian_neighbors=[("n1", float("nan")), ("n2", 0.5)])
    (msg,) = [r.getMessage() for r in caplog.records if r.name == "brain.utils.finite"]
    assert "graveyard entry" in msg and "hebbian_neighbors[0][1]=nan" in msg and "salience_at_drop=nan" in msg
    plan = _build_restore_plan(tmp_path, source_dir=None)
    assert plan.graveyard_neighbors[mem.id] == [("n1", 0.0), ("n2", 0.5)]


def test_mut_float_bearing_logs_without_a_float_reader_write_nan_as_null(tmp_path: Path) -> None:
    """arc_storage's removed-arc log is read for name and removed_at only; the other writers below carry floats
    into files nothing parses numerically. A NaN reaches disk as null, not as an exception."""
    from brain.engines.reflex import ReflexArc
    from brain.growth.arc_storage import append_removed_arc, read_removed_arcs
    from brain.health.soul_candidate_repair import _save_candidates
    arc = ReflexArc(name="a", description="d", trigger={"love": float("nan")}, days_since_human_min=1.0,
                    cooldown_hours=1.0, action="x", output_memory_type="reflex", prompt_template="p")
    append_removed_arc(tmp_path, arc=arc, removed_at=NOW, removed_by="user_edit", reasoning=None)
    (row,) = read_removed_arcs(tmp_path)
    assert row["trigger_snapshot"] == {"love": None} and row["name"] == "a"

    _save_candidates(tmp_path / "c.jsonl", [{"memory_id": "m", "importance": float("nan")}])
    assert json.loads((tmp_path / "c.jsonl").read_text())["importance"] is None



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


def test_mut_maker_charge_with_a_legacy_nan_clamps_only_the_charge(tmp_path: Path, caplog) -> None:
    import logging

    import brain.maker.charge as mc

    p = mc._charge_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"charge": NaN, "last_tick_ts": "t1", "last_fire_ts": "f1", "prior_soul_count": 2}')
    with caplog.at_level(logging.WARNING, logger="brain.maker.charge"):
        state = mc.load_charge(tmp_path)
    assert (state.charge, state.last_tick_ts, state.last_fire_ts, state.prior_soul_count) == (0.0, "t1", "f1", 2)
    assert any("not finite" in r.getMessage() for r in caplog.records)  # one line of evidence
    mc.save_charge(tmp_path, state)  # and the next strict save does not raise
    assert json.loads(p.read_text())["last_fire_ts"] == "f1"


def test_mut_stream_raw_closes_quietly_when_the_caller_closed_the_file_first(tmp_path: Path) -> None:
    """An abandoned iterator can be closed after the caller's `with` closed the file: nothing is left to detach."""
    p = tmp_path / "s.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n')
    fh = open(p, "rb")  # noqa: SIM115
    gen = jio._stream_raw(fh, p, False, False)
    assert next(gen)[1] == {"a": 1}
    fh.close()
    gen.close()  # raised ValueError ("I/O operation on closed file") with an unconditional detach
