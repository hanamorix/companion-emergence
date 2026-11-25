"""Self-authored reconcile tool — reconcile_self_read.

She revises her own read of herself (accept/revise), lets a gap pass (dismiss),
or names an unnamed pressure into a vocab-crystalliser CANDIDATE (name).

Load-bearing guards:
  - R-B2: post-reconcile channel cooldown blocks a new gap on that channel.
  - R-A:  name → candidate ONLY via the guarded crystalliser path; never a
          direct/placeholder mint into emotion_vocabulary.json (the v0.0.32 bug).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from brain.self_model.gap import Gap
from brain.self_model.reconcile import reconcile_self_read
from brain.self_model.state import SelfModelState, load_or_recover, save


def _seed_open_gap(persona_dir: Path, **kw) -> None:
    base = {
        "per_channel": {"grief": 4.0},
        "magnitude": 4.0,
        "unnamed_pressure": 0.0,
        "status": "open",
    }
    base.update(kw)
    save(persona_dir, SelfModelState(current_gap=Gap(**base)))


def test_accept_marks_gap_acknowledged_and_sets_cooldown(tmp_path: Path) -> None:
    _seed_open_gap(tmp_path)
    result = reconcile_self_read(
        persona_dir=tmp_path, action="accept", channel="grief", delta=0.2
    )
    assert result.get("ok") is True
    state, _ = load_or_recover(tmp_path)
    assert state.current_gap is not None
    assert state.current_gap.status == "acknowledged"
    assert "grief" in state.current_gap.channel_cooldowns


def test_revise_writes_no_row_and_still_acknowledges_the_gap(tmp_path: Path) -> None:
    """Accept/revise used to commit a standalone `self_model_reconcile` row for the old aggregate to pool. That aggregate
    is gone, so no row is written; the gap is still acknowledged and the channel still gets its cooldown."""
    from brain.memory.store import MemoryStore

    _seed_open_gap(tmp_path)
    result = reconcile_self_read(persona_dir=tmp_path, action="revise", channel="grief", delta=0.5)
    assert result.get("ok") is True

    store = MemoryStore(tmp_path / "memories.db")
    try:
        mems = [m for m in store.list_active() if m.memory_type == "self_model_reconcile"]
    finally:
        store.close()
    assert mems == []
    state, _ = load_or_recover(tmp_path)
    assert state.current_gap is not None
    assert state.current_gap.status == "acknowledged"
    assert "grief" in state.current_gap.channel_cooldowns


def test_dismiss_marks_gap_dismissed_and_sets_cooldown(tmp_path: Path) -> None:
    _seed_open_gap(tmp_path)
    result = reconcile_self_read(persona_dir=tmp_path, action="dismiss", channel="grief")
    assert result.get("ok") is True
    state, _ = load_or_recover(tmp_path)
    assert state.current_gap is not None
    assert state.current_gap.status == "dismissed"
    assert "grief" in state.current_gap.channel_cooldowns


def test_reconcile_cooldown_blocks_new_gap_on_channel(tmp_path: Path) -> None:
    # R-B2: after a reconcile, the channel is in cooldown — a new gap on it is
    # suppressed until the cooldown expires.
    from datetime import UTC, datetime, timedelta

    from brain.self_model.reconcile import is_channel_in_cooldown

    _seed_open_gap(tmp_path)
    reconcile_self_read(persona_dir=tmp_path, action="accept", channel="grief", delta=0.2)
    state, _ = load_or_recover(tmp_path)
    gap = state.current_gap
    now = datetime.now(UTC)

    # Right now (within the window) → in cooldown, a new grief gap is blocked.
    assert is_channel_in_cooldown(gap, "grief", now=now) is True
    # A different channel is NOT in cooldown.
    assert is_channel_in_cooldown(gap, "joy", now=now) is False
    # Well past the cooldown window → no longer blocked.
    assert is_channel_in_cooldown(gap, "grief", now=now + timedelta(days=7)) is False


def _seed_blend_memories(persona_dir: Path, a: str, b: str, n: int = 3) -> None:
    """Commit n memories whose two strongest emotions are a+b (both intense),
    so crystallize_vocabulary detects a repeated blend."""
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        for i in range(n):
            mem = Memory.create_new(
                content=f"a moment of {a} and {b} #{i}",
                memory_type="episodic",
                domain="self",
                emotions={a: 8.0, b: 7.0},
                importance=8.0,
            )
            store.create(mem)
    finally:
        store.close()


def test_name_grows_vocab_only_via_crystalliser_not_placeholder_stub(tmp_path: Path) -> None:
    """R-A (references the v0.0.32 stub-flood bug).

    A full name->candidate cycle grows emotion_vocabulary.json ONLY through the
    guarded crystalliser: the new entry carries a real 45-day half-life and a
    real description, NOT the 1.0-day placeholder stub the v0.0.32 bug minted.
    Her literal 'name' word is NEVER minted directly.
    """
    import json

    from brain.emotion import vocabulary

    vocabulary.set_bridge_owner(True)  # as in the bridge, the one process that adds a name (tests/conftest.py restores the flag)
    _seed_open_gap(tmp_path, per_channel={}, magnitude=0.0, unnamed_pressure=0.6)
    _seed_blend_memories(tmp_path, "grief", "loneliness", n=3)

    vocab_path = tmp_path / "emotion_vocabulary.json"
    assert not vocab_path.exists()  # nothing minted yet

    # She names the pressure with an arbitrary word — must NOT be minted directly.
    result = reconcile_self_read(persona_dir=tmp_path, action="name", name="that-hollow-ache")
    assert result.get("ok") is True

    assert vocab_path.exists(), "crystalliser should have grown the vocab from real evidence"
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    names = [e["name"] for e in data["emotions"]]

    # Her literal word is NEVER a vocab entry (R-A: no direct mint).
    assert "that-hollow-ache" not in names
    # The crystalliser proposed an evidence-grounded blend instead.
    assert any(n.endswith("_blend") for n in names)

    entry = next(e for e in data["emotions"] if e["name"].endswith("_blend"))
    # Proper half-life — NOT the 1.0-day placeholder stub the v0.0.32 bug minted.
    assert entry["decay_half_life_days"] == 45.0
    assert entry["decay_half_life_days"] != 1.0
    # Real description, not an empty/placeholder string.
    assert isinstance(entry.get("description"), str) and len(entry["description"]) > 20


def test_name_dedups_near_duplicate_of_existing_channel(tmp_path: Path) -> None:
    """R-A: a blend whose name already exists in the vocabulary is not re-minted."""
    import json

    from brain.health.attempt_heal import save_with_backup

    # Pre-seed the vocab with the exact blend the evidence would propose.
    vocab_path = tmp_path / "emotion_vocabulary.json"
    save_with_backup(
        vocab_path,
        {
            "version": 1,
            "emotions": [
                {
                    "name": "grief_loneliness_blend",
                    "description": "an existing registered blend",
                    "category": "persona_extension",
                    "decay_half_life_days": 45.0,
                    "intensity_clamp": 10.0,
                }
            ],
        },
    )
    before = json.loads(vocab_path.read_text(encoding="utf-8"))

    _seed_open_gap(tmp_path, per_channel={}, magnitude=0.0, unnamed_pressure=0.6)
    _seed_blend_memories(tmp_path, "grief", "loneliness", n=3)

    result = reconcile_self_read(persona_dir=tmp_path, action="name", name="another-word")
    assert result.get("ok") is True

    after = json.loads(vocab_path.read_text(encoding="utf-8"))
    after_names = [e["name"] for e in after["emotions"]]
    # No duplicate grief_loneliness_blend entry — deduped by the guarded path.
    assert after_names.count("grief_loneliness_blend") == 1
    # The vocab did not grow (the only candidate was a near-duplicate).
    assert len(after["emotions"]) == len(before["emotions"])


def test_reconcile_through_dispatch(tmp_path: Path) -> None:
    """reconcile_self_read routes through the brain-tool dispatcher (5-edit)."""
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools.dispatch import dispatch

    _seed_open_gap(tmp_path)
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(db_path=tmp_path / "hebbian.db")
    try:
        result = dispatch(
            "reconcile_self_read",
            {"action": "accept", "channel": "grief", "delta": 0.2},
            store=store,
            hebbian=hebbian,
            persona_dir=tmp_path,
        )
    finally:
        store.close()
        hebbian.close()
    assert isinstance(result, dict)
    assert result.get("ok") is True
    state, _ = load_or_recover(tmp_path)
    assert state.current_gap.status == "acknowledged"


def test_successful_reconcile_increments_audit_counter(tmp_path: Path) -> None:
    """C2: a successful reconcile bumps reconciles_called (the dead-loop audit
    counter — resolve.py docstring claim becomes true).

    The counter makes 'she surfaces the gap repeatedly but never reconciles'
    observable; before this wiring it stayed 0 even on real reconciles.
    """
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.self_model.resolve import load_audit
    from brain.tools.dispatch import dispatch

    assert load_audit(tmp_path)["reconciles_called"] == 0

    _seed_open_gap(tmp_path)
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(db_path=tmp_path / "hebbian.db")
    try:
        dispatch(
            "reconcile_self_read",
            {"action": "dismiss", "channel": "grief"},
            store=store,
            hebbian=hebbian,
            persona_dir=tmp_path,
        )
    finally:
        store.close()
        hebbian.close()

    assert load_audit(tmp_path)["reconciles_called"] == 1


def test_name_does_not_acknowledge_a_pure_channel_gap(tmp_path: Path) -> None:
    """Minor: ``name`` resolves unnamed pressure. A gap with NO unnamed_pressure
    (a pure declared/derived channel divergence) is not what ``name`` addresses,
    so its status must be left untouched rather than spuriously acknowledged.
    """
    # A channel gap with zero unnamed pressure — name does not address it.
    _seed_open_gap(tmp_path, per_channel={"grief": 4.0}, magnitude=4.0, unnamed_pressure=0.0)
    result = reconcile_self_read(persona_dir=tmp_path, action="name", name="some-word")
    assert result.get("ok") is True
    state, _ = load_or_recover(tmp_path)
    assert state.current_gap is not None
    assert state.current_gap.status == "open", (
        "name must not acknowledge a gap it did not address (no unnamed_pressure)"
    )


def test_failed_reconcile_does_not_increment_audit_counter(tmp_path: Path) -> None:
    """C2: a malformed reconcile (error dict, no mutation) must NOT bump the counter."""
    from brain.self_model.resolve import load_audit

    # No 'channel' on an accept → error, nothing acted on.
    result = reconcile_self_read(persona_dir=tmp_path, action="accept")
    assert "error" in result
    assert load_audit(tmp_path)["reconciles_called"] == 0


def test_reconcile_keeps_the_gaps_directions_and_reset_seen(tmp_path: Path) -> None:
    """A field missing from the round trip would be dropped at the first reconcile or dismiss."""
    _seed_open_gap(
        tmp_path,
        directions={"grief": -1},
        reset_seen="2026-11-04T10:00:00+00:00",
    )
    reconcile_self_read(persona_dir=tmp_path, action="dismiss", channel="grief")
    state, _ = load_or_recover(tmp_path)
    assert state.current_gap.status == "dismissed"
    assert state.current_gap.directions == {"grief": -1}
    assert state.current_gap.reset_seen == "2026-11-04T10:00:00+00:00"


# ---------------------------------------------------------------------------
# Emotion rework B1 PR 2: a revision folds a signed change once the engine is live
# ---------------------------------------------------------------------------


def _go_live(persona_dir: Path, **seed: float):
    """A live persona with no state file applies nothing, so seed a memory first, then write the switch."""
    from datetime import UTC, datetime

    from brain.emotion import engine
    from brain.memory.store import Memory, MemoryStore
    from tests.unit.brain.emotion._engine_support import make_live

    store = MemoryStore(persona_dir / "memories.db")
    try:
        store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions=seed or {"joy": 5.0}))
    finally:
        store.close()
    make_live(persona_dir, datetime.now(UTC))
    return engine.engine_for(persona_dir)


def _settled(eng):
    from datetime import UTC, datetime

    eng.note_active(datetime.now(UTC))
    return eng.read()


def _reconcile_rows(persona_dir: Path) -> list:
    from brain.memory.pending import PendingQueue
    from brain.memory.store import MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        stored = store.list_by_type("self_model_reconcile", active_only=False)
    finally:
        store.close()
    return stored + PendingQueue(persona_dir).read_recent("self_model_reconcile", limit=50)


def test_live_revise_lowers_the_feeling_and_writes_no_row(tmp_path: Path) -> None:
    eng = _go_live(tmp_path, joy=5.0)
    before = _settled(eng)["joy"].short
    result = reconcile_self_read(persona_dir=tmp_path, action="revise", channel="joy", delta=-0.2)
    assert result["delta_written"] is True
    assert _settled(eng)["joy"].short < before
    assert _reconcile_rows(tmp_path) == []


def test_live_revise_off_vocab_channel_folds_nothing(tmp_path: Path) -> None:
    """The engine's fold checks the channel against her vocabulary (#801 section 6): with no bridge a channel she lacks is discarded."""
    eng = _go_live(tmp_path, joy=5.0)
    before = _settled(eng)
    result = reconcile_self_read(persona_dir=tmp_path, action="revise", channel="zorblesque", delta=0.5)
    assert result["delta_written"] is False  # a channel her vocabulary cannot hold is not folded, and the tool says so
    assert _settled(eng) == before
    assert _reconcile_rows(tmp_path) == []


# --- a step after the state change raises (RC18, #803 step 19): fail-soft, state kept, counter not incremented ---------------


def _raising_tick(exc: Exception):
    def tick(*_args, **_kwargs):
        raise exc

    return tick


_NAMING_STEM = "the naming step did not finish this time"
_CHANGE_STEM = "the step that applies your change to your emotions did not finish this time"

# The reason each error class gets (one table, both paths): (exception, reason).
def _reason_rows():
    import sqlite3

    from brain.bridge.cli_throttle import ShutdownDeferred, ThrottleDeferred
    from brain.bridge.provider_auth import ProviderAuthDeferred
    from brain.utils.file_lock import LockTimeoutError

    return [
        (ShutdownDeferred("stop flag set"), "the app is shutting down."),
        (ProviderAuthDeferred("token rejected"), "background work needs the login renewed."),
        (ThrottleDeferred("a CLI slot was not free"), "background work was paused, so you can make the same call again later."),
        (
            LockTimeoutError("waited for the vocabulary lock"),
            "something you needed was busy for too long, so you can make the same call again in a moment.",
        ),
        (sqlite3.OperationalError("unable to open database file"), "your memory database could not be opened or read."),
        (PermissionError(13, "Permission denied", "emotion_vocabulary.json"), "something could not be read or saved."),
        (RuntimeError("a bug"), "an unexpected error happened."),
    ]


def test_name_growth_tick_raising_returns_error_keeps_state_and_skips_counter(tmp_path: Path, monkeypatch, caplog) -> None:
    import logging

    from brain.self_model.resolve import load_audit

    for exc, reason in _reason_rows():
        _seed_open_gap(tmp_path, per_channel={}, magnitude=0.0, unnamed_pressure=0.6)
        before = load_audit(tmp_path)["reconciles_called"]
        monkeypatch.setattr("brain.growth.scheduler.run_growth_tick", _raising_tick(exc))
        caplog.clear()

        with caplog.at_level(logging.WARNING):
            result = reconcile_self_read(persona_dir=tmp_path, action="name", name="that-hollow-ache")

        assert result == {"error": f"{_NAMING_STEM}: {reason}"}
        assert str(exc) not in result["error"]  # the raw error stays out of the message
        assert "nothing was changed" not in result["error"]
        logged = " ".join(r.getMessage() for r in caplog.records)  # and goes to the log: step, class, text, file when named
        assert "naming step failed" in logged and type(exc).__name__ in logged and str(exc) in logged
        if isinstance(exc, OSError) and exc.filename:
            assert "emotion_vocabulary.json" in logged
        state, _ = load_or_recover(tmp_path)
        assert state.current_gap is not None
        assert state.current_gap.status == "acknowledged"  # the state change comes first and stays saved
        assert load_audit(tmp_path)["reconciles_called"] == before


def test_the_reason_table_tests_the_most_specific_class_first() -> None:
    from brain.bridge.cli_throttle import ShutdownDeferred, ThrottleDeferred
    from brain.self_model.reconcile import _REASON_UNEXPECTED, _REASONS, _reason_for
    from brain.utils.file_lock import LockTimeoutError

    assert _reason_for(ShutdownDeferred) == "the app is shutting down."
    assert _reason_for(ThrottleDeferred).startswith("background work was paused")
    assert _reason_for(LockTimeoutError).startswith("something you needed was busy")  # not the OSError row
    assert _reason_for(FileNotFoundError) == "something could not be read or saved."
    assert _reason_for(Exception) == "an unexpected error happened."
    for text in [reason for _cls, reason in _REASONS] + [_REASON_UNEXPECTED, _NAMING_STEM, _CHANGE_STEM]:
        assert not {"she", "her"} & set(text.lower().replace(".", "").split())  # kindled-facing: "you", never "she/her"
        assert chr(0x2014) not in text


def test_name_growth_tick_raising_on_a_pure_channel_gap_leaves_status_alone(tmp_path: Path, monkeypatch) -> None:
    from brain.self_model.resolve import load_audit

    _seed_open_gap(tmp_path, unnamed_pressure=0.0)
    monkeypatch.setattr("brain.growth.scheduler.run_growth_tick", _raising_tick(RuntimeError("boom")))
    result = reconcile_self_read(persona_dir=tmp_path, action="name", name="some-word")
    assert "error" in result
    state, _ = load_or_recover(tmp_path)
    assert state.current_gap is not None and state.current_gap.status == "open"
    assert load_audit(tmp_path)["reconciles_called"] == 0


def test_accept_and_revise_delta_step_raising_returns_error_keeps_state_and_skips_counter(tmp_path: Path, monkeypatch) -> None:
    from brain.self_model.resolve import load_audit

    for action in ("accept", "revise"):
        for exc, reason in _reason_rows():
            _seed_open_gap(tmp_path)

            def boom(*_args, _exc=exc, **_kwargs):
                raise _exc

            with monkeypatch.context() as m:
                m.setattr("brain.emotion.vocabulary_access.names_used", boom)
                result = reconcile_self_read(persona_dir=tmp_path, action=action, channel="grief", delta=0.3)
            assert result == {"error": f"{_CHANGE_STEM}: {reason}"}  # the reason by the class that reached the catch
            assert str(exc) not in result["error"]
            state, _ = load_or_recover(tmp_path)
            assert state.current_gap is not None
            assert state.current_gap.status == "acknowledged"
            assert "grief" in state.current_gap.channel_cooldowns
            assert load_audit(tmp_path)["reconciles_called"] == 0


# --- a change the engine swallowed a failure for is not reported as written (RE19 item 2, #803 step 20) -----------------


def _change_log_lines(persona_dir: Path) -> int:
    from brain.emotion import engine_files

    return sum(len(p.read_text(encoding="utf-8").splitlines()) for p in engine_files.changes_dir(persona_dir).glob("*.jsonl"))


def test_a_failed_queue_append_returns_the_error_leaves_the_change_log_empty_and_a_retry_writes_one_copy(tmp_path: Path, monkeypatch) -> None:
    from brain.emotion import engine_files
    from brain.self_model.resolve import load_audit

    _go_live(tmp_path, joy=5.0)
    _seed_open_gap(tmp_path, per_channel={"joy": 4.0})
    queued_before = len(engine_files.read_queue(tmp_path, from_line=0).events)
    with monkeypatch.context() as m:
        m.setattr(engine_files, "append_event", lambda *_a, **_k: False)
        result = reconcile_self_read(persona_dir=tmp_path, action="revise", channel="joy", delta=-0.2)
    assert result == {"error": f"{_CHANGE_STEM}: something could not be read or saved."}
    assert "delta_written" not in result
    assert _change_log_lines(tmp_path) == 0  # nothing in either file
    assert len(engine_files.read_queue(tmp_path, from_line=0).events) == queued_before
    state, _ = load_or_recover(tmp_path)
    assert state.current_gap is not None and state.current_gap.status == "acknowledged"  # as RC18 left it
    assert load_audit(tmp_path)["reconciles_called"] == 0

    again = reconcile_self_read(persona_dir=tmp_path, action="revise", channel="joy", delta=-0.2)  # callable straight away
    assert again["delta_written"] is True
    assert len(engine_files.read_queue(tmp_path, from_line=0).events) == queued_before + 1
    assert _change_log_lines(tmp_path) == 1  # one copy of the change, not two


def test_a_lock_timeout_or_a_bug_inside_the_engine_is_an_error_not_a_written_change(tmp_path: Path, monkeypatch) -> None:
    from brain.emotion import engine_files
    from brain.self_model.resolve import load_audit
    from brain.utils.file_lock import LockTimeoutError

    _go_live(tmp_path, joy=5.0)
    for raised, reason in (
        (LockTimeoutError("engine lock"), "something you needed was busy for too long, so you can make the same call again in a moment."),
        (PermissionError("denied"), "something could not be read or saved."),
        (RuntimeError("engine bug"), "an unexpected error happened."),
    ):
        _seed_open_gap(tmp_path, per_channel={"joy": 4.0})

        def boom(*_a, _raised=raised, **_k):
            raise _raised

        with monkeypatch.context() as m:
            m.setattr(engine_files, "state_lock", boom)
            result = reconcile_self_read(persona_dir=tmp_path, action="accept", channel="joy", delta=0.3)
        assert result == {"error": f"{_CHANGE_STEM}: {reason}"}
        assert load_audit(tmp_path)["reconciles_called"] == 0
        assert engine_files.read_queue(tmp_path, from_line=0).events == []  # nothing reached the queue


@pytest.mark.no_base_failure(reason="pins behaviour that already holds and that RE19 keeps: a change dropped whole for its names is delta_written false with no error")
def test_a_change_dropped_whole_for_its_names_is_false_with_no_error(tmp_path: Path) -> None:
    _go_live(tmp_path, joy=5.0)
    _seed_open_gap(tmp_path, per_channel={"zorblesque": 4.0})
    result = reconcile_self_read(persona_dir=tmp_path, action="revise", channel="zorblesque", delta=0.5)
    assert result["delta_written"] is False and "error" not in result


# --- the log line for a swallowed engine failure says specifically what went wrong (#803 step 20b) ----------------------


def _change_log_text(tmp_path: Path, caplog) -> str:
    return " ".join(r.getMessage() for r in caplog.records if r.getMessage().startswith("reconcile_self_read:"))


def test_the_log_line_for_a_swallowed_lock_timeout_names_the_class_and_text(tmp_path: Path, monkeypatch, caplog) -> None:
    import logging

    from brain.emotion import engine_files
    from brain.utils.file_lock import LockTimeoutError

    _go_live(tmp_path, joy=5.0)
    _seed_open_gap(tmp_path, per_channel={"joy": 4.0})

    def boom(*_a, **_k):
        raise LockTimeoutError("waited 30s for the engine lock")

    with monkeypatch.context() as m, caplog.at_level(logging.WARNING):
        m.setattr(engine_files, "state_lock", boom)
        result = reconcile_self_read(persona_dir=tmp_path, action="accept", channel="joy", delta=0.3)
    assert result == {"error": f"{_CHANGE_STEM}: something you needed was busy for too long, so you can make the same call again in a moment."}
    logged = _change_log_text(tmp_path, caplog)
    assert "the change step failed" in logged and "LockTimeoutError: waited 30s for the engine lock" in logged


def test_the_log_line_for_a_swallowed_oserror_names_class_text_file_and_carries_the_traceback(tmp_path: Path, monkeypatch, caplog) -> None:
    import logging

    from brain.emotion import engine_files

    _go_live(tmp_path, joy=5.0)
    _seed_open_gap(tmp_path, per_channel={"joy": 4.0})
    refused = tmp_path / "emotion" / "queue.jsonl"

    def boom(*_a, **_k):
        raise PermissionError(13, "Permission denied", str(refused))

    with monkeypatch.context() as m, caplog.at_level(logging.WARNING):
        m.setattr(engine_files, "state_lock", boom)
        result = reconcile_self_read(persona_dir=tmp_path, action="accept", channel="joy", delta=0.3)
    assert result == {"error": f"{_CHANGE_STEM}: something could not be read or saved."}
    records = [r for r in caplog.records if r.getMessage().startswith("reconcile_self_read:")]
    assert len(records) == 1
    logged = records[0].getMessage()
    assert "PermissionError" in logged and "Permission denied" in logged and str(refused) in logged
    assert records[0].exc_info is not None and records[0].exc_info[0] is PermissionError  # the traceback is kept


def test_the_log_line_for_an_append_that_returned_false_says_so_and_names_the_queue_file(tmp_path: Path, monkeypatch, caplog) -> None:
    import logging

    from brain.emotion import engine_files

    _go_live(tmp_path, joy=5.0)
    _seed_open_gap(tmp_path, per_channel={"joy": 4.0})
    with monkeypatch.context() as m, caplog.at_level(logging.WARNING):
        m.setattr(engine_files, "append_event", lambda *_a, **_k: False)
        result = reconcile_self_read(persona_dir=tmp_path, action="revise", channel="joy", delta=-0.2)
    assert result == {"error": f"{_CHANGE_STEM}: something could not be read or saved."}
    records = [r for r in caplog.records if r.getMessage().startswith("reconcile_self_read:")]
    assert len(records) == 1
    logged = records[0].getMessage()
    assert "returned False" in logged and str(engine_files.queue_path(tmp_path)) in logged
    assert records[0].exc_info is None  # no exception exists for this kind


def test_the_log_line_for_an_unexpected_exception_in_the_engine_names_class_and_text(tmp_path: Path, monkeypatch, caplog) -> None:
    import logging

    from brain.bridge.cli_throttle import ThrottleDeferred
    from brain.emotion import engine_files

    _go_live(tmp_path, joy=5.0)
    _seed_open_gap(tmp_path, per_channel={"joy": 4.0})

    def boom(*_a, **_k):
        raise RuntimeError("a bug in the hook")

    with monkeypatch.context() as m, caplog.at_level(logging.WARNING):
        m.setattr(engine_files, "state_lock", boom)
        result = reconcile_self_read(persona_dir=tmp_path, action="accept", channel="joy", delta=0.3)
    assert result == {"error": f"{_CHANGE_STEM}: an unexpected error happened."}
    records = [r for r in caplog.records if r.getMessage().startswith("reconcile_self_read:")]
    assert len(records) == 1 and "RuntimeError: a bug in the hook" in records[0].getMessage()
    assert records[0].exc_info is not None and records[0].exc_info[0] is RuntimeError

    # The reason row follows the guard's class, not the swallowed exception's own class: a ThrottleDeferred that the
    # guard swallowed as a bug still gets the unexpected-error row, as before this step.
    _seed_open_gap(tmp_path, per_channel={"joy": 4.0})

    def deferred(*_a, **_k):
        raise ThrottleDeferred("a CLI slot was not free")

    with monkeypatch.context() as m:
        m.setattr(engine_files, "state_lock", deferred)
        again = reconcile_self_read(persona_dir=tmp_path, action="accept", channel="joy", delta=0.3)
    assert again == {"error": f"{_CHANGE_STEM}: an unexpected error happened."}
