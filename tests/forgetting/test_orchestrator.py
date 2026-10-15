# tests/forgetting/test_orchestrator.py
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from brain.felt_time.state import FeltTimeState
from brain.felt_time.state import persist as persist_felt_time
from brain.forgetting import run_pass
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def persona_with_low_salience_memory(tmp_path):
    # Set up FeltTimeState with non-cold lived_age so recent-buffer
    # exemption doesn't apply.
    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"),
        tmp_path,
    )
    store = MemoryStore(tmp_path / "memories.db")
    # An OLD memory with no emotions, no soul link, no recalls → very low salience
    m = Memory.create_new(
        content="boring old memory body",
        memory_type="episodic",
        domain="chat",
        emotions={},
    )
    # Make it old enough to escape the recent-buffer exemption.
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=40))
    store.create(m)
    store.close()
    return tmp_path, m.id


def test_run_pass_fades_low_salience_memory(persona_with_low_salience_memory):
    persona_dir, mem_id = persona_with_low_salience_memory
    event_bus = MagicMock()
    summary = run_pass(persona_dir, event_bus=event_bus)
    assert summary["faded"] >= 1

    store = MemoryStore(persona_dir / "memories.db")
    row = store._conn.execute("SELECT state FROM memories WHERE id = ?", (mem_id,)).fetchone()
    assert row["state"] == "fading"
    store.close()


def test_run_pass_exempts_recent_buffer(tmp_path):
    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"), tmp_path
    )
    store = MemoryStore(tmp_path / "memories.db")
    # Just-created memory — within RECENT_LIVED_HOURS exemption window
    m = Memory.create_new(content="x", memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.close()

    event_bus = MagicMock()
    summary = run_pass(tmp_path, event_bus=event_bus)
    assert summary["faded"] == 0
    assert summary["exempt"] >= 1


def test_run_pass_exempts_recent_factual_memory_within_grace(tmp_path):
    """Part A through-pass: the exact user report — a memory from a few days
    ago, no emotion/recall/soul, must stay ACTIVE under the 30-day grace.

    RED before Part A: 5 wall-days (120h) exceeds the old 24h grace, so this
    memory fades. GREEN after: 120h < 720h → exempt, stays active.
    """
    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"), tmp_path
    )
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(
        content="we talked about the dog's name a few days ago",
        memory_type="episodic",
        domain="chat",
        emotions={},
    )
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=5))
    store.create(m)
    store.close()

    summary = run_pass(tmp_path, event_bus=MagicMock())
    assert summary["faded"] == 0
    assert summary["exempt"] >= 1

    store = MemoryStore(tmp_path / "memories.db")
    row = store._conn.execute(
        "SELECT state FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    assert row["state"] == "active"
    store.close()


def test_run_pass_lost_after_two_consecutive_low_passes(persona_with_low_salience_memory):
    persona_dir, mem_id = persona_with_low_salience_memory
    event_bus = MagicMock()
    # First pass: fades the memory (active → fading), consecutive_low_passes=1
    summary1 = run_pass(persona_dir, event_bus=event_bus)
    assert summary1["faded"] >= 1
    # Second pass: counter increments to 2, but LOSE threshold needs salience<0.10
    # — with no emotions, no recalls, no soul, no freshness, salience is essentially 0
    summary2 = run_pass(persona_dir, event_bus=event_bus)
    summary3 = run_pass(persona_dir, event_bus=event_bus)
    # By the third pass the memory should be lost OR fading->lost transition fired
    total_lost = summary1["lost"] + summary2["lost"] + summary3["lost"]
    assert total_lost >= 1

    store = MemoryStore(persona_dir / "memories.db")
    assert store.get(mem_id) is None  # row deleted
    store.close()

    # Graveyard entry exists
    from brain.forgetting import graveyard

    entries = graveyard.read_all(persona_dir)
    assert any(e["memory_id"] == mem_id for e in entries)


def test_run_pass_publishes_aggregate_event(persona_with_low_salience_memory):
    persona_dir, _mem_id = persona_with_low_salience_memory
    event_bus = MagicMock()
    summary = run_pass(persona_dir, event_bus=event_bus)
    assert "faded" in summary
    assert "unfaded" in summary
    assert "lost" in summary
    assert "exempt" in summary
    assert "total" in summary
    assert "duration_ms" in summary
    # event_bus should have been called once with this event
    event_bus.publish.assert_called_once()


def test_run_pass_cold_start_no_memories(tmp_path):
    event_bus = MagicMock()
    summary = run_pass(tmp_path, event_bus=event_bus)
    assert summary["total"] == 0
    assert summary["faded"] == 0
    assert summary["lost"] == 0


def test_run_pass_intensity_drivers_protects_borderline_memory(tmp_path):
    """intensity_drivers.narrative_weight threads to policy and protects memories from fading."""
    from datetime import timedelta
    from unittest.mock import patch

    from brain.felt_time.lived_age import IntensityDrivers

    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"),
        tmp_path,
    )
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(content="arc memory", memory_type="episodic", domain="chat", emotions={})
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=40))
    store.create(m)
    store.close()

    # Pin salience to just below baseline FADE_THRESHOLD but above the arc-adjusted threshold.
    # effective_fade at narrative_weight=1.0 → FADE_THRESHOLD / 2 = 0.125
    borderline_salience = 0.20  # 0.125 < 0.20 < 0.25

    event_bus = MagicMock()
    with patch("brain.forgetting.salience.score", return_value=borderline_salience):
        # Without arc pressure: salience 0.20 < FADE_THRESHOLD 0.25 → should fade
        summary_no_arc = run_pass(tmp_path, event_bus=event_bus)
        assert summary_no_arc["faded"] >= 1, "baseline: borderline memory should fade without arc"

    # Reload memory to active state
    store = MemoryStore(tmp_path / "memories.db")
    store._conn.execute("UPDATE memories SET state = 'active' WHERE id = ?", (m.id,))
    store._conn.commit()
    store.close()

    event_bus2 = MagicMock()
    drivers = IntensityDrivers(narrative_weight=1.0)
    with patch("brain.forgetting.salience.score", return_value=borderline_salience):
        # With full arc pressure: effective_fade = 0.125, 0.20 >= 0.125 → should NOT fade
        summary_arc = run_pass(tmp_path, event_bus=event_bus2, intensity_drivers=drivers)
        assert summary_arc["faded"] == 0, "arc pressure: borderline memory should be protected"


def test_run_pass_recovers_from_corrupt_forgetting_state(persona_with_low_salience_memory):
    persona_dir, _mem_id = persona_with_low_salience_memory
    # Pre-corrupt the state file.
    (persona_dir / "forgetting_state.json").write_text("{not valid json")
    event_bus = MagicMock()
    summary = run_pass(persona_dir, event_bus=event_bus)
    # Should NOT crash; consecutive_low_passes counters all reset to 0
    assert "faded" in summary


def test_run_pass_exempts_memory_under_soul_review_through_real_candidates_file(tmp_path):
    """CANARY — pins the full composition run_pass → _load_soul_linked_ids →
    brain.soul.candidates.list_under_review_memory_ids against a REAL
    soul_candidates.jsonl.

    History: forgetting's lazy import of brain.soul.candidates referenced a
    module that did not exist from v0.0.14 until 2026-06-12 (the fail-soft
    except swallowed the ImportError), so the under-review exemption was
    silently dead for ~18 versions. This test fails if that composition ever
    breaks again (module rename, signature drift, status-set drift).
    """
    from brain.ingest.soul_queue import queue_soul_candidate
    from brain.ingest.types import ExtractedItem

    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"),
        tmp_path,
    )
    store = MemoryStore(tmp_path / "memories.db")
    # Identical low-salience shape to the fades test — old, no emotions, no recalls.
    m = Memory.create_new(
        content="quiet moment awaiting soul review",
        memory_type="monologue_soul_candidate",
        domain="monologue",
        emotions={},
    )
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=40))
    store.create(m)
    store.close()

    # Real candidates file via the real queue path: auto_pending = under review.
    queue_soul_candidate(
        tmp_path,
        memory_id=m.id,
        item=ExtractedItem(text="quiet moment awaiting soul review", label="observation", importance=8),
        session_id="monologue",
    )

    summary = run_pass(tmp_path, event_bus=MagicMock())

    store = MemoryStore(tmp_path / "memories.db")
    row = store._conn.execute("SELECT state FROM memories WHERE id = ?", (m.id,)).fetchone()
    store.close()
    assert row["state"] == "active", (
        "memory under soul review must be exempt from fading — the "
        "run_pass→soul.candidates composition is broken"
    )
    assert summary["exempt"] >= 1


def _old_low_salience_memory(persona_dir) -> str:
    """An old, emotionless, never-recalled memory: the shape forgetting fades."""
    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"),
        persona_dir,
    )
    store = MemoryStore(persona_dir / "memories.db")
    m = Memory.create_new(
        content="a moment the soul review accepted",
        memory_type="monologue_soul_candidate",
        domain="monologue",
        emotions={},
    )
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=40))
    store.create(m)
    store.close()
    return m.id


def _accept_candidate_for_real(persona_dir, memory_id: str) -> str:
    """Queue a candidate and accept it through the REAL review code.

    Uses ``queue_soul_candidate`` for the record shape and ``_apply_accept``
    for the accept (it mutates the record to ``accepted`` and creates the
    Crystallization), then writes ``soul_candidates.jsonl`` back the way the
    review pass does. Nothing here injects ids into forgetting. Returns the
    crystallization id.
    """
    import json

    from brain.ingest.soul_queue import queue_soul_candidate
    from brain.ingest.types import ExtractedItem
    from brain.soul.review import Decision, _apply_accept
    from brain.soul.store import SoulStore

    queue_soul_candidate(
        persona_dir,
        memory_id=memory_id,
        item=ExtractedItem(
            text="a moment the soul review accepted", label="observation", importance=9
        ),
        session_id="monologue",
    )
    path = persona_dir / "soul_candidates.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    decision = Decision(
        candidate_id=memory_id,
        decision="accept",
        confidence=9,
        reasoning="test accept",
        why_it_matters="test",
    )
    soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        crystallization_id = _apply_accept(records[0], decision, soul_store, dry_run=False)
    finally:
        soul_store.close()
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    assert crystallization_id is not None
    return crystallization_id


def test_run_pass_exempts_soul_crystallised_memory_through_real_files(tmp_path):
    """CANARY (#362) — pins run_pass → _load_soul_linked_ids → the crystallised
    leg against a REAL accepted candidate + a REAL active Crystallization.

    History: forgetting imported ``brain.soul.audit.list_crystallised_memory_ids``,
    which was never defined; the fail-soft ``except Exception`` swallowed the
    ImportError, so ``crystallised_ids`` was always empty and a memory the soul
    review accepted faded like any other. ``tests/forgetting/test_policy.py``
    passes ids in directly, so nothing composed this path.
    """
    memory_id = _old_low_salience_memory(tmp_path)
    _accept_candidate_for_real(tmp_path, memory_id)

    summary = run_pass(tmp_path, event_bus=MagicMock())

    store = MemoryStore(tmp_path / "memories.db")
    row = store._conn.execute("SELECT state FROM memories WHERE id = ?", (memory_id,)).fetchone()
    store.close()
    assert row["state"] == "active", (
        "a memory the soul review accepted must be exempt from forgetting — "
        "the run_pass→soul crystallised composition is broken"
    )
    assert summary["exempt"] >= 1
    assert summary["faded"] == 0


def _crystallise_directly(persona_dir, *, source_memory_id=None) -> str:
    """Make a crystallisation the way the model does: the real tool through the real ``dispatch``
    (the MCP server calls exactly this), against the persona's own memories.db."""
    from brain.memory.hebbian import HebbianMatrix
    from brain.tools.dispatch import dispatch

    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    try:
        args = {"moment": "a moment she chose", "love_type": "craft", "why_it_matters": "it mattered"}
        if source_memory_id is not None:
            args["source_memory_id"] = source_memory_id
        result = dispatch("crystallize_soul", args, store=store, hebbian=hebbian, persona_dir=persona_dir)
    finally:
        store.close()
        hebbian.close()
    assert result["created"] is True
    return result["id"]


def _state_of(persona_dir, memory_id) -> str:
    store = MemoryStore(persona_dir / "memories.db")
    try:
        return store._conn.execute("SELECT state FROM memories WHERE id = ?", (memory_id,)).fetchone()["state"]
    finally:
        store.close()


def test_run_pass_exempts_a_memory_named_by_a_direct_crystallisation_through_dispatch(tmp_path):
    """CANARY (#371) — the direct path end to end: dispatch("crystallize_soul", {..., source_memory_id})
    -> SoulStore link -> list_crystallised_memory_ids -> _load_soul_linked_ids -> run_pass is_exempt.

    Two personas of identical shape. The LINKED arm names its memory and nothing may fade (the frozen C1
    wording: faded == 0, exempt >= 1). The CONTROL arm crystallises WITHOUT naming a memory and its memory
    must fade, so a pass in the linked arm means the link did the protecting, not a gentle fixture."""
    linked_dir, control_dir = tmp_path / "linked", tmp_path / "control"
    linked_dir.mkdir()
    control_dir.mkdir()

    linked = _old_low_salience_memory(linked_dir)
    _crystallise_directly(linked_dir, source_memory_id=linked)
    summary = run_pass(linked_dir, event_bus=MagicMock())
    assert _state_of(linked_dir, linked) == "active", "a memory named by a direct crystallisation must be exempt"
    assert summary["faded"] == 0
    assert summary["exempt"] >= 1

    control = _old_low_salience_memory(control_dir)
    _crystallise_directly(control_dir)
    control_summary = run_pass(control_dir, event_bus=MagicMock())
    assert _state_of(control_dir, control) == "fading", "the unnamed control memory must still fade"
    assert control_summary["faded"] == 1


def test_revoking_a_direct_crystallisation_makes_its_named_memory_forgettable_again(tmp_path):
    """#371 C2 / L2-a: protection must be real BEFORE the revoke (otherwise 'fading afterwards' proves
    nothing), and revoke must end it. Same #362 ruling as for accepted candidates."""
    from brain.soul.revoke import revoke_crystallization
    from brain.soul.store import SoulStore

    memory_id = _old_low_salience_memory(tmp_path)
    crystallization_id = _crystallise_directly(tmp_path, source_memory_id=memory_id)

    run_pass(tmp_path, event_bus=MagicMock())
    assert _state_of(tmp_path, memory_id) == "active", "the link must protect the memory first"

    soul = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        assert revoke_crystallization(soul, crystallization_id, "no longer true") is not None
    finally:
        soul.close()

    run_pass(tmp_path, event_bus=MagicMock())
    assert _state_of(tmp_path, memory_id) == "fading", "a revoked crystallisation must stop protecting it"


def test_run_pass_forgets_memory_whose_crystallisation_was_revoked(tmp_path):
    """#362 decision (2026-10-13): a revoked crystallisation stops protecting its
    source memory — revoke is Hana ruling "this is not a soul moment", and the
    under-review docstring already says terminal statuses become forgettable.

    ``revoke_crystallization`` only touches crystallizations.db; the candidate
    record stays ``accepted`` — so the exemption has to consult the store.
    """
    from brain.soul.revoke import revoke_crystallization
    from brain.soul.store import SoulStore

    memory_id = _old_low_salience_memory(tmp_path)
    crystallization_id = _accept_candidate_for_real(tmp_path, memory_id)
    soul_store = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        assert revoke_crystallization(soul_store, crystallization_id, "no longer true") is not None
    finally:
        soul_store.close()

    run_pass(tmp_path, event_bus=MagicMock())

    store = MemoryStore(tmp_path / "memories.db")
    row = store._conn.execute("SELECT state FROM memories WHERE id = ?", (memory_id,)).fetchone()
    store.close()
    assert row["state"] == "fading", "a revoked crystallisation must not keep protecting the memory"


def test_load_soul_linked_ids_logs_instead_of_swallowing(tmp_path, monkeypatch, caplog):
    """#362: both fail-soft handlers stay fail-soft (empty set, no crash) but must
    say so — the silent swallow is what hid the dead crystallised leg."""
    import logging

    import brain.soul.candidates as candidates
    from brain.forgetting import _load_soul_linked_ids

    def _boom(_persona_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(candidates, "list_crystallised_memory_ids", _boom)
    monkeypatch.setattr(candidates, "list_under_review_memory_ids", _boom)

    with caplog.at_level(logging.WARNING, logger="brain.forgetting"):
        crystallised, under_review = _load_soul_linked_ids(tmp_path)

    assert (crystallised, under_review) == (set(), set())
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("crystallised" in m for m in messages), messages
    assert any("under-review" in m for m in messages), messages


def test_unreadable_candidates_file_is_logged_for_both_legs(tmp_path, caplog):
    """#370: a REAL read failure (a directory where soul_candidates.jsonl should
    be) must surface for the under-review leg too. list_under_review_memory_ids
    used to swallow it and return [], so under-review memories silently lost
    their exemption with nothing in the log."""
    import logging

    from brain.forgetting import _load_soul_linked_ids

    (tmp_path / "soul_candidates.jsonl").mkdir()

    with caplog.at_level(logging.WARNING, logger="brain.forgetting"):
        crystallised, under_review = _load_soul_linked_ids(tmp_path)

    assert (crystallised, under_review) == (set(), set())
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("crystallised" in m for m in messages), messages
    assert any("under-review" in m for m in messages), messages
