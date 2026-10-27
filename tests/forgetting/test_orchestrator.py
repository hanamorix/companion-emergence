# tests/forgetting/test_orchestrator.py
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from brain.felt_time.state import FeltTimeState
from brain.felt_time.state import persist as persist_felt_time
from brain.forgetting import FORGETTING_STATE_FILENAME, run_pass
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


@pytest.mark.slow(reason='2.2 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
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


@pytest.mark.slow(reason='1.6 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
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


@pytest.mark.slow(reason='1.6 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
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


@pytest.mark.slow(reason='1.6 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
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


def test_load_soul_linked_ids_raises_and_logs_both_legs_when_they_cannot_be_read(
    tmp_path, monkeypatch, caplog
):
    """#392: an exemption set that cannot be read must not become an empty set. Both legs are still
    attempted and logged (so one look at the log shows everything that failed), then the loader raises."""
    import logging

    import brain.soul.candidates as candidates
    from brain.forgetting import SoulExemptionsUnavailableError, _load_soul_linked_ids

    def _boom(_persona_dir):
        raise RuntimeError("boom")

    monkeypatch.setattr(candidates, "list_crystallised_memory_ids", _boom)
    monkeypatch.setattr(candidates, "list_under_review_memory_ids", _boom)

    with caplog.at_level(logging.WARNING, logger="brain.forgetting"):
        with pytest.raises(SoulExemptionsUnavailableError):
            _load_soul_linked_ids(tmp_path)

    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("crystallised" in m for m in messages), messages
    assert any("under-review" in m for m in messages), messages


def test_unreadable_candidates_file_raises_and_is_logged_for_both_legs(tmp_path, caplog):
    """#370 / #392: a REAL read failure (a directory where soul_candidates.jsonl should be) hits both
    legs. Each is logged, and the loader raises instead of returning two empty sets."""
    import logging

    from brain.forgetting import SoulExemptionsUnavailableError, _load_soul_linked_ids

    (tmp_path / "soul_candidates.jsonl").mkdir()

    with caplog.at_level(logging.WARNING, logger="brain.forgetting"):
        with pytest.raises(SoulExemptionsUnavailableError):
            _load_soul_linked_ids(tmp_path)

    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("crystallised" in m for m in messages), messages
    assert any("under-review" in m for m in messages), messages


def _break_candidates_file(persona_dir, _memory_id, _monkeypatch):
    (persona_dir / "soul_candidates.jsonl").mkdir()


def _break_revoked_read(persona_dir, memory_id, monkeypatch):
    import sqlite3

    from brain.soul.store import SoulStore

    # An accepted candidate is what makes the revoked list get read at all.
    _accept_candidate_for_real(persona_dir, memory_id)

    def _boom(self):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(SoulStore, "list_revoked", _boom)


def _break_store_open(persona_dir, _memory_id, _monkeypatch):
    (persona_dir / "crystallizations.db").write_bytes(b"this is not a sqlite database")


def _break_under_review_leg(_persona_dir, _memory_id, monkeypatch):
    import brain.soul.candidates as candidates

    def _boom(_persona_dir):
        raise PermissionError("soul_candidates.jsonl")

    monkeypatch.setattr(candidates, "list_under_review_memory_ids", _boom)


@pytest.mark.parametrize(
    "break_it",
    [_break_candidates_file, _break_revoked_read, _break_store_open, _break_under_review_leg],
    ids=["candidates-file", "revoked-read", "store-open", "under-review-leg"],
)
def test_run_pass_changes_nothing_when_the_soul_exemptions_cannot_be_read(
    tmp_path, monkeypatch, caplog, break_it
):
    """CANARY (#392) — fail closed. If the exemption sets cannot be read, the pass must not run with an
    empty set: fading is reversible, losing is not, and 'could not read' is not 'nothing is exempt'.

    One failure injected at each place the load can fail. The memory is the shape forgetting fades; the
    control below shows it fades when the load is healthy and empty, so 'still active' here is the skip,
    not a gentle fixture."""
    import logging

    memory_id = _old_low_salience_memory(tmp_path)
    break_it(tmp_path, memory_id, monkeypatch)
    bus = MagicMock()

    with caplog.at_level(logging.WARNING, logger="brain.forgetting"):
        summary = run_pass(tmp_path, event_bus=bus)

    assert _state_of(tmp_path, memory_id) == "active"
    assert summary["faded"] == summary["unfaded"] == summary["lost"] == 0
    assert summary["exemptions_unavailable"] == 1
    # A skipped pass writes nothing: no counters persisted, no resume cursor left behind.
    assert not (tmp_path / FORGETTING_STATE_FILENAME).exists()
    published = [c.args[0] for c in bus.publish.call_args_list]
    assert any(e["type"] == "forgetting_pass" and e["exemptions_unavailable"] == 1 for e in published)
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("skipping this pass" in m for m in messages), messages


def test_run_pass_control_the_same_memory_fades_when_the_load_is_healthy_and_empty(tmp_path):
    """The control for the canary above: no exemptions at all, nothing broken -> the memory fades."""
    memory_id = _old_low_salience_memory(tmp_path)

    summary = run_pass(tmp_path, event_bus=MagicMock())

    assert _state_of(tmp_path, memory_id) == "fading"
    assert summary["faded"] == 1
    assert "exemptions_unavailable" not in summary
    # ...and a pass that runs does persist counters, which is what makes the canary's "not written" mean something.
    assert (tmp_path / FORGETTING_STATE_FILENAME).exists()


# --------------------------------------------------------------------------- #430
def _old_memory(store, content):
    m = Memory.create_new(content=content, memory_type="conversation", domain="us", emotions={})
    object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=40))
    store.create(m)
    return m.id


def _merge_a_candidate_into_a_target(persona_dir) -> tuple[str, str, str]:
    """Build a merge through the REAL gate: an old target row, a gated candidate merged into it by
    ``run_consolidation`` (stub merge classifier), and a control row shaped like a post-merge target
    (same age, one recall bump from ``store.get``, as the gate's own ``store.get`` gives the target).
    Returns (candidate_id, target_id, control_id). The candidate id never becomes a row."""
    from brain.engines.consolidation import Decision, run_consolidation
    from brain.memory.pending import PendingQueue

    persist_felt_time(
        FeltTimeState(lived_age_hours=100.0, last_tick_ts="2026-05-18T00:00:00+00:00"),
        persona_dir,
    )
    store = MemoryStore(persona_dir / "memories.db")
    try:
        target_id = _old_memory(store, "Bob likes tea")
        control_id = _old_memory(store, "Bob walked to the harbour")
        store.get(control_id)  # the same recall bump the gate gives the target
        cand = Memory.create_new(
            content="Bob likes tea with honey", memory_type="dream", domain="us", emotions={}
        )
        PendingQueue(persona_dir).enqueue(cand, source="x")
        res = run_consolidation(
            store,
            persona_dir=persona_dir,
            classifier=lambda c, _ctx: Decision(
                "merge", target_id=target_id, merged_content="Bob likes tea with honey"
            ),
        )
        assert res.merged == 1
        assert store.get(cand.id, bump=False) is None  # merged away: never a row
    finally:
        store.close()
    return cand.id, target_id, control_id


def _candidate_records(persona_dir) -> list[dict]:
    import json

    path = persona_dir / "soul_candidates.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_430_c1_accepted_candidate_merged_away_protects_its_merge_target(tmp_path):
    """#430 C1: the soul accepted a candidate whose memory the gate had merged into an existing row.
    The moment now lives in the merge target, so forgetting must exempt the target. The control row,
    shaped like a post-merge target but linked to nothing, fades in the same pass."""
    cand_id, target_id, control_id = _merge_a_candidate_into_a_target(tmp_path)
    _accept_candidate_for_real(tmp_path, cand_id)
    records = _candidate_records(tmp_path)
    assert [(r["memory_id"], r["status"]) for r in records] == [(cand_id, "accepted")]

    run_pass(tmp_path, event_bus=MagicMock())

    assert _state_of(tmp_path, control_id) == "fading", "control must fade, or the target's survival proves nothing"
    assert _state_of(tmp_path, target_id) == "active", "the merge target of an accepted candidate must be exempt"


def test_430_c2_candidate_under_review_merged_away_protects_its_merge_target(tmp_path):
    """#430 C2: same merge, the candidate still under review (auto_pending): the target is exempt."""
    from brain.ingest.soul_queue import queue_soul_candidate
    from brain.ingest.types import ExtractedItem

    cand_id, target_id, control_id = _merge_a_candidate_into_a_target(tmp_path)
    queue_soul_candidate(
        tmp_path,
        memory_id=cand_id,
        item=ExtractedItem(text="Bob likes tea with honey", label="observation", importance=8),
        session_id="monologue",
    )
    assert [r["status"] for r in _candidate_records(tmp_path)] == ["auto_pending"]

    run_pass(tmp_path, event_bus=MagicMock())

    assert _state_of(tmp_path, control_id) == "fading"
    assert _state_of(tmp_path, target_id) == "active", "the merge target of a candidate under review must be exempt"


def test_430_c3_revoked_crystallisation_does_not_protect_the_merge_target(tmp_path):
    """#430 C3 (#362 ruling): accepted, merged, then revoked -> the target is forgettable like the control."""
    from brain.soul.revoke import revoke_crystallization
    from brain.soul.store import SoulStore

    cand_id, target_id, control_id = _merge_a_candidate_into_a_target(tmp_path)
    crystallization_id = _accept_candidate_for_real(tmp_path, cand_id)
    assert [(r["memory_id"], r["status"]) for r in _candidate_records(tmp_path)] == [(cand_id, "accepted")]
    soul = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        assert revoke_crystallization(soul, crystallization_id, "no longer true") is not None
    finally:
        soul.close()

    run_pass(tmp_path, event_bus=MagicMock())

    assert _state_of(tmp_path, control_id) == "fading"
    assert _state_of(tmp_path, target_id) == "fading", "a revoked crystallisation must not protect its merge target"


def test_430_c4a_unreadable_merge_archive_fails_closed(tmp_path, caplog):
    """#430 C4a (#392 semantics): a directory at the archive path is a real OSError. The loader raises
    naming the merge-target leg, and run_pass changes no memory."""
    import logging

    from brain.engines.consolidation import _ARCHIVE_FILENAME
    from brain.forgetting import SoulExemptionsUnavailableError, _load_soul_linked_ids

    memory_id = _old_low_salience_memory(tmp_path)
    (tmp_path / _ARCHIVE_FILENAME).mkdir()

    with pytest.raises(SoulExemptionsUnavailableError, match="merge targets"):
        _load_soul_linked_ids(tmp_path)
    with caplog.at_level(logging.WARNING, logger="brain.forgetting"):
        summary = run_pass(tmp_path, event_bus=MagicMock())

    assert summary["exemptions_unavailable"] == 1
    assert summary["faded"] == summary["lost"] == 0
    assert _state_of(tmp_path, memory_id) == "active"
