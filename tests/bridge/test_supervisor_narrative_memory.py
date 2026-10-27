"""Integration tests: supervisor wires narrative_memory_run_pass into soul-review cadence.

Verifies two behaviours:

1. The narrative-memory pass fires on the soul-review cadence, AFTER the
   forgetting pass within the same cadence — order matters because forgetting
   drops memories first, then arc-update reads the surviving pool.
2. An exception raised by the narrative-memory wrapper is fault-isolated —
   the soul-review loop keeps running undisturbed.

Mirrors ``tests/bridge/test_supervisor_forgetting.py`` precedent (drives the
full ``run_folded`` loop with short cadences rather than a one-shot helper).

Entry point under test: ``brain.bridge.supervisor.run_folded``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from brain.bridge.supervisor import _run_narrative_memory_pass, run_folded
from brain.memory.store import Memory, MemoryStore


def test_supervisor_runs_arc_update_after_forgetting_on_soul_review_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One soul-review tick should call forgetting then arc-update, in that order."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    call_order: list[str] = []
    stop_event = threading.Event()

    def _fake_forgetting(persona_dir, *, event_bus, **_):
        call_order.append("forgetting")
        return {"faded": 0, "lost": 0, "total": 0, "exempt": 0, "unfaded": 0, "duration_ms": 0}

    def _fake_arc_update(*args, **kwargs):
        call_order.append("arc_update")
        if call_order.count("arc_update") >= 1:
            stop_event.set()

    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", _fake_forgetting)
    monkeypatch.setattr("brain.bridge.supervisor._run_narrative_memory_pass", _fake_arc_update)
    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", lambda *a, **k: (0, 0))
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())
    # #154: voice-reflection (background-generative tier) now builds its own
    # real Sonnet-tier provider, and calls the LLM unconditionally before its
    # own evidence gate — a real-subprocess hazard on this bare tmp_path
    # persona (no persona_config.json); not about this test, neutralise it.
    monkeypatch.setattr(
        "brain.bridge.supervisor._run_voice_reflection_tick", lambda *a, **k: None
    )

    provider = MagicMock()
    event_bus = MagicMock()

    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=provider,
        event_bus=event_bus,
        tick_interval_s=0.05,
        heartbeat_interval_s=None,
        soul_review_interval_s=0.05,
        finalize_interval_s=None,
        # #154: interest_sweep now builds its own real Haiku-tier provider
        # (persona_dir here has no persona_config.json, so it would default to
        # a real ClaudeCliProvider and attempt a genuine subprocess call this
        # test isn't set up to handle) — disabled, out of scope for this
        # narrative-memory-focused test.
        interest_sweep_interval_s=None,
    )

    # Forgetting must precede arc_update inside the same cadence tick.
    assert "forgetting" in call_order
    assert "arc_update" in call_order
    forgetting_idx = call_order.index("forgetting")
    arc_idx = call_order.index("arc_update")
    assert forgetting_idx < arc_idx, (
        f"forgetting must run before arc_update; got order: {call_order}"
    )


@pytest.mark.slow(reason='2.2 s on windows in CI runs 40593062549 and 40596717161 (junit time: setup, call and teardown); not yet reduced')
def test_supervisor_arc_update_failure_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the narrative-memory wrapper raises, the supervisor loop keeps running."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    def _exploding_arc_update(*args, **kwargs):
        raise RuntimeError("synthetic arc-update failure")

    monkeypatch.setattr("brain.bridge.supervisor._run_narrative_memory_pass", _exploding_arc_update)
    monkeypatch.setattr(
        "brain.bridge.supervisor.forgetting_run_pass",
        lambda *a, **k: {
            "faded": 0,
            "lost": 0,
            "total": 0,
            "exempt": 0,
            "unfaded": 0,
            "duration_ms": 0,
        },
    )
    monkeypatch.setattr("brain.bridge.supervisor._run_heartbeat_tick", lambda *a, **k: None)
    monkeypatch.setattr("brain.bridge.supervisor.FeltTime", MagicMock())
    # #154: voice-reflection (background-generative tier) now builds its own
    # real Sonnet-tier provider, and calls the LLM unconditionally before its
    # own evidence gate — a real-subprocess hazard on this bare tmp_path
    # persona (no persona_config.json); not about this test, neutralise it.
    monkeypatch.setattr(
        "brain.bridge.supervisor._run_voice_reflection_tick", lambda *a, **k: None
    )

    soul_review_calls: list[int] = [0]
    stop_event = threading.Event()

    def _soul_review_counter(*a, **k):
        soul_review_calls[0] += 1
        if soul_review_calls[0] >= 2:
            stop_event.set()
        return 0, 0  # _run_soul_review_tick now returns (model_failures, eligible)

    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", _soul_review_counter)

    provider = MagicMock()
    event_bus = MagicMock()

    run_folded(
        stop_event,
        persona_dir=persona_dir,
        provider=provider,
        event_bus=event_bus,
        tick_interval_s=0.05,
        heartbeat_interval_s=None,
        soul_review_interval_s=0.05,
        finalize_interval_s=None,
        # #154: interest_sweep now builds its own real Haiku-tier provider
        # (persona_dir here has no persona_config.json, so it would default to
        # a real ClaudeCliProvider and attempt a genuine subprocess call this
        # test isn't set up to handle) — disabled, out of scope for this
        # narrative-memory-focused test.
        interest_sweep_interval_s=None,
    )

    # Soul-review kept running even though arc-update raised each tick.
    assert soul_review_calls[0] >= 2


def test_embeddings_adapter_reads_the_warm_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1 #259 step 3: the narrative_memory `EmbeddingsView` adapter
    (`_EmbeddingsByMemoryId`, nested inside `_run_narrative_memory_pass`)
    must be a PURE read off the warm matrix, keyed by memory_id directly —
    no more `store.get()` (which bumps recall_count) + `embeddings_cache.
    get_or_compute()` compute-on-miss. Verifies both halves of that
    contract directly against the real `_run_narrative_memory_pass`: an
    embedded row's vector comes back, and an unembedded (or wholly unknown)
    memory id returns None rather than triggering a compute."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    store = MemoryStore(persona_dir / "memories.db")
    embedded = Memory.create_new(content="has a vector", memory_type="event", domain="d")
    unembedded = Memory.create_new(content="no vector yet", memory_type="event", domain="d")
    store.create(embedded)
    store.create(unembedded)
    vec = np.full(384, 0.25, dtype=np.float32)
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (vec.tobytes(), "test-model", embedded.id),
    )
    store._conn.commit()  # noqa: SLF001
    store.close()

    from brain.bridge import model_tier

    monkeypatch.setitem(model_tier.LOCAL_MODEL_IDS, model_tier.TIER_EMBEDDING, "test-model")

    captured: dict[str, object] = {}

    def _capture_run_pass(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("brain.bridge.supervisor.narrative_memory_run_pass", _capture_run_pass)

    _run_narrative_memory_pass(persona_dir, provider=MagicMock(), event_bus=MagicMock())

    embeddings_view = captured["embeddings"]
    np.testing.assert_array_equal(embeddings_view.get(embedded.id), vec)
    assert embeddings_view.get(unembedded.id) is None, "an unembedded memory must return None, not compute one"
    assert embeddings_view.get("unknown-memory-id") is None


def test_narrative_pass_is_skipped_when_the_soul_exemptions_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """#392: the narrative pass shares the forgetting loader. With the exemption sets unreadable it must
    not build arcs from an empty set, and it must say so once rather than raise a traceback every tick."""
    import logging

    from brain.forgetting import SoulExemptionsUnavailableError

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    store = MemoryStore(persona_dir / "memories.db")
    store.create(Memory.create_new(content="a memory", memory_type="event", domain="d"))
    store.close()

    def _unavailable(_persona_dir):
        raise SoulExemptionsUnavailableError("soul-crystallised memory ids")

    monkeypatch.setattr("brain.forgetting._load_soul_linked_ids", _unavailable)
    ran: list[bool] = []
    monkeypatch.setattr(
        "brain.bridge.supervisor.narrative_memory_run_pass", lambda *a, **k: ran.append(True)
    )

    with caplog.at_level(logging.WARNING, logger="brain.bridge.supervisor"):
        _run_narrative_memory_pass(persona_dir, provider=MagicMock(), event_bus=MagicMock())

    assert ran == [], "the narrative pass must not run on an empty exemption set"
    assert any("narrative" in r.getMessage() and "skipping" in r.getMessage() for r in caplog.records)


def test_narrative_pass_is_skipped_when_the_real_exemption_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#392: the same skip, but through the REAL loader (a directory where soul_candidates.jsonl should
    be) rather than a patched one, so the wiring between the loader's error and this wrapper is exercised."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    store = MemoryStore(persona_dir / "memories.db")
    store.create(Memory.create_new(content="a memory", memory_type="event", domain="d"))
    store.close()
    (persona_dir / "soul_candidates.jsonl").mkdir()
    ran: list[bool] = []
    monkeypatch.setattr(
        "brain.bridge.supervisor.narrative_memory_run_pass", lambda *a, **k: ran.append(True)
    )

    _run_narrative_memory_pass(persona_dir, provider=MagicMock(), event_bus=MagicMock())

    assert ran == []
    assert not (persona_dir / "hebbian.db").exists(), "a skipped pass must not open the stores"


def test_a_stop_after_forgetting_skips_the_narrative_pass_and_keeps_the_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C3 (#456, D2): forgetting finishes its last item with the stop already requested. The narrative pass (model
    calls) must not start, the maintenance job reports PAUSED, and its cadence file is unchanged."""
    from brain.bridge import central_cadence, cli_throttle
    from brain.paths import cadence_state_path

    cli_throttle.reset()
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    cadence = cadence_state_path(persona_dir, "maintenance_cadence.json")
    cadence.parent.mkdir(parents=True, exist_ok=True)
    cadence.write_text('{"next_at": "2026-01-01T00:00:00+00:00"}', encoding="utf-8")
    before = cadence.read_bytes()
    stop_event = threading.Event()
    calls: list[str] = []
    decisions: list = []

    def _forgetting_whose_last_item_sees_the_stop(persona_dir, *, event_bus, should_pause, progress_out, **_):
        calls.append("forgetting")
        stop_event.set()  # requested while the last item ran; the pass then ends normally, not paused
        return {"faded": 0, "lost": 0, "total": 1, "exempt": 0, "unfaded": 0, "duration_ms": 0}

    real_pass = central_cadence.run_central_pass

    def _recording_pass(*a, **kw):
        out = real_pass(*a, **kw)
        decisions.extend(out)
        return out

    monkeypatch.setattr("brain.bridge.supervisor.forgetting_run_pass", _forgetting_whose_last_item_sees_the_stop)
    monkeypatch.setattr(
        "brain.bridge.supervisor._run_narrative_memory_pass", lambda *a, **k: calls.append("narrative")
    )
    monkeypatch.setattr("brain.bridge.supervisor.central_cadence.run_central_pass", _recording_pass)
    monkeypatch.setattr("brain.bridge.supervisor._run_soul_review_tick", lambda *a, **k: (0, 0))
    monkeypatch.setattr("brain.bridge.supervisor._run_voice_reflection_tick", lambda *a, **k: None)
    try:
        run_folded(
            stop_event,
            persona_dir=persona_dir,
            provider=MagicMock(),
            event_bus=MagicMock(),
            tick_interval_s=0.05,
            heartbeat_interval_s=None,
            soul_review_interval_s=3600.0,
            finalize_interval_s=None,
            interest_sweep_interval_s=None,
            initiate_review_interval_s=None,
            self_model_interval_s=None,
            compaction_interval_s=None,
            calibration_interval_s=None,
            judge_selftune_interval_s=None,
            clustering_interval_s=None,
            maker_enabled=False,
            notes_enabled=False,
            kindled_link_enabled=False,
        )
    finally:
        cli_throttle.reset()
    assert calls == ["forgetting"]
    assert ("maintenance", "paused") in {(d.job, d.action) for d in decisions}
    assert cadence.read_bytes() == before


def test_430_c9_merge_target_of_accepted_candidate_stays_out_of_open_arcs(tmp_path: Path) -> None:
    """#430 C9: the narrative pass shares the forgetting loader, so the merge target of an accepted soul
    candidate is exempt there too and the candidate sweep skips it (the treatment every crystallised
    memory already gets). The control row, joined to the same arc by the same hebbian edge, does join."""
    from datetime import UTC, datetime, timedelta

    from brain.memory.hebbian import HebbianMatrix
    from brain.narrative_memory.arc import Arc, ArcMember
    from brain.narrative_memory.state import ArcsState, load_or_recover, save_state
    from tests.forgetting.test_orchestrator import (
        _accept_candidate_for_real,
        _merge_a_candidate_into_a_target,
    )

    cand_id, target_id, control_id = _merge_a_candidate_into_a_target(tmp_path)
    _accept_candidate_for_real(tmp_path, cand_id)

    opened = (datetime.now(UTC) - timedelta(days=50)).isoformat()
    seed = ArcMember(memory_id="seed", joined_at_iso=opened, lived_age_at_join=1.0, salience_at_join=0.9)
    arc = Arc(
        id="arc_1",
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dreams.log.jsonl:1",
        seed_memory_ids=("seed",),
        title="tea",
        opened_at_iso=opened,
        lived_age_at_open=1.0,
        last_extended_at_iso=datetime.now(UTC).isoformat(),
        closed_at_iso=None,
        lived_age_at_close=None,
        members=(seed,),
    )
    save_state(tmp_path, ArcsState(open={"arc_1": arc}, last_pass_ts_iso=datetime.now(UTC).isoformat()))
    hebbian = HebbianMatrix(tmp_path / "hebbian.db")
    try:
        hebbian.set_edge_weight(target_id, "seed", 5.0)
        hebbian.set_edge_weight(control_id, "seed", 5.0)
    finally:
        hebbian.close()

    _run_narrative_memory_pass(tmp_path, provider=MagicMock(), event_bus=MagicMock())

    members = {m.memory_id for m in load_or_recover(tmp_path).open["arc_1"].members}
    assert control_id in members, "control must join, or the target's absence proves nothing"
    assert target_id not in members, "an exempt merge target must not join a new arc"
