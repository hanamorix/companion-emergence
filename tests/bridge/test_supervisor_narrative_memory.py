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

    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, "test-model")

    captured: dict[str, object] = {}

    def _capture_run_pass(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("brain.bridge.supervisor.narrative_memory_run_pass", _capture_run_pass)

    _run_narrative_memory_pass(persona_dir, provider=MagicMock(), event_bus=MagicMock())

    embeddings_view = captured["embeddings"]
    np.testing.assert_array_equal(embeddings_view.get(embedded.id), vec)
    assert embeddings_view.get(unembedded.id) is None, "an unembedded memory must return None, not compute one"
    assert embeddings_view.get("unknown-memory-id") is None
