# tests/forgetting/test_corecall_decay_balance.py
"""W5 promotion gate — co-recall reinforcement is a recency effect, not a ratchet.

A heavily co-recalled cluster, once abandoned (its hebbian edges decay and are
GC'd, and the reach-recency lapses with the passage of lived time), MUST still
fade on the normal forgetting schedule. If it did not, co-recall reinforcement
would be pinning memories permanently — a real design failure against spec §3
(decay-subordinate by construction).

Why abandonment has TWO arms here: reaching for a cluster via search_memories
strengthens the hebbian co-recall edges (the W5 signal — `salience.hebbian`),
but the bare `store.search_text` underneath the tool ALSO bumps each hit's
monotone `recall_count` and `last_accessed_at` (the pre-existing recall/freshness
salience inputs). The W5 invariant is specifically that the *edges* don't ratchet
the memory permanently vivid — so a faithful abandonment must (a) decay+GC the
edges AND (b) let the reach-recency lapse (lived time passing since the last
reach), exactly as it would in lived use. With both, the cluster drops out of
'active' and fades, proving the edges were not the thing keeping it alive.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from brain.felt_time.state import FeltTimeState
from brain.felt_time.state import persist as persist_felt_time
from brain.forgetting import run_pass
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.tools.impls.search_memories import search_memories


def test_corecall_cluster_still_fades_after_abandonment(tmp_path):
    """A heavily co-recalled cluster, once abandoned, still fades on the normal
    schedule — co-recall reinforcement does not create a permanent attractor.
    This is the W5 promotion gate."""
    # Non-cold lived age so the recent-buffer exemption doesn't apply.
    persist_felt_time(
        FeltTimeState(lived_age_hours=200.0, last_tick_ts="2026-05-18T00:00:00+00:00"),
        tmp_path,
    )

    # --- Phase 1: build a 4-memory low-salience cluster (no emotion, no soul,
    # old enough to be forgetting-eligible) and reach for them together. ---
    store = MemoryStore(tmp_path / "memories.db")
    ids = []
    for i in range(4):
        m = Memory.create_new(
            content=f"henryk cluster memory {i}",
            memory_type="episodic",
            domain="chat",
            emotions={},  # low intrinsic salience: no emotion, no soul
        )
        # Mirror test_orchestrator.py's low-salience fixture seeding, but older
        # (20d) so the fade gate fires well clear of the recent-buffer window.
        object.__setattr__(m, "created_at", datetime.now(UTC) - timedelta(days=20))
        store.create(m)
        ids.append(m.id)
    store.close()

    heb = HebbianMatrix(tmp_path / "hebbian.db")
    anchor = ids[0]

    # Reach for them together via the tool path, repeatedly → edges strengthen.
    store = MemoryStore(tmp_path / "memories.db")
    for _ in range(8):
        search_memories("henryk", store=store, hebbian=heb, persona_dir=tmp_path)
    store.close()

    assert heb.activation_count(anchor) >= 1, "co-recall should have created edges"

    # --- Phase 2: ABANDON. ---
    # (a) Edges: decay every weight to zero, then GC them away.
    for _ in range(50):
        heb.decay_all(rate=0.5)
    pruned = heb.garbage_collect(threshold=0.01)
    assert pruned >= 1, "abandoned edges should be GC'd"
    assert heb.activation_count(anchor) == 0, "no edges remain after abandonment"
    heb.close()

    # (b) Reach-recency: lived time passes since the last reach, so the
    # recall/freshness recency signal lapses. Push last_accessed_at well past
    # the 30-lived-day freshness horizon — the reaches are now in the deep past.
    store = MemoryStore(tmp_path / "memories.db")
    old_iso = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    store._conn.execute("UPDATE memories SET last_accessed_at = ?", (old_iso,))
    store._conn.commit()
    store.close()

    # --- Phase 3: run forgetting; the abandoned cluster must fade. ---
    faded_or_lost = 0
    for _ in range(4):
        summary = run_pass(tmp_path, event_bus=MagicMock())
        faded_or_lost += summary["faded"] + summary["lost"]

    assert faded_or_lost >= 1, (
        "abandoned co-recalled cluster must still fade/lose — reinforcement is "
        "not a permanent attractor"
    )

    # The anchor must have decayed off 'active': either its row carries a decayed
    # state ('fading'), or the LOSE path hard-deleted the row entirely (lost → no
    # row, graveyard entry). A still-active anchor would mean the GC'd co-recall
    # edges had somehow pinned it — the failure the W5 gate guards against.
    store = MemoryStore(tmp_path / "memories.db")
    row = store._conn.execute(
        "SELECT state FROM memories WHERE id = ?", (anchor,)
    ).fetchone()
    store.close()

    if row is None:
        # Hard-deleted by LOSE — confirm it landed in the graveyard.
        from brain.forgetting import graveyard

        entries = graveyard.read_all(tmp_path)
        assert any(e["memory_id"] == anchor for e in entries), (
            "anchor row is gone but no graveyard entry — unexpected deletion path"
        )
    else:
        assert row["state"] in ("fading", "lost"), (
            f"anchor should have decayed off 'active', got state={row['state']!r}"
        )
