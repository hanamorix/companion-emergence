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


def test_revise_writes_clamped_registered_emotion_memory(tmp_path: Path) -> None:
    from brain.memory.store import MemoryStore

    _seed_open_gap(tmp_path)
    # delta over the [-1, 1] bound — must be clamped to 1.0 → importance 10.
    result = reconcile_self_read(
        persona_dir=tmp_path, action="revise", channel="grief", delta=5.0
    )
    assert result.get("ok") is True
    assert result.get("delta_written") is True

    store = MemoryStore(tmp_path / "memories.db")
    try:
        mems = [m for m in store.list_active() if m.memory_type == "self_model_reconcile"]
    finally:
        store.close()
    assert len(mems) == 1
    # clamped to 1.0 → abs(1.0)*10 = 10.0 intensity on the grief channel
    assert mems[0].emotions.get("grief") == 10.0


def test_revise_off_vocab_channel_is_dropped(tmp_path: Path) -> None:
    from brain.memory.store import MemoryStore

    _seed_open_gap(tmp_path)
    result = reconcile_self_read(
        persona_dir=tmp_path, action="revise", channel="zorblefright", delta=0.5
    )
    # call succeeds but writes nothing — off-vocab name silently dropped
    assert result.get("ok") is True
    assert result.get("delta_written") is False

    store = MemoryStore(tmp_path / "memories.db")
    try:
        mems = [m for m in store.list_active() if m.memory_type == "self_model_reconcile"]
    finally:
        store.close()
    assert mems == []


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
