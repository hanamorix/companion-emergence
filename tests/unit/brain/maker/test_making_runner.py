import json

import pytest

from brain.maker.making_runner import make_and_wire
from brain.memory.store import MemoryStore
from brain.works.store import WorksStore


def _promote_pending(persona_dir):
    """Drain the consolidation gate with a promote-all classifier.

    `making` is a GATED type (Root-2 stopgap): make_and_wire enqueues the act-memory; it
    reaches memories.db only after a consolidation drain."""
    from brain.engines.consolidation import Decision, run_consolidation

    s = MemoryStore(persona_dir / "memories.db")
    try:
        run_consolidation(
            s, persona_dir=persona_dir, classifier=lambda _c, _ctx: Decision("new")
        )
    finally:
        s.close()


class _FakeProvider:
    def complete(self, prompt):  # match maker.make()'s call
        return json.dumps({"type": "vignette", "title": "Dusk", "content": "Light fell slow.",
                           "disposition": "eventual_share"})


def test_make_and_wire_persists_a_making(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memories.db")
    # charge sources file / state present
    from brain.maker.charge import MakerCharge, save_charge
    save_charge(tmp_path, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0))
    make_and_wire(persona_dir=tmp_path, store=store, provider=_FakeProvider(),
                  now=None)
    works = WorksStore(tmp_path / "works.db")
    # at least one making persisted
    assert works._conn.execute("SELECT COUNT(*) FROM works").fetchone()[0] == 1
    works.close()
    store.close()


def test_make_and_wire_writes_act_memory_with_emotion_for_nondiscard(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    from brain.maker.charge import MakerCharge, save_charge
    save_charge(tmp_path, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0))
    make_and_wire(persona_dir=tmp_path, store=store, provider=_FakeProvider(), now=None)
    # making is GATED: the act-memory enqueues, promoted on drain.
    assert not store.list_by_type("making", limit=10)
    store.close()
    _promote_pending(tmp_path)
    store2 = MemoryStore(tmp_path / "memories.db")
    mems = store2.list_by_type("making", limit=10)
    assert len(mems) == 1
    # the emotion delta was applied W8-style: seeded onto the act-memory
    assert mems[0].emotions
    store2.close()


class _DiscardProvider:
    def complete(self, prompt):
        return json.dumps({"type": "vignette", "title": "Trash", "content": "x",
                           "disposition": "discard"})


def test_make_and_wire_discard_moves_no_emotion(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    from brain.maker.charge import MakerCharge, save_charge
    save_charge(tmp_path, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0))
    make_and_wire(persona_dir=tmp_path, store=store, provider=_DiscardProvider(), now=None)
    # making is GATED: the act-memory enqueues, promoted on drain.
    assert not store.list_by_type("making", limit=10)
    store.close()
    _promote_pending(tmp_path)
    store2 = MemoryStore(tmp_path / "memories.db")
    mems = store2.list_by_type("making", limit=10)
    assert len(mems) == 1
    # discard moves nothing — empty-emotion act-memory only
    assert not mems[0].emotions
    store2.close()


def test_make_and_wire_gives_a_baseline_persona_pride_for_satisfaction(tmp_path):
    """#720: through the real path, a persona without "satisfaction" feels pride from a making, not nothing."""
    from brain.emotion import vocabulary
    from brain.maker.charge import MakerCharge, save_charge

    assert vocabulary.get("satisfaction") is None
    store = MemoryStore(tmp_path / "memories.db")
    save_charge(tmp_path, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0))
    make_and_wire(persona_dir=tmp_path, store=store, provider=_FakeProvider(), now=None)
    store.close()
    _promote_pending(tmp_path)
    store2 = MemoryStore(tmp_path / "memories.db")
    try:
        [made] = store2.list_by_type("making", limit=10)
    finally:
        store2.close()
    assert made.emotions == {"pride": 0.12, "tenderness": 0.08}


def test_make_and_wire_applies_the_accent_of_what_built_the_charge(tmp_path):
    """#721: a dream-built charge gives the dream accent (wonder, which a baseline persona feels as awe, #720)."""
    import pytest

    from brain.emotion import vocabulary
    from brain.maker.charge import MakerCharge, save_charge

    assert vocabulary.get("wonder") is None
    store = MemoryStore(tmp_path / "memories.db")
    save_charge(tmp_path, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0, emotion_part=9.0, soul_part=0.0, dream_part=90.0))
    make_and_wire(persona_dir=tmp_path, store=store, provider=_FakeProvider(), now=None)
    store.close()
    _promote_pending(tmp_path)
    store2 = MemoryStore(tmp_path / "memories.db")
    try:
        [made] = store2.list_by_type("making", limit=10)
    finally:
        store2.close()
    assert made.emotions == pytest.approx({"pride": 0.12, "tenderness": 0.08, "awe": 0.10})


# ---- emotion rework B1 PR 3: the making nudge is on the 0-10 scale once the engine is live ----


def _go_live(persona_dir, store):
    """A live persona with no state file applies nothing, so seed a memory first, then record the switch-over."""
    from datetime import UTC, datetime

    from brain.memory.store import Memory
    from tests.unit.brain.emotion._engine_support import make_live

    store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions={"joy": 5.0}))
    make_live(persona_dir, datetime.now(UTC))


def _made_emotions(tmp_path, *, live, ruin=False, progress=0):
    """`live` switches the persona over; `ruin` then leaves its state unreadable; `progress` is the x10 fix's note (B1 PR 7)."""
    from brain.maker.charge import MakerCharge, save_charge
    from tests.unit.brain.emotion._engine_support import ruin_the_state

    store = MemoryStore(tmp_path / "memories.db")
    store.set_x10_progress(progress)
    if live:
        _go_live(tmp_path, store)
    if ruin:
        ruin_the_state(tmp_path)
    save_charge(tmp_path, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0))
    make_and_wire(persona_dir=tmp_path, store=store, provider=_FakeProvider(), now=None)
    store.close()
    _promote_pending(tmp_path)
    store2 = MemoryStore(tmp_path / "memories.db")
    try:
        return store2.list_by_type("making", limit=10)[0].emotions
    finally:
        store2.close()


def test_make_and_wire_stores_the_nudge_ten_times_when_live(tmp_path):
    # A baseline persona feels pride for satisfaction (#720), so both names reach her at ten times the old values.
    assert _made_emotions(tmp_path, live=True) == pytest.approx({"pride": 1.2, "tenderness": 0.8})


def test_make_and_wire_keeps_todays_nudge_when_off(tmp_path):
    assert _made_emotions(tmp_path, live=False) == pytest.approx({"pride": 0.12, "tenderness": 0.08})


# ---- B1 PR 7: the scale follows the switch-over, then the x10 fix's progress (cut 2: a making goes to the pending queue) ----


def test_make_and_wire_is_on_the_engine_scale_when_switched_over_but_state_unreadable(tmp_path):
    # The queue step has started (progress 2), so nothing multiplies a queued value again: it must be 0-10 already.
    assert _made_emotions(tmp_path, live=True, ruin=True, progress=2) == pytest.approx({"pride": 1.2, "tenderness": 0.8})


@pytest.mark.no_base_failure(reason="B1 PR 7 pins the per-writer cut: a queued making stays small at progress 1; main does too")
def test_make_and_wire_keeps_todays_scale_while_the_x10_queue_step_has_not_started(tmp_path):
    # Progress 1: the rows step ran but the queue step has not, and it multiplies the queue, so a 0-10 making would be
    # multiplied again. This is where a single cut of 1 for every writer would be wrong (ThinkerOfThoughts, #633).
    assert _made_emotions(tmp_path, live=False, progress=1) == pytest.approx({"pride": 0.12, "tenderness": 0.08})
