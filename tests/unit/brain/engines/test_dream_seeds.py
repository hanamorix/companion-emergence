from datetime import UTC, datetime

from brain.emotion.state import EmotionalState
from brain.engines import dream_seeds
from brain.memory.store import Memory


def _mem(mid: str, *, emotions=None, importance=0.0, mtype="meta") -> Memory:
    return Memory(
        id=mid,
        content=f"content-{mid}",
        memory_type=mtype,
        domain="us",
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        emotions=emotions or {},
        importance=importance,
    )


def test_emotional_congruence_sums_shared_emotions():
    mood = EmotionalState(emotions={"grief": 6.0, "joy": 1.0})
    mem = _mem("a", emotions={"grief": 4.0, "anger": 9.0})
    # only the shared emotion (grief) counts: 6.0 * 4.0
    assert dream_seeds.emotional_congruence(mem, mood) == 24.0


def test_emotional_congruence_zero_when_no_overlap():
    mood = EmotionalState(emotions={"grief": 6.0})
    mem = _mem("a", emotions={"joy": 5.0})
    assert dream_seeds.emotional_congruence(mem, mood) == 0.0


def test_mood_is_active_true_above_floor():
    mood = EmotionalState(emotions={"grief": 6.0})
    assert dream_seeds.mood_is_active(mood, floor=0.5) is True


def test_mood_is_active_false_when_neutral():
    assert dream_seeds.mood_is_active(EmotionalState(), floor=0.5) is False


def test_mood_is_active_false_below_floor():
    mood = EmotionalState(emotions={"grief": 0.3})
    assert dream_seeds.mood_is_active(mood, floor=0.5) is False
