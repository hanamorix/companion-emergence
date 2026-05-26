from datetime import UTC, datetime, timedelta

import pytest

from brain.emotion.state import EmotionalState
from brain.engines import dream_seeds
from brain.memory.store import Memory
from brain.soul.crystallization import Crystallization


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


def _cryst(cid: str, *, moment: str, resonance: int, age_days: float = 0.0) -> Crystallization:
    return Crystallization(
        id=cid,
        moment=moment,
        love_type="devotion",
        why_it_matters="",
        crystallized_at=datetime(2026, 5, 26, tzinfo=UTC) - timedelta(days=age_days),
        resonance=resonance,
    )


def _mem_text(content: str) -> Memory:
    return Memory(
        id=f"id-{content[:8]}",
        content=content,
        memory_type="meta",
        domain="us",
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        emotions={},
        importance=0.0,
    )


def test_identity_congruence_zero_without_crystallizations():
    assert dream_seeds.identity_congruence(_mem_text("the writing desk at dawn"), []) == 0.0


def test_identity_congruence_resonance_weighted_max_over_overlap():
    # mem shares {writing, desk} with c_hi (Jaccard 1.0) and nothing with c_lo.
    mem = _mem_text("the writing desk")
    crysts = [
        _cryst("c_hi", moment="my writing desk", resonance=8),
        _cryst("c_lo", moment="hana laughing loudly", resonance=10),
    ]
    # max(1.0 * 8/10, 0.0 * 10/10) == 0.8
    assert dream_seeds.identity_congruence(mem, crysts) == 0.8


def test_identity_congruence_partial_overlap_jaccard():
    # mem tokens {writing, desk, dawn}; moment tokens {writing, desk} -> 2/3.
    mem = _mem_text("writing desk dawn")
    crysts = [_cryst("c1", moment="writing desk", resonance=10)]
    assert dream_seeds.identity_congruence(mem, crysts) == pytest.approx(2 / 3)


def test_identity_congruence_ignores_age():
    mem = _mem_text("the writing desk")
    old = [_cryst("c1", moment="writing desk", resonance=10, age_days=900)]
    # No time decay: an ancient high-resonance crystallization still boosts fully.
    assert dream_seeds.identity_congruence(mem, old) == 1.0


def test_grief_pull_zero_for_non_grief_memory():
    mem = _mem("a", emotions={"memory_grief": 9.0}, mtype="meta")
    assert dream_seeds.grief_pull(mem) == 0.0


def test_grief_pull_normalizes_grief_event_intensity():
    mem = _mem("g", emotions={"memory_grief": 8.0}, mtype="grief_event")
    assert dream_seeds.grief_pull(mem) == 0.8


def test_grief_pull_zero_when_no_grief_emotion():
    mem = _mem("g", emotions={}, mtype="grief_event")
    assert dream_seeds.grief_pull(mem) == 0.0


def test_refractory_penalty_one_when_recently_seeded():
    assert dream_seeds.refractory_penalty("a", ["x", "a", "y"]) == 1.0


def test_refractory_penalty_zero_when_not_recently_seeded():
    assert dream_seeds.refractory_penalty("a", ["x", "y"]) == 0.0


def test_refractory_penalty_zero_with_empty_history():
    assert dream_seeds.refractory_penalty("a", []) == 0.0


def test_composite_score_blends_terms():
    # importance 5.0/10 = 0.5; no crysts -> identity 0; grief_event 6.0 -> 0.6;
    # not recently seeded -> refractory 0. Total = 0.5 + 0.6 = 1.1
    mem = _mem("g", emotions={"memory_grief": 6.0}, importance=5.0, mtype="grief_event")
    score = dream_seeds.composite_score(mem, EmotionalState(), [], recent_seed_ids=[])
    assert score == 1.1


def test_composite_score_applies_refractory_penalty():
    mem = _mem("a", importance=10.0)  # 1.0 baseline
    score = dream_seeds.composite_score(mem, EmotionalState(), [], recent_seed_ids=["a"])
    # 1.0 - W_REFRACTORY(2.0) * 1.0 = -1.0
    assert score == -1.0
