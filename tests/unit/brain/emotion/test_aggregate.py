"""Tests for brain.emotion.aggregate."""

from __future__ import annotations

import itertools
import math
from datetime import UTC, datetime, timedelta

import pytest

from brain.emotion.aggregate import aggregate_state
from brain.emotion.state import EmotionalState
from brain.memory.store import Memory


def _mem(emotions: dict[str, float], age_hours: float = 0.0) -> Memory:
    return Memory(
        id=f"m-{emotions}-{age_hours}",
        content="x",
        memory_type="conversation",
        domain="us",
        created_at=datetime.now(UTC) - timedelta(hours=age_hours),
        emotions=dict(emotions),
    )


def test_aggregate_empty_returns_empty_state():
    result = aggregate_state([])
    assert isinstance(result, EmotionalState)
    assert result.emotions == {}


def test_aggregate_max_pools_per_emotion():
    memories = [
        _mem({"love": 6.0, "curiosity": 4.0}),
        _mem({"love": 8.0, "defiance": 3.0}),
    ]
    result = aggregate_state(memories)
    assert result.emotions["love"] == 8.0
    assert result.emotions["curiosity"] == 4.0
    assert result.emotions["defiance"] == 3.0


def test_aggregate_ignores_unknown_emotions_silently():
    memories = [_mem({"not_a_real_emotion": 9.0, "love": 5.0})]
    result = aggregate_state(memories)
    assert "love" in result.emotions
    assert "not_a_real_emotion" not in result.emotions


def test_dropped_unregistered_emotion_warns_once(caplog):
    import logging

    from brain.emotion.aggregate import _warned_unregistered

    _warned_unregistered.clear()  # isolate from other test runs
    mem = Memory.create_new(
        content="x",
        memory_type="note",
        domain="us",
        emotions={"warmth": 8.0},  # warmth = unregistered persona-extension
    )
    with caplog.at_level(logging.WARNING):
        aggregate_state([mem])
        aggregate_state([mem])  # second call must NOT re-warn
    warnings = [r for r in caplog.records if "warmth" in r.getMessage()]
    assert len(warnings) == 1, (
        f"expected exactly one warning for the dropped stored emotion, got {len(warnings)}"
    )


# --- spelling variants pool under their canonical name (#557) ---


def test_aggregate_pools_a_variant_spelling_under_its_canonical_name():
    mem = _mem({"comfort-seeking": 7.0})
    result = aggregate_state([mem])
    assert result.emotions == {"comfort_seeking": 7.0}
    # Read-only over the input: the stored spelling is not rewritten.
    assert mem.emotions == {"comfort-seeking": 7.0}


@pytest.mark.parametrize(
    ("first", "second"),
    [("comfort-seeking", "Comfort_Seeking"), ("comfort_seeking", "comfort-seeking")],
)
@pytest.mark.parametrize(
    ("first_value", "second_value", "expected"),
    [(7.0, 3.0, 7.0), (3.0, 7.0, 7.0), (5.0, 5.0, 5.0)],
)
def test_aggregate_two_variants_of_one_emotion_pool_to_the_max(
    first, second, first_value, second_value, expected
):
    # The pooling rule is the max per emotion (module docstring), so two spellings
    # of one emotion give one channel holding the larger value.
    memories = [_mem({first: first_value}), _mem({second: second_value})]
    assert aggregate_state(memories).emotions == {"comfort_seeking": expected}
    assert aggregate_state(list(reversed(memories))).emotions == {"comfort_seeking": expected}


def test_aggregate_blend_part_order_variants_share_one_channel():
    from brain.emotion import vocabulary

    for name in ("grief_love_blend", "calm_grief_love_blend"):
        vocabulary.register(
            vocabulary.Emotion(
                name=name,
                description="x",
                category="persona_extension",
                decay_half_life_days=14.0,
                intensity_clamp=10.0,
            )
        )
    try:
        two_part = [_mem({"love_grief_blend": 6.0}), _mem({"grief_love_blend": 4.0})]
        assert aggregate_state(two_part).emotions == {"grief_love_blend": 6.0}
        three_part = [_mem({"love_calm_grief_blend": 3.0}), _mem({"Calm-Grief-Love_blend": 8.0})]
        assert aggregate_state(three_part).emotions == {"calm_grief_love_blend": 8.0}
    finally:
        vocabulary._unregister("grief_love_blend")
        vocabulary._unregister("calm_grief_love_blend")


def test_aggregate_unknown_names_are_neither_kept_nor_merged():
    from brain.emotion import vocabulary
    from brain.emotion.vocabulary import canonical_name

    # "comfort" is a prefix look-alike of a registered name but is not registered itself.
    assert vocabulary.get("comfort") is None
    assert vocabulary.get("not-a-real-emotion") is None
    memories = [
        _mem(
            {
                "not-a-real-emotion": 9.0,
                "Not_A_Real_Emotion": 4.0,
                "comfort": 8.0,
                "comfort-seeking": 7.0,
            }
        )
    ]
    # Unregistered names stay dropped, none of them joins or removes the registered channel.
    assert aggregate_state(memories).emotions == {"comfort_seeking": 7.0}
    # Both spellings of an unknown name collapse to one key, distinct from the look-alike
    # and from every registered name.
    unknown = canonical_name("not-a-real-emotion")
    assert canonical_name("Not_A_Real_Emotion") == unknown
    assert unknown != canonical_name("comfort")
    assert vocabulary.get(unknown) is None


@pytest.mark.parametrize(
    ("emotions", "expected"),
    [
        (
            {"climax": 8.0, "comfort-seeking": 5.0},
            {"comfort_seeking": 7.0, "rest_need": 2.0},
        ),
        (
            {"Climax": 8.0, "arousal": 6.0},
            {"arousal": pytest.approx(1.2), "comfort_seeking": 2.0, "rest_need": 2.0},
        ),
    ],
)
def test_climax_reset_sees_a_variant_spelled_channel(emotions, expected):
    # The reset adds 2.0 to comfort_seeking and acts on climax; a variant spelling of
    # either must land on the same channel instead of leaving a second key beside it.
    assert aggregate_state([_mem(emotions)]).emotions == expected


# --- a value above its emotion's clamp is judged per stored spelling, before spellings merge (#557) ---

_OVER_RANGE_AND_TWO_IN_RANGE_SPELLINGS = [
    {"comfort-seeking": 12.0},
    {"comfort_seeking": 5.0},
    {"Comfort_Seeking": 3.0},
]


@pytest.mark.parametrize(
    "order",
    list(itertools.permutations(range(3))),
    ids=lambda order: "".join(str(i) for i in order),
)
def test_aggregate_keeps_the_in_range_spelling_when_another_is_out_of_range(order):
    # Dream memories store unclamped sums. The spelling whose strongest value is above the
    # clamp is skipped on its own, as before spellings shared a channel; it must not take
    # the in-range spellings of the same emotion down with it.
    memories = [_mem(_OVER_RANGE_AND_TWO_IN_RANGE_SPELLINGS[i]) for i in order]
    assert aggregate_state(memories).emotions == {"comfort_seeking": 5.0}


@pytest.mark.parametrize(
    ("over_value", "expected"),
    [(10.0, 10.0), (10.5, 4.0)],
    ids=["exactly-at-the-clamp-is-kept", "above-the-clamp-is-skipped"],
)
def test_aggregate_range_limit_is_inclusive_of_the_clamp(over_value, expected):
    memories = [_mem({"comfort-seeking": over_value}), _mem({"Comfort_Seeking": 4.0})]
    assert aggregate_state(memories).emotions == {"comfort_seeking": expected}


def test_aggregate_range_limit_is_the_emotions_own_clamp():
    from brain.emotion import vocabulary

    vocabulary.register(
        vocabulary.Emotion(
            name="tight_clamp",
            description="x",
            category="persona_extension",
            decay_half_life_days=14.0,
            intensity_clamp=5.0,
        )
    )
    try:
        # 6.0 is inside the module-wide 10.0 but above this emotion's own clamp of 5.0.
        memories = [
            _mem({"tight-clamp": 6.0}),
            _mem({"tight_clamp": 5.0}),
            _mem({"Tight_Clamp": 2.0}),
        ]
        assert aggregate_state(memories).emotions == {"tight_clamp": 5.0}
    finally:
        vocabulary._unregister("tight_clamp")


def test_aggregate_out_of_range_spelling_leaves_other_emotions_alone():
    memories = [_mem({"comfort-seeking": 12.0, "love": 4.0}), _mem({"Love": 2.0})]
    assert aggregate_state(memories).emotions == {"love": 4.0}


def test_aggregate_ignores_infinite_and_nan_values_without_raising():
    memories = [
        _mem({"comfort-seeking": math.inf}),
        _mem({"Comfort_Seeking": math.nan}),
        _mem({"comfort_seeking": 4.0}),
        _mem({"COMFORT_SEEKING": 2.0}),
    ]
    assert aggregate_state(memories).emotions == {"comfort_seeking": 4.0}
