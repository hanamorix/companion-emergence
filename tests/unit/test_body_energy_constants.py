"""The #650 body energy constants have exactly the values of the spec's section 6 table (the snapshot pins them as well)."""

from __future__ import annotations

from brain import dev_constants


def test_the_scalar_constants_have_the_spec_values() -> None:
    assert dev_constants.BODY_ENERGY_BASELINE == 6.5
    assert (dev_constants.BODY_ENERGY_MIN, dev_constants.BODY_ENERGY_MAX) == (1.0, 10.0)
    assert dev_constants.BODY_ENERGY_FUNNEL_PIVOT == 5.0
    assert dev_constants.BODY_ENERGY_STEP == 0.375
    assert dev_constants.BODY_ENERGY_HER_MESSAGE_WEIGHT == 1.0
    assert dev_constants.BODY_ENERGY_USER_MESSAGE_WEIGHT == 0.5
    assert dev_constants.BODY_ENERGY_DRIFT_STEP == 0.0985
    assert dev_constants.BODY_ENERGY_LENGTH_EXCESS_SHARE == 0.5
    assert dev_constants.BODY_ENERGY_MEAN_FLOOR == 1.0
    assert dev_constants.BODY_STATS_BUSIEST_COUNT == 3
    assert dev_constants.BODY_ENERGY_LENGTH_WINDOW_DAYS == 30


def test_the_three_feeling_lists_have_exactly_the_spec_members_and_do_not_overlap() -> None:
    lift = dev_constants.BODY_ENERGY_LIFT_FEELINGS
    drain = dev_constants.BODY_ENERGY_DRAIN_FEELINGS
    calm = dev_constants.BODY_ENERGY_CALM_FEELINGS
    assert lift == ("joy", "love", "gratitude", "belonging", "defiance")
    assert drain == ("grief", "memory_grief", "shame", "guilt", "loneliness", "jealousy", "touch_hunger", "rest_need")
    assert calm == ("peace", "calm", "contentment", "groundedness")
    assert len(set(lift) | set(drain) | set(calm)) == len(lift) + len(drain) + len(calm)
    # Named in the spec as on neither list.
    for name in ("anger", "nostalgia", "fear"):
        assert name not in lift + drain + calm
