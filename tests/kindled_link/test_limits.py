"""Relations between the kindled peer-link caps and budget constants (values live in brain/dev_constants.py).

The values themselves are guarded by tests/unit/test_dev_constants_snapshot.py. These are the orderings that must hold
whatever the values are: a refill that is positive and not larger than the whole budget, a tighten threshold strictly
inside it, and an inbound flood cap that is a positive whole number.
"""

from brain import dev_constants


def test_budget_orderings_hold():
    assert 0 < dev_constants.KINDLED_LINK_BUDGET_REFILL_PER_DAY <= dev_constants.KINDLED_LINK_BUDGET_MAX
    assert 0 < dev_constants.KINDLED_LINK_BUDGET_TIGHTEN_THRESHOLD < dev_constants.KINDLED_LINK_BUDGET_MAX


def test_inbound_flood_cap_is_a_positive_int():
    assert isinstance(dev_constants.KINDLED_LINK_INBOUND_FLOOD_CAP, int)
    assert dev_constants.KINDLED_LINK_INBOUND_FLOOD_CAP > 0
