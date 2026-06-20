from brain.kindled_link import limits


def test_limits_expose_caps_and_budget_constants():
    assert limits.DAILY_PROVIDER_CAP == 60
    assert limits.SESSION_MSG_CAP == 24
    assert limits.DAILY_OUTBOUND_CAP == 20
    assert limits.MIN_OUTBOUND_GAP_SECONDS == 60
    assert limits.SESSION_COOLDOWN_HOURS == 6
    assert limits.BUDGET_MAX == 1.0
    assert 0 < limits.BUDGET_REFILL_PER_DAY <= limits.BUDGET_MAX
    assert 0 < limits.BUDGET_TIGHTEN_THRESHOLD < limits.BUDGET_MAX
