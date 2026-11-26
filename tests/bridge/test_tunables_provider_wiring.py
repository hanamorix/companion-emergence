"""Spec §5 canary 5: a value in the file actually changes the streaming loop's
timeout — through the provider path, not the getter in isolation."""
from __future__ import annotations

import pytest

from tests.tunables_file import write_values


@pytest.fixture()
def tunables_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as t

    t._reset_for_tests()
    yield tmp_path
    t._reset_for_tests()


def test_stream_timeouts_honor_overrides(tunables_home):
    write_values(tunables_home, {
        "provider.stream_per_event_idle_seconds": 240.0,
        "provider.stream_first_event_seconds": 300.0,
    })
    from brain.bridge import provider as p

    assert p._stream_per_event_idle_seconds() == 240.0
    assert p._stream_first_event_seconds() == 300.0


def test_stream_timeouts_default_without_file(tunables_home):
    from brain.bridge import provider as p

    assert p._stream_per_event_idle_seconds() == 60.0
    assert p._stream_first_event_seconds() == 120.0


def test_turn_budget_override_per_tier(tunables_home):
    write_values(tunables_home, {"provider.max_turn_budget_usd.sonnet": 2.50})
    from brain.bridge import provider as p

    assert p._MAX_TURN_BUDGET_USD("claude-sonnet-4-6") == 2.50
    assert p._MAX_TURN_BUDGET_USD("opus") == 3.00      # untouched tier
    assert p._MAX_TURN_BUDGET_USD("mystery-model") == 1.50  # fallback untouched
