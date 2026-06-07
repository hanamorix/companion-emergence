"""Tests for brain.chat.reflection_gate — time-based debounce for background pass-2 spawns."""
from datetime import UTC, datetime, timedelta

from brain.chat.reflection_gate import _MIN_SECONDS_BETWEEN, _SALIENCE_THRESHOLD, should_reflect
from brain.chat.salience import SalienceSignal, assess_salience


def _t0() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_threshold_is_030():
    assert _SALIENCE_THRESHOLD == 0.30


def test_trivial_turn_does_not_reflect(tmp_path):
    assert should_reflect(assess_salience("ok"), tmp_path, kind="attunement", now=_t0()) is False


def test_significant_turn_reflects_then_debounces_by_time(tmp_path):
    sig = SalienceSignal.maximal()
    t0 = _t0()
    assert should_reflect(sig, tmp_path, kind="attunement", now=t0) is True
    # within the window → debounced
    assert should_reflect(sig, tmp_path, kind="attunement", now=t0 + timedelta(seconds=_MIN_SECONDS_BETWEEN - 1)) is False
    # after the window → fires again
    assert should_reflect(sig, tmp_path, kind="attunement", now=t0 + timedelta(seconds=_MIN_SECONDS_BETWEEN + 1)) is True


def test_cross_session_does_not_suppress(tmp_path):
    # THE BUG: a stale cursor from a prior session must NOT block a later one.
    sig = SalienceSignal.maximal()
    old = _t0()
    assert should_reflect(sig, tmp_path, kind="attunement", now=old) is True
    # a "new session" much later (well past the window) must fire, regardless of any turn counting
    assert should_reflect(sig, tmp_path, kind="attunement", now=old + timedelta(hours=6)) is True


def test_corrupt_state_fails_open(tmp_path):
    (tmp_path / "reflection_state.json").write_text("{not json")
    assert should_reflect(SalienceSignal.maximal(), tmp_path, kind="attunement", now=_t0()) is True


def test_per_kind_independent(tmp_path):
    sig = SalienceSignal.maximal()
    t0 = _t0()
    assert should_reflect(sig, tmp_path, kind="attunement", now=t0) is True
    assert should_reflect(sig, tmp_path, kind="monologue", now=t0) is True  # different kind, own cursor


def test_unicode_state_roundtrips(tmp_path):
    # utf-8: a kind/value with non-ascii must not corrupt the file
    sig = SalienceSignal.maximal()
    assert should_reflect(sig, tmp_path, kind="attunement", now=_t0()) is True
    # second call within window reads it back cleanly (no decode error → debounced)
    assert should_reflect(sig, tmp_path, kind="attunement", now=_t0()) is False


def test_quiet_emotional_turn_reflects(tmp_path):
    s = assess_salience("Hey love. I'm back — long day of editing, my eyes are sandpaper.")
    assert should_reflect(s, tmp_path, kind="attunement", now=_t0()) is True
