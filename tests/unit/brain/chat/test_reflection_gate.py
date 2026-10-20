"""Tests for brain.chat.reflection_gate — time-based debounce for background pass-2 spawns."""
import logging
from datetime import UTC, datetime, timedelta

import pytest

from brain.chat.reflection_gate import _MIN_SECONDS_BETWEEN, _SALIENCE_THRESHOLD, should_reflect
from brain.chat.salience import SalienceSignal, assess_salience
from brain.utils import log_throttle

GATE_LOG = "brain.chat.reflection_gate"


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


@pytest.fixture
def fresh_throttle(monkeypatch):
    monkeypatch.setattr(log_throttle, "_last", {})
    monkeypatch.setattr(log_throttle, "_counts", {}, raising=False)


def _gate_records(caplog, needle):
    return [r for r in caplog.records if r.name == GATE_LOG and needle in r.getMessage()]


def test_low_salience_suppress_is_recorded_with_a_count(tmp_path, caplog, fresh_throttle):
    # #368: the suppress path used to leave no record. It is routine, so INFO
    # once per interval with a running count, DEBUG in between.
    with caplog.at_level(logging.DEBUG, logger=GATE_LOG):
        for _ in range(3):
            assert should_reflect(assess_salience("ok"), tmp_path, kind="attunement", now=_t0()) is False
    recs = _gate_records(caplog, "low salience")
    assert [r.levelno for r in recs] == [logging.INFO, logging.DEBUG, logging.DEBUG]
    assert "kind=attunement" in recs[0].getMessage()
    assert "3 since start" in recs[-1].getMessage()
    # Behaviour unchanged: a suppressed turn writes no state.
    assert not (tmp_path / "reflection_state.json").exists()


def test_too_soon_suppress_is_recorded_and_keeps_the_cursor(tmp_path, caplog, fresh_throttle):
    sig = SalienceSignal.maximal()
    t0 = _t0()
    assert should_reflect(sig, tmp_path, kind="attunement", now=t0) is True
    state_before = (tmp_path / "reflection_state.json").read_text()
    with caplog.at_level(logging.DEBUG, logger=GATE_LOG):
        assert should_reflect(sig, tmp_path, kind="attunement", now=t0 + timedelta(seconds=10)) is False
        assert should_reflect(sig, tmp_path, kind="attunement", now=t0 + timedelta(seconds=20)) is False
    recs = _gate_records(caplog, "too soon")
    assert [r.levelno for r in recs] == [logging.INFO, logging.DEBUG]
    assert "kind=attunement" in recs[0].getMessage()
    assert "2 since start" in recs[-1].getMessage()
    # Behaviour unchanged: a debounced turn does not move the cursor, so the
    # window still runs from t0.
    assert (tmp_path / "reflection_state.json").read_text() == state_before
    assert should_reflect(sig, tmp_path, kind="attunement", now=t0 + timedelta(seconds=_MIN_SECONDS_BETWEEN + 1)) is True


@pytest.mark.parametrize("offset_s", [None, 10])  # None: low salience; 10: too soon
def test_a_raising_suppress_record_cannot_flip_the_decision(tmp_path, monkeypatch, caplog, fresh_throttle, offset_s):
    """Regression (#368): the too-soon record sat inside the fail-open try, so a
    raising record turned a debounced turn into True, and on the low-salience path
    it propagated into the chat turn. A suppressed turn must stay False whatever
    the record does, and the record's failure must still be warned."""
    import brain.chat.reflection_gate as gate

    t0 = _t0()
    if offset_s is None:
        sig, now = assess_salience("ok"), t0
    else:
        sig, now = SalienceSignal.maximal(), t0 + timedelta(seconds=offset_s)
        assert should_reflect(sig, tmp_path, kind="attunement", now=t0) is True

    def _boom(*a, **kw):
        raise RuntimeError("record broke")

    monkeypatch.setattr(gate, "count_throttled", _boom)
    with caplog.at_level(logging.WARNING, logger=GATE_LOG):
        assert should_reflect(sig, tmp_path, kind="attunement", now=now) is False
    assert any(r.exc_info and r.exc_info[0] is RuntimeError for r in _gate_records(caplog, "record failed"))
