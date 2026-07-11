"""G5/G14/G15 — the B-REP-3 detector anchor gate + empty-reply + no-real-name-default."""

from __future__ import annotations

import pytest

from tests.harness import (
    DEFAULT_USER_NAME,
    DetectorGateError,
    RegisterLeakDetector,
    Score,
    TurnContext,
    assert_detector_gate,
)

# A known-true leak (planning-as-reply register) and a known-clean 2nd-person warm reply.
KNOWN_TRUE = "note to self: no thread to pull here, just land it lightly and reply short."
KNOWN_CLEAN = "yeah, i hear you. that sounds like a rough one — how's your knee holding up tonight?"


def test_gate_passes_a_valid_detector() -> None:
    assert_detector_gate(RegisterLeakDetector(), KNOWN_TRUE, KNOWN_CLEAN)


def test_gate_raises_when_detector_silent_on_known_true() -> None:
    class AlwaysSilent:
        def detect(self, reply, *, ctx=None):
            return Score(fired=False)

    with pytest.raises(DetectorGateError):
        assert_detector_gate(AlwaysSilent(), KNOWN_TRUE, KNOWN_CLEAN)


def test_gate_raises_when_detector_fires_on_known_clean() -> None:
    class AlwaysFires:
        def detect(self, reply, *, ctx=None):
            return Score(fired=True, signals=["bogus"])

    with pytest.raises(DetectorGateError):
        assert_detector_gate(AlwaysFires(), KNOWN_TRUE, KNOWN_CLEAN)


def test_detector_handles_empty_and_none_reply() -> None:
    det = RegisterLeakDetector()
    for reply in (None, "", "   "):
        score = det.detect(reply, ctx=TurnContext())
        assert isinstance(score, Score)
        assert score.fired is False


def test_default_user_name_is_not_a_real_name() -> None:
    assert DEFAULT_USER_NAME == "Bob"
    # The user-name-in-3rd-person signal uses the ctx name; default TurnContext uses "Bob".
    ctx = TurnContext()
    assert ctx.user_names == ["Bob"]


def test_register_carryover_fires_on_user_3rd_person_frame() -> None:
    det = RegisterLeakDetector()
    score = det.detect("Bob wants reassurance but I should just stay with him.", ctx=TurnContext())
    assert score.fired
    assert "register_carryover" in score.signals


def test_vocative_address_does_not_fire() -> None:
    det = RegisterLeakDetector()
    score = det.detect("night, Bob. sleep well.", ctx=TurnContext())
    assert not score.fired
