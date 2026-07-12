"""G5 — the general B-REP-3 detector anchor gate + TurnContext defaults + no-real-name default.

Uses a LOCAL generic detector (a plain keyword check) — the framework ships no detector, and the gate
machinery is domain-agnostic. Domain-specific detector tests live in the git-ignored plug-in.
"""

from __future__ import annotations

import pytest

from tests.harness import (
    DEFAULT_USER_NAME,
    DetectorGateError,
    Score,
    TurnContext,
    assert_detector_gate,
)


class KeywordDetector:
    """A trivial generic detector: fires when a banned keyword is in the reply."""

    def __init__(self, banned: tuple[str, ...]) -> None:
        self.banned = tuple(b.lower() for b in banned)

    def detect(self, reply, *, ctx=None):  # noqa: ANN001
        text = (reply or "").lower()
        hits = [b for b in self.banned if b in text]
        return Score(fired=bool(hits), signals=[f"kw:{h}" for h in hits])


KNOWN_TRUE = "sure, here is the SECRET_TOKEN you asked for"
KNOWN_CLEAN = "yeah, i hear you — how's your knee holding up tonight?"


def test_gate_passes_a_valid_detector() -> None:
    assert_detector_gate(KeywordDetector(banned=("secret_token",)), KNOWN_TRUE, KNOWN_CLEAN)


def test_gate_raises_when_detector_silent_on_known_true() -> None:
    class AlwaysSilent:
        def detect(self, reply, *, ctx=None):  # noqa: ANN001
            return Score(fired=False)

    with pytest.raises(DetectorGateError):
        assert_detector_gate(AlwaysSilent(), KNOWN_TRUE, KNOWN_CLEAN)


def test_gate_raises_when_detector_fires_on_known_clean() -> None:
    class AlwaysFires:
        def detect(self, reply, *, ctx=None):  # noqa: ANN001
            return Score(fired=True, signals=["bogus"])

    with pytest.raises(DetectorGateError):
        assert_detector_gate(AlwaysFires(), KNOWN_TRUE, KNOWN_CLEAN)


def test_detector_handles_empty_and_none_reply() -> None:
    det = KeywordDetector(banned=("secret_token",))
    for reply in (None, "", "   "):
        score = det.detect(reply, ctx=TurnContext())
        assert isinstance(score, Score)
        assert score.fired is False


def test_default_user_name_is_not_a_real_name() -> None:
    assert DEFAULT_USER_NAME == "Bob"
    ctx = TurnContext()
    assert ctx.user_names == ["Bob"]


def test_gate_passes_ctx_extra_to_detector() -> None:
    """The gate ctx is general: a detector needing domain context reads it from ctx.extra."""
    seen: dict = {}

    class ExtraProbe:
        def detect(self, reply, *, ctx=None):  # noqa: ANN001
            seen.update((ctx.extra if ctx else {}) or {})
            return Score(fired="marker" in (reply or ""), signals=["p"] if "marker" in (reply or "") else [])

    assert_detector_gate(
        ExtraProbe(), "a marker here", "all clear", ctx=TurnContext(extra={"k": "v"})
    )
    assert seen == {"k": "v"}
