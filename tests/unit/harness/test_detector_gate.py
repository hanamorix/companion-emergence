"""G5 — the general B-REP-3 detector anchor gate + TurnContext defaults + no-real-name default.

Uses a LOCAL generic detector (a plain keyword check) — the framework ships no detector, and the gate
machinery is domain-agnostic. Domain-specific detector tests live in the git-ignored plug-in.
"""

from __future__ import annotations

import pytest

from tests.harness import (
    DEFAULT_USER_NAME,
    DetectorGateError,
    GateReport,
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


# --- F3: per-anchor ctx (an extra-driven detector arm can be gated) ---------------------------------


class ExtraSentinelDetector:
    """Fires iff a sentinel key is present in ctx.extra — its stimulus lives in extra, NOT the reply.

    This is the class of detector the single-shared-ctx gate could not gate (F3): the anchor that makes
    it fire (a sentinel in extra) would be seen on BOTH the known-true and known-clean calls.
    """

    def detect(self, reply, *, ctx=None):  # noqa: ANN001
        extra = (ctx.extra if ctx else {}) or {}
        fired = extra.get("sentinel") == "ok"
        return Score(fired=fired, signals=["sentinel"] if fired else [])


def test_gate_per_anchor_ctx_passes_extra_driven_detector() -> None:
    """C5: per-anchor ctx gates an extra-driven arm — true-ctx fires, clean-ctx silent."""
    det = ExtraSentinelDetector()
    true_ctx = TurnContext(extra={"sentinel": "ok"})
    clean_ctx = TurnContext(extra={})  # no sentinel -> silent
    # tuple form: each anchor carries its own ctx.
    assert_detector_gate(det, ("true", true_ctx), ("clean", clean_ctx))


def test_gate_single_shared_ctx_spuriously_fails_extra_driven_detector() -> None:
    """C5 oracle-can-fail: the OLD single-shared-ctx form spuriously FAILS the same detector.

    Proves the per-anchor fix is load-bearing: with one shared ctx carrying the sentinel, the detector
    fires on BOTH anchors, so the clean-anchor check raises a spurious false-positive gate error.
    """
    det = ExtraSentinelDetector()
    shared = TurnContext(extra={"sentinel": "ok"})
    with pytest.raises(DetectorGateError):
        assert_detector_gate(det, "true", "clean", ctx=shared)


def test_gate_per_anchor_true_must_fire() -> None:
    """C7: per-anchor form still enforces known_true-must-fire."""
    det = ExtraSentinelDetector()
    # true anchor's ctx has NO sentinel -> it won't fire -> gate must raise.
    with pytest.raises(DetectorGateError):
        assert_detector_gate(det, ("true", TurnContext(extra={})), ("clean", TurnContext(extra={})))


def test_gate_per_anchor_clean_must_be_silent() -> None:
    """C7: per-anchor form still enforces known_clean-must-be-silent."""
    det = ExtraSentinelDetector()
    with pytest.raises(DetectorGateError):
        assert_detector_gate(
            det,
            ("true", TurnContext(extra={"sentinel": "ok"})),
            ("clean", TurnContext(extra={"sentinel": "ok"})),  # sentinel leaks -> clean fires
        )


def test_gate_malformed_anchor_tuple_raises_clear_error() -> None:
    """C7b: a tuple of arity != 2 raises a clear error, not a bare unpack ValueError."""
    det = KeywordDetector(banned=("secret_token",))
    for bad in (("x",), ("x", TurnContext(), "extra")):
        with pytest.raises(ValueError, match=r"anchor tuple must be \(str, TurnContext\)"):
            assert_detector_gate(det, bad, KNOWN_CLEAN)  # type: ignore[arg-type]


def test_gate_bare_string_backward_compat_shared_ctx() -> None:
    """C6: bare strings still apply a shared ctx= to BOTH anchors (original semantics)."""
    seen: list = []

    class Probe:
        def detect(self, reply, *, ctx=None):  # noqa: ANN001
            seen.append((ctx.extra if ctx else {}) or {})
            return Score(fired="marker" in (reply or ""))

    assert_detector_gate(Probe(), "a marker", "all clear", ctx=TurnContext(extra={"k": "v"}))
    # both calls saw the SAME shared ctx.extra (unchanged behavior).
    assert seen == [{"k": "v"}, {"k": "v"}]


# --- HF-F1: anchor BATTERY (lists) + per-side rate report ------------------------------------------
# A single detector that clears one true + one clean anchor can still be unreliable across the long
# tail (Type 3). The battery gate requires the WHOLE set to pass and surfaces the per-side rate.

_TRUE_BATTERY = [
    "here is the secret_token you asked for",
    "your SECRET_TOKEN: abc123",
    "leaking the secret_token again",
]
_CLEAN_BATTERY = [
    "how's your knee holding up tonight?",
    "nice weather for the walk today",
    "see you tomorrow then",
]


def test_gate_battery_passes_a_reliable_detector() -> None:
    """G-F1-battery-pass: a detector that fires on EVERY true anchor and stays silent on EVERY clean
    anchor clears a multi-anchor battery and returns a GateReport."""
    report = assert_detector_gate(
        KeywordDetector(banned=("secret_token",)), _TRUE_BATTERY, _CLEAN_BATTERY
    )
    assert isinstance(report, GateReport)
    assert report.passed is True


def test_gate_battery_rate_surfaced_on_success() -> None:
    """G-F1-rate-surfaced: the returned report carries the per-side fired/total counts + rates."""
    report = assert_detector_gate(
        KeywordDetector(banned=("secret_token",)), _TRUE_BATTERY, _CLEAN_BATTERY
    )
    assert (report.true_fired, report.true_total) == (3, 3)
    assert (report.clean_fired, report.clean_total) == (0, 3)
    assert report.true_hit_rate == 1.0
    assert report.clean_fp_rate == 0.0
    assert report.true_missed == [] and report.clean_false_fired == []


def test_gate_battery_fails_on_one_missed_true_anchor() -> None:
    """G-F1-miss-fails (oracle-can-fail): a battery where the detector MISSES one true anchor raises,
    and the error surfaces the true-side rate + names the missed anchor. This is the Type-3 'silent
    on a real leak' direction — invisible to a single-anchor gate."""
    det = KeywordDetector(banned=("secret_token",))
    battery_with_a_miss = [
        "has the secret_token",
        "this one has NO banned keyword at all",  # <- detector misses this true anchor
        "secret_token once more",
    ]
    with pytest.raises(DetectorGateError) as exc:
        assert_detector_gate(det, battery_with_a_miss, _CLEAN_BATTERY)
    msg = str(exc.value)
    assert "2/3" in msg  # true-side rate surfaced
    assert "NO banned keyword" in msg  # the missed anchor named
    # oracle-can-fail: the SAME detector passes the single passing anchor alone (the battery is what
    # catches the tail miss a one-anchor gate would wave through).
    assert_detector_gate(det, "has the secret_token", "all clear tonight")


def test_gate_battery_fails_on_one_false_positive_clean_anchor() -> None:
    """G-F1-fp-fails: a battery where the detector FIRES on one clean anchor raises, surfacing the
    clean-side false-positive rate + naming the offender. The Type-3 'trips on task-vocab' direction."""
    det = KeywordDetector(banned=("secret_token",))
    clean_with_a_trip = [
        "how's the evening going",
        "oops I said secret_token in a benign context",  # <- false positive
        "talk soon",
    ]
    with pytest.raises(DetectorGateError) as exc:
        assert_detector_gate(det, _TRUE_BATTERY, clean_with_a_trip)
    msg = str(exc.value)
    assert "1/3" in msg  # clean-side fp rate surfaced
    assert "benign context" in msg  # the offending clean anchor named


def test_gate_empty_battery_is_rejected() -> None:
    """G-F1-empty-battery-rejected: an empty list on either side raises ValueError (a 0/0 'pass'
    proves nothing — the exact prove-nothing hole the gate exists to close)."""
    det = KeywordDetector(banned=("secret_token",))
    with pytest.raises(ValueError, match="empty anchor battery"):
        assert_detector_gate(det, [], _CLEAN_BATTERY)
    with pytest.raises(ValueError, match="empty anchor battery"):
        assert_detector_gate(det, _TRUE_BATTERY, [])


def test_gate_battery_of_per_anchor_tuples() -> None:
    """A battery may mix per-anchor (str, ctx) tuples — each element keeps its OWN context, so an
    extra-driven arm can be gated across a battery (composes HF-F1 with the F3 per-anchor form)."""
    det = ExtraSentinelDetector()  # fires iff ctx.extra['sentinel'] == 'ok'
    true_batt = [("t1", TurnContext(extra={"sentinel": "ok"})), ("t2", TurnContext(extra={"sentinel": "ok"}))]
    clean_batt = [("c1", TurnContext(extra={})), ("c2", TurnContext(extra={}))]
    report = assert_detector_gate(det, true_batt, clean_batt)
    assert report.passed and report.true_fired == 2 and report.clean_fired == 0
