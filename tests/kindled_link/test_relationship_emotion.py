from datetime import UTC, datetime

from brain.kindled_link import limits
from brain.kindled_link.relationship import (
    apply_peer_emotion,
    relationship_emotion_delta,
)
from brain.kindled_link.store import KindledLinkStore

NOW = datetime(2026, 6, 20, 12, 0, tzinfo=UTC)


def test_delta_is_vocab_filtered_and_small():
    d = relationship_emotion_delta(dominant_source="warmth")
    assert all(0.0 < v <= 0.2 for v in d.values())


def test_apply_scales_down_as_window_fills(tmp_path):
    s = KindledLinkStore(tmp_path / "k.db")
    applied = []
    for _ in range(20):
        d = apply_peer_emotion(s, "kid_a", {"tenderness": 0.1}, NOW)
        applied.append(sum(d.values()))
    # cumulative applied influence never exceeds the cap (+ a small epsilon)
    assert sum(applied) <= limits.PEER_EMOTION_WINDOW_CAP + 1e-6


def test_fully_capped_peer_yields_empty_delta(tmp_path):
    s = KindledLinkStore(tmp_path / "k.db")
    s.add_peer_emotion("kid_a", limits.PEER_EMOTION_WINDOW_CAP, NOW)
    assert apply_peer_emotion(s, "kid_a", {"tenderness": 0.1}, NOW) == {}


def test_instantaneous_cap_holds_across_decay_reload(tmp_path):
    # M3: the cap is a decay leaky-bucket BY DESIGN (anti-burst-domination: a peer
    # can never dominate her felt state at any instant). After partial decay a peer
    # may re-engage, but the INSTANTANEOUS accumulated influence is always <= cap.
    from datetime import timedelta
    s = KindledLinkStore(tmp_path / "k.db")
    t = NOW
    for i in range(40):
        t = NOW + timedelta(hours=i * 2)  # advance through the window repeatedly
        apply_peer_emotion(s, "kid_a", {"tenderness": 0.1}, t)
        assert s.get_peer_emotion_accumulated("kid_a", t) <= limits.PEER_EMOTION_WINDOW_CAP + 1e-6


def test_nan_magnitude_does_not_defeat_cap(tmp_path):
    # stage-6 Major: a NaN magnitude must NOT slip past the guard, be applied at
    # full strength, or leave the accumulator uncharged (unbounded repeat).
    s = KindledLinkStore(tmp_path / "k.db")
    assert apply_peer_emotion(s, "kid_a", {"tenderness": float("nan")}, NOW) == {}
    assert s.get_peer_emotion_accumulated("kid_a", NOW) == 0.0
    # repeat: still bounded
    apply_peer_emotion(s, "kid_a", {"tenderness": float("nan")}, NOW)
    assert s.get_peer_emotion_accumulated("kid_a", NOW) == 0.0


def test_emotion_delta_is_all_finite():
    d = relationship_emotion_delta(dominant_source="warmth")
    import math as _m
    assert all(_m.isfinite(v) for v in d.values())
