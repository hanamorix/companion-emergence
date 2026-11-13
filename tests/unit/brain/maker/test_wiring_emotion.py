# tests/unit/brain/maker/test_wiring_emotion.py
import pytest

from brain.emotion import vocabulary
from brain.maker.maker import Making
from brain.maker.wiring import making_emotion_delta


def test_emotion_delta_is_vocab_filtered_and_source_varied(monkeypatch):
    # registered vocab present
    import brain.chat.extractor as ext
    monkeypatch.setattr(ext, "_filter_to_registered", lambda d: {k: v for k, v in d.items() if k in {"tenderness", "satisfaction"}})
    d = making_emotion_delta(Making("elegy", "t", "c", "private", None), dominant_source="grief")
    assert set(d).issubset({"tenderness", "satisfaction"})
    assert all(v > 0 for v in d.values())


# ---- #720: a persona without "satisfaction" or "wonder" feels the closest baseline name instead ----



@pytest.fixture
def baseline_only():
    """Only the baseline names: neither satisfaction nor wonder is registered (as on a fresh persona)."""
    assert vocabulary.get("satisfaction") is None and vocabulary.get("wonder") is None


def test_baseline_persona_feels_pride_for_satisfaction(baseline_only):
    d = making_emotion_delta(Making("ode", "t", "c", "private", None), dominant_source="joy")
    assert d == pytest.approx({"pride": 0.22, "tenderness": 0.08})


def test_baseline_persona_feels_awe_for_wonder(baseline_only):
    d = making_emotion_delta(Making("ode", "t", "c", "private", None), dominant_source="dream")
    assert d == pytest.approx({"pride": 0.12, "tenderness": 0.08, "awe": 0.10})


@pytest.fixture
def grown_words():
    """A persona that has grown satisfaction and wonder (registered as extensions, then removed)."""
    names = ("satisfaction", "wonder")
    for name in names:
        assert vocabulary.get(name) is None
        vocabulary.register(
            vocabulary.Emotion(name=name, description="test extension", category="persona_extension", decay_half_life_days=7.0)
        )
    yield
    for name in names:
        vocabulary._unregister(name)


@pytest.mark.no_base_failure(reason="pins that a persona with her own words keeps them (#720); main keeps them too")
def test_a_persona_with_the_words_keeps_them(grown_words):
    d = making_emotion_delta(Making("ode", "t", "c", "private", None), dominant_source="dream")
    assert d == pytest.approx({"satisfaction": 0.12, "tenderness": 0.08, "wonder": 0.10})


def test_making_delta_live_is_ten_times():
    """B1 PR 3: once the engine is live, the making nudge is on the 0-10 scale, ten times the old values."""
    d = making_emotion_delta(Making("ode", "t", "c", "private", None), dominant_source="joy", live=True)
    # A baseline persona feels pride for satisfaction (#720).
    assert d == pytest.approx({"pride": 2.2, "tenderness": 0.8})
