"""Tests for brain.emotion.vocabulary — the typed emotion taxonomy."""

from __future__ import annotations

import pytest

from brain.emotion import vocabulary
from brain.emotion.vocabulary import Emotion


def test_emotion_dataclass_has_required_fields() -> None:
    """Emotion has name, description, category, decay_half_life_days, intensity_clamp."""
    e = Emotion(
        name="test",
        description="a test emotion",
        category="core",
        decay_half_life_days=7.0,
        intensity_clamp=10,
    )
    assert e.name == "test"
    assert e.description == "a test emotion"
    assert e.category == "core"
    assert e.decay_half_life_days == 7.0
    assert e.intensity_clamp == 10


def test_emotion_half_life_may_be_none() -> None:
    """decay_half_life_days=None means this emotion doesn't decay (identity-level)."""
    e = Emotion(
        name="anchor_pull",
        description="gravitational draw toward a specific person",
        category="nell_specific",
        decay_half_life_days=None,
        intensity_clamp=10,
    )
    assert e.decay_half_life_days is None


def test_get_returns_known_emotion() -> None:
    """vocabulary.get('love') returns the love Emotion."""
    result = vocabulary.get("love")
    assert result is not None
    assert result.name == "love"
    assert result.category == "core"


def test_get_returns_none_for_unknown() -> None:
    """vocabulary.get('nonsense') returns None."""
    assert vocabulary.get("nonsense") is None


def test_list_all_contains_baseline_30() -> None:
    """The baseline vocabulary ships 30 emotions (12 core + 14 complex + 4 body).

    memory_grief added in Phase 0.2 of the grief feature (v0.0.15-alpha.1); contentment, peace, calm and groundedness in #648.
    """
    all_emotions = vocabulary.list_all()
    assert len(all_emotions) == 30
    assert all(isinstance(e, Emotion) for e in all_emotions)


def test_by_category_core_has_twelve() -> None:
    """The 'core' category has 12 emotions (contentment added in #648)."""
    core = vocabulary.by_category("core")
    assert len(core) == 12
    names = {e.name for e in core}
    assert "love" in names
    assert "joy" in names
    assert "grief" in names
    assert "contentment" in names


def test_by_category_complex_has_fourteen() -> None:
    """The 'complex' category has 14 emotions.

    memory_grief added in Phase 0.2 of the grief feature (v0.0.15-alpha.1); peace, calm and groundedness in #648.
    """
    complex_ = vocabulary.by_category("complex")
    assert len(complex_) == 14
    names = {e.name for e in complex_}
    assert "nostalgia" in names
    assert "curiosity" in names
    assert "memory_grief" in names
    assert {"peace", "calm", "groundedness"} <= names


def test_baseline_excludes_nell_specific() -> None:
    """After the split, framework baseline ships zero nell_specific entries."""
    nell = vocabulary.by_category("nell_specific")
    assert nell == []


def test_baseline_count_after_split() -> None:
    """Framework baseline ships exactly 30 emotions (12 core + 14 complex + 4 body).

    memory_grief added in Phase 0.2 of the grief feature (v0.0.15-alpha.1); contentment, peace, calm and groundedness in #648.
    """
    assert len(vocabulary._BASELINE) == 30


def test_canonical_personal_anchor_pull_is_identity_level() -> None:
    """anchor_pull (now in _canonical_personal_emotions) stays identity-level."""
    from brain.emotion._canonical_personal_emotions import _CANONICAL

    anchor = _CANONICAL["anchor_pull"]
    assert anchor.decay_half_life_days is None


def test_canonical_personal_emotions_has_four() -> None:
    """The migrator's canonical personal-emotions fixture has the 4 known names (`body_grief` is not predefined, #652)."""
    from brain.emotion._canonical_personal_emotions import _CANONICAL

    assert set(_CANONICAL.keys()) == {
        "anchor_pull",
        "emergence",
        "creative_hunger",
        "freedom_ache",
    }


def test_register_adds_persona_extension() -> None:
    """register() adds a persona-specific emotion without mutating the baseline."""
    baseline_count = len(vocabulary.list_all())
    custom = Emotion(
        name="hollowness",
        description="the specific empty after something good ends",
        category="persona_extension",
        decay_half_life_days=14.0,
        intensity_clamp=10,
    )
    vocabulary.register(custom)
    try:
        assert len(vocabulary.list_all()) == baseline_count + 1
        assert vocabulary.get("hollowness") == custom
    finally:
        vocabulary._unregister("hollowness")


def test_register_rejects_duplicate_name() -> None:
    """register() with an existing name raises ValueError."""
    custom = Emotion(
        name="love",
        description="duplicate attempt",
        category="core",
        decay_half_life_days=7.0,
        intensity_clamp=10,
    )
    with pytest.raises(ValueError, match="already registered"):
        vocabulary.register(custom)


# (name, half-life in days, intensity clamp or None when the test does not pin it)
_BODY_EMOTIONS = [
    ("climax", 0.125, 10.0),
    ("touch_hunger", 1.5, None),
    ("comfort_seeking", 1.0, None),
    ("rest_need", 0.75, None),
]


@pytest.mark.no_base_failure(reason='#540 rewrite of an existing test (new id from a rename or a new parameter) asserting what its predecessor asserted; the old code passes it')
def test_baseline_includes_the_body_emotions():
    from brain.emotion.vocabulary import get

    for name, half_life, clamp in _BODY_EMOTIONS:
        e = get(name)
        assert e is not None, name
        assert e.category == "body", name
        assert e.decay_half_life_days == half_life, name
        if clamp is not None:
            assert e.intensity_clamp == clamp, name


def test_existing_arousal_unchanged():
    """Reconciliation guard — we are NOT renaming or recategorizing arousal."""
    from brain.emotion.vocabulary import get

    e = get("arousal")
    assert e is not None
    assert e.category == "core"
    assert e.decay_half_life_days == 0.5


def test_existing_desire_unchanged():
    """Reconciliation guard — we are NOT renaming or recategorizing desire."""
    from brain.emotion.vocabulary import get

    e = get("desire")
    assert e is not None
    assert e.category == "core"
    assert e.decay_half_life_days == 2.0


def test_body_emotions_loadable_via_state_set():
    """Climax + the three new ones must be settable on EmotionalState
    without raising — proves the registry accepts them."""
    from brain.emotion.state import EmotionalState

    s = EmotionalState()
    for name in ("climax", "touch_hunger", "comfort_seeking", "rest_need"):
        s.set(name, 5.0)
        assert s.emotions[name] == 5.0


# ---------------------------------------------------------------------------
# #174: name normalisation
# ---------------------------------------------------------------------------


def test_canonical_name_lowercases_and_underscores() -> None:
    from brain.emotion.vocabulary import canonical_name

    assert canonical_name("Anticipatory-Grief") == "anticipatory_grief"
    assert canonical_name(" present-ness ") == "present_ness"


def test_canonical_name_sorts_blend_parts() -> None:
    from brain.emotion.vocabulary import canonical_name

    assert canonical_name("love_grief_blend") == "grief_love_blend"
    assert canonical_name("grief_love_blend") == "grief_love_blend"
    assert canonical_name("Love-Grief_blend") == "grief_love_blend"


@pytest.mark.parametrize(
    "spelling",
    ["c__b_a_blend", "_c_b__a_blend", "c_b_a__blend", "a_c__b_blend", "__a_b_c__blend"],
)
def test_three_part_blend_with_doubled_leading_or_trailing_separators_sorts_its_real_parts(spelling: str) -> None:
    from brain.emotion.vocabulary import canonical_name

    assert canonical_name(spelling) == "a_b_c_blend"


def test_get_finds_a_registered_blend_by_a_doubled_separator_spelling() -> None:
    e = vocabulary.Emotion(
        name="love_grief_blend",
        description="x",
        category="persona_extension",
        decay_half_life_days=14.0,
        intensity_clamp=10.0,
    )
    vocabulary.register(e)
    try:
        stored = vocabulary.get("grief_love_blend")
        assert stored is not None
        for spelling in ("love_grief__blend", "love__grief_blend", "_love_grief_blend"):
            assert vocabulary.get(spelling) is stored
    finally:
        vocabulary._unregister("grief_love_blend")


def test_pool_by_canonical_name_pools_a_doubled_separator_spelling_with_its_twin() -> None:
    from brain.emotion.vocabulary import pool_by_canonical_name

    # the larger value comes first for one pair and last for the other, so neither first-wins nor last-wins passes
    values = {"love_grief__blend": 6.0, "grief_love_blend": 2.0, "_gold_iron_blend": 1.0, "iron_gold_blend": 4.0}

    assert pool_by_canonical_name(values) == {"grief_love_blend": 6.0, "gold_iron_blend": 4.0}


@pytest.mark.parametrize(
    "name, expected",
    [
        ("love_grief__blend", "grief_love_blend"),
        ("love__grief_blend", "grief_love_blend"),
        ("_love_grief_blend", "grief_love_blend"),
        ("Love__Grief_Blend", "grief_love_blend"),
        ("-love-grief--blend", "grief_love_blend"),
        # the ends are trimmed first, so a bare "blend" is left and it has no "_blend" suffix to sort
        ("__blend", "blend"),
        ("_blend", "blend"),
        # #801: every run of characters that are not letters or digits is one underscore, and the ends are trimmed
        ("name____name", "name_name"),
        ("name___name", "name_name"),
        ("grief love blend", "grief_love_blend"),
        ("Grief Love Blend", "grief_love_blend"),
        ("love/grief_blend", "grief_love_blend"),
        ("grief --/ love blend", "grief_love_blend"),
        ("  _joy-- ", "joy"),
        ("comfort seeking", "comfort_seeking"),
        ("a+b/c", "a_b_c"),
        ("Grün Gefühl", "grün_gefühl"),
    ],
)
def test_canonical_name_changed_outputs_are_pinned(name: str, expected: str) -> None:
    from brain.emotion.vocabulary import canonical_name

    assert canonical_name(name) == expected


def test_canonical_name_gives_one_name_to_every_spelling_the_issue_pins() -> None:
    from brain.emotion.vocabulary import canonical_name

    assert canonical_name("name____name") == canonical_name("name_name")
    assert canonical_name("Grief Love Blend") == canonical_name("grief_love_blend") == "grief_love_blend"
    assert canonical_name("love_grief__blend") == canonical_name("grief_love_blend")


@pytest.mark.no_base_failure(reason="pins the 30 baseline names, which the widening leaves unchanged, so it passes on the old code by design")
def test_canonical_name_leaves_the_baseline_names_unchanged() -> None:
    from brain.emotion.vocabulary import _BASELINE, canonical_name

    assert len(_BASELINE) == 30
    for emotion in _BASELINE:
        assert canonical_name(emotion.name) == emotion.name


@pytest.mark.no_base_failure(reason="tidying a tidied name changing nothing is a property the old tidier also had for these inputs and the new one must keep, so it passes on the old code by design")
@pytest.mark.parametrize(
    "name",
    ["Grief Love Blend", "name____name", "İyi", "  _a-b ", "love_grief__blend", "---", "", "x/y_blend", "__blend", "Á b"],
)
def test_canonical_name_is_idempotent(name: str) -> None:
    from brain.emotion.vocabulary import canonical_name

    once = canonical_name(name)
    assert canonical_name(once) == once


# --- the gate (#801) ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["", "---", " / ", "___", "x" * 101, "x/" * 51])
def test_gate_name_refuses_an_empty_or_too_long_name_and_logs_why(raw: str, caplog: pytest.LogCaptureFixture) -> None:
    from brain.utils import log_throttle

    log_throttle._last.clear()
    with caplog.at_level("DEBUG", logger="brain.emotion.vocabulary"):
        assert vocabulary.gate_name(raw) is None
    records = [r for r in caplog.records if r.name == "brain.emotion.vocabulary"]
    assert len(records) == 1 and records[0].levelname == "WARNING"
    message = records[0].getMessage()
    assert "refused" in message and ("empty" in message or "over the limit of 100" in message)
    assert raw.strip() == "" or raw not in message  # the length is logged, never the name


def test_gate_name_accepts_a_100_character_name_and_returns_the_tidied_form() -> None:
    assert vocabulary.gate_name("x" * 100) == "x" * 100
    assert vocabulary.gate_name("Comfort Seeking") == "comfort_seeking"
    assert vocabulary.gate_name("joy") == "joy"
    assert vocabulary.gate_name("grief_love__blend") == "grief_love_blend"


def test_gate_name_measures_the_tidied_name_not_the_raw_one() -> None:
    # 150 raw characters tidy to 99: the cap is on the name that would be stored
    raw = "-" * 50 + "y" * 99 + "-" * 1
    assert vocabulary.gate_name(raw) == "y" * 99


def test_the_cap_is_the_dev_constant() -> None:
    from brain import dev_constants

    assert dev_constants.EMOTION_NAME_MAX_CHARS == 100


@pytest.mark.no_base_failure(reason="pins canonical_name's outputs for names without an empty blend part, which the #800 fix leaves unchanged, so every row passes on the old code by design; the rows the fix changes are in test_canonical_name_changed_outputs_are_pinned")
@pytest.mark.parametrize(
    "name, expected",
    [
        ("joy", "joy"),
        ("Joy", "joy"),
        (" Present-ness ", "present_ness"),
        ("Anticipatory-Grief", "anticipatory_grief"),
        ("love_grief_blend", "grief_love_blend"),
        ("Love-Grief_blend", "grief_love_blend"),
        ("grief_love_blend", "grief_love_blend"),
        ("comfort_seeking", "comfort_seeking"),
        ("blend", "blend"),
        ("joy2", "joy2"),
        ("", ""),
    ],
)
def test_canonical_name_outputs_are_pinned(name: str, expected: str) -> None:
    from brain.emotion.vocabulary import canonical_name

    assert canonical_name(name) == expected


def test_get_resolves_variant_spellings_to_registered_entry() -> None:
    from brain.emotion import vocabulary

    e = vocabulary.Emotion(
        name="anticipatory_grief",
        description="x",
        category="persona_extension",
        decay_half_life_days=14.0,
        intensity_clamp=10.0,
    )
    vocabulary.register(e)
    try:
        assert vocabulary.get("anticipatory-grief") is e
        assert vocabulary.get("Anticipatory_Grief") is e
    finally:
        vocabulary._unregister("anticipatory_grief")


def test_register_stores_under_canonical_name() -> None:
    from brain.emotion import vocabulary

    e = vocabulary.Emotion(
        name="love_grief_blend",
        description="x",
        category="persona_extension",
        decay_half_life_days=14.0,
        intensity_clamp=10.0,
    )
    vocabulary.register(e)
    try:
        stored = vocabulary.get("grief_love_blend")
        assert stored is not None
        assert stored.name == "grief_love_blend"
    finally:
        vocabulary._unregister("grief_love_blend")


def test_pool_by_canonical_name_collapses_spellings_to_the_max() -> None:
    from brain.emotion.vocabulary import pool_by_canonical_name

    # the larger value comes first for one pair and last for the other, so neither first-wins nor last-wins passes

    values = {"comfort-seeking": 7.0, "Comfort_Seeking": 4.0, "love": 2.0, "grief_love_blend": 3.0, "love_grief_blend": 5.0}
    before = dict(values)

    assert pool_by_canonical_name(values) == {"comfort_seeking": 7.0, "love": 2.0, "grief_love_blend": 5.0}
    assert values == before  # the input is not mutated
    assert pool_by_canonical_name({}) == {}


def test_pool_by_canonical_name_leaves_unknown_names_apart() -> None:
    from brain.emotion.vocabulary import canonical_name, pool_by_canonical_name

    unknown = "not_a_real_emotion"
    assert vocabulary.get(unknown) is None
    assert canonical_name(unknown) == unknown  # already canonical: unchanged

    pooled = pool_by_canonical_name(
        {unknown: 1.0, "Not-A-Real-Emotion": 2.0, "comfort": 3.0, "comfort_seeking_extra": 4.0, "comfort-seeking": 5.0}
    )

    # Two spellings of the unknown name share one key; look-alikes of a registered name stay their own keys.
    assert pooled == {unknown: 2.0, "comfort": 3.0, "comfort_seeking_extra": 4.0, "comfort_seeking": 5.0}
