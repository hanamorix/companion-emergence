from brain.kindled_link.relationship import (
    _is_grounded,
    _normalise,
)


def test_normalise_casefolds_and_collapses_whitespace():
    assert _normalise("  The   DOG  ") == "the dog"


def test_is_grounded_true_for_substring():
    assert _is_grounded("the dog rolled", "Earlier: The dog rolled over today.")


def test_is_grounded_false_for_absent_quote():
    assert _is_grounded("quote never said", "We talked about books.") is False


def test_reflection_prompt_fences_transcript_untrusted():
    from brain.kindled_link.relationship import _build_reflection_prompt
    p = _build_reflection_prompt(current_stage="stranger",
                                 transcript="peer: I value slow trust")
    assert "UNTRUSTED PEER TEXT" in p
    assert p.index("BEGIN UNTRUSTED") < p.index("I value slow trust") < p.index("END UNTRUSTED")
    assert "stranger" in p
    assert "one stage" in p.lower() or "at most one" in p.lower()


def test_is_grounded_rejects_trivial_short_quote():
    # stage-6 Minor: a 1-char / sub-12-char quote present anywhere must NOT ground.
    assert _is_grounded("a", "banana") is False
    assert _is_grounded("short", "this has short in it") is False
    # a real >=12-char grounded quote still grounds
    assert _is_grounded("the dog rolled over", "Earlier: The dog rolled over today.") is True
