"""S195 / S199: the pure pieces of the paragraph selection that does not fit the
time budget: the word-boundary prefix and the cut to a token cap. The selection
order itself (all, paragraphs 1/3/5, the cut whole message; the first-ever
variant) is tested through `run_semantic_recall` in test_per_paragraph_recall.py.
"""

from __future__ import annotations

import pytest

from brain.memory import semantic_recall as sr
from brain.tools.impls import search_memories as tool


class _Probe:
    """Token counts = characters (the suite's fake unit) unless scripted."""

    def __init__(self, counter=len):
        self.counter = counter
        self.probed: list[str] = []

    def token_lengths_timed(self, texts):
        self.probed.extend(texts)
        return [self.counter(t) for t in texts], 0.01


def test_a_text_that_fits_is_returned_whole_with_no_probe() -> None:
    probe = _Probe()
    cut = sr._cut_to_tokens(probe, "alpha beta", 10, 10)
    assert cut == sr._Cut("alpha beta", 10, 0.0) and probe.probed == []


def test_the_cut_is_a_word_boundary_prefix_within_the_cap() -> None:
    probe = _Probe()
    text = "alpha beta gamma delta"
    cut = sr._cut_to_tokens(probe, text, len(text), 13)
    assert cut is not None and cut.text == "alpha beta" and cut.tokens == 10
    assert cut.probe_seconds == pytest.approx(0.01 * len(probe.probed))
    assert text.startswith(cut.text)


def test_a_prefix_ending_right_after_a_newline_keeps_the_whole_word_before_it() -> None:
    assert sr._word_prefix("abc def\nghi", 8) == "abc def"
    assert sr._word_prefix("abc def\nghi", 9) == "abc def"


def test_a_text_without_a_boundary_is_cut_by_characters() -> None:
    probe = _Probe()
    text = "x" * 100
    cut = sr._cut_to_tokens(probe, text, 100, 30)
    assert cut is not None and cut.text == "x" * 30 and cut.tokens == 30


def test_the_cut_shrinks_until_the_probe_agrees() -> None:
    """A tokenizer that is dense at the start of the text (3 tokens per character for
    the first 30 characters, then 1): the first proportional guess overshoots the cap
    and the loop shortens it strictly each step until the probe's count fits."""

    def dense_start(t: str) -> int:
        return len(t) + 2 * min(len(t), 30)

    probe = _Probe(dense_start)
    text = "word " * 20
    cut = sr._cut_to_tokens(probe, text, dense_start(text), 80)
    assert cut is not None and cut.tokens <= 80 and cut.tokens == dense_start(cut.text)
    lengths = [len(t) for t in probe.probed]
    assert len(lengths) >= 2, "the first guess overshot"
    assert lengths == sorted(lengths, reverse=True) and len(set(lengths)) == len(lengths)


@pytest.mark.parametrize("cap", [0, -3])
def test_nothing_fits_when_the_cap_is_below_one_token(cap: int) -> None:
    assert sr._cut_to_tokens(_Probe(), "alpha beta", 10, cap) is None


def test_a_failing_probe_cannot_size_a_cut() -> None:
    class _Broken:
        def token_lengths_timed(self, texts):
            raise RuntimeError("tokenizer down")

    assert sr._cut_to_tokens(_Broken(), "alpha beta gamma", 16, 5) is None


def test_the_search_tool_goes_through_the_same_paragraph_search() -> None:
    """F-C2: the tool shares `search_paragraphs`, so the selection above applies to it."""
    assert tool.search_paragraphs is sr.search_paragraphs
