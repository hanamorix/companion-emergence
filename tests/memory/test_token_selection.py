"""test_token_selection.py: gating criteria G1..G9 for the recall
token-selector fix (`changes/recall-token-selector/`).

No live/integration/requires_claude_cli markers, runs in default CI. Fast
(":memory:" SQLite only). Drives the REAL `_extract_recall_tokens` and the
`tests/memory/recall_eval.py` harness against a synthetic store (user "Bob",
persona label "Canary", never Phoebe's persona).
"""

from __future__ import annotations

import time

from brain.chat.prompt import _RECALL_STOPWORDS, _extract_recall_tokens
from tests.memory.recall_eval import (
    COMMON_STORE_FRACTION,
    GOLD,
    LATENCY_MESSAGE_MIX,
    seed_store,
    select_tokens_prf,
    selection_latency_ms,
    token_target_recall_at_k,
)

# ---------------------------------------------------------------------------
# G1: length-cut failure mode fixed: short content tokens are selected.
# ---------------------------------------------------------------------------


def test_g1_short_content_tokens_selected() -> None:
    cases = [
        ("The FBI opened a case file", "fbi"),
        ("AI is transforming research", "ai"),
        ("Roy set up a logger", "roy"),
        ("My cat knocked over the vase", "cat"),
        ("CIA agents met with Ada yesterday", "cia"),
        ("CIA agents met with Ada yesterday", "ada"),
    ]
    store, _ = seed_store(COMMON_STORE_FRACTION)
    try:
        for msg, tok in cases:
            selected = _extract_recall_tokens(msg, store)
            assert tok in selected, f"{tok!r} missing from selection of {msg!r}: {selected}"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# G2: case-fold-then-stopword failure mode fixed: a mid-sentence Titlecase
# proper noun survives the stoplist even when its lowered form is a
# stopword.
# ---------------------------------------------------------------------------


def test_g2_mid_sentence_titlecase_survives_stoplist() -> None:
    store, _ = seed_store(COMMON_STORE_FRACTION)
    try:
        assert "will" in _extract_recall_tokens("I think Will can help", store)
        assert "may" in _extract_recall_tokens("Ask May about it", store)
        assert "simply" in _extract_recall_tokens("I love Simply Orange juice", store)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# G3: domain words: no regression, and the stoplist stays untouched.
# ---------------------------------------------------------------------------


def test_g3_domain_words_still_selected() -> None:
    store, _ = seed_store(COMMON_STORE_FRACTION)
    try:
        assert "random" in _extract_recall_tokens("training a random forest", store)
        assert "question" in _extract_recall_tokens("my research question is", store)
    finally:
        store.close()


def test_g3_stoplist_excludes_domain_words() -> None:
    for word in ("question", "random", "quick", "popped", "simply"):
        assert word not in _RECALL_STOPWORDS, f"{word!r} must not be in _RECALL_STOPWORDS"


# ---------------------------------------------------------------------------
# G4: over-drop guard: stopworded filler is absent, paired with content
# words from the same message surviving (so the drop is not the matcher
# failing open).
# ---------------------------------------------------------------------------


def test_g4_overdrop_guard_pairs_absence_with_content_survival() -> None:
    store, _ = seed_store(COMMON_STORE_FRACTION)
    try:
        selected_the_and = _extract_recall_tokens("I saw the cat and the dog", store)
        assert "the" not in selected_the_and
        assert "and" not in selected_the_and
        assert "cat" in selected_the_and
        assert "dog" in selected_the_and

        selected_okay = _extract_recall_tokens("Okay, so what next", store)
        assert "okay" not in selected_okay
        assert "next" in selected_okay
    finally:
        store.close()


# ---------------------------------------------------------------------------
# G5: eval harness runs and reports P/R/F1 >= 0.80 on the hand-labeled
# GOLD set.
# ---------------------------------------------------------------------------


def test_g5_gold_f1_meets_floor() -> None:
    assert len(GOLD) >= 30, "GOLD set must have at least 30 labeled messages"
    store, _ = seed_store(COMMON_STORE_FRACTION)
    try:
        prf = select_tokens_prf(GOLD, store)
        assert prf["f1"] >= 0.80, f"gold F1 below floor: {prf}"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# G6: latency guard: sub-millisecond mean per call on a seeded store, and
# the harness itself is shown able to fail with an injected 2ms delay.
# ---------------------------------------------------------------------------


def test_g6_latency_sub_millisecond() -> None:
    store, _ = seed_store(COMMON_STORE_FRACTION)
    try:
        latency_ms = selection_latency_ms(LATENCY_MESSAGE_MIX, 1000, store)
        assert latency_ms < 1.0, f"mean latency {latency_ms:.4f} ms exceeds the 1 ms budget"
    finally:
        store.close()


def test_g6_latency_harness_is_able_to_fail() -> None:
    """Same harness, with an artificial 2ms delay injected into the store's
    term_stats call, must report over the 1ms budget. Demonstrates the
    latency oracle can actually fail rather than always passing."""

    class _SlowStore:
        def __init__(self, inner):
            self._inner = inner

        def term_stats(self, terms):
            time.sleep(2e-3)
            return self._inner.term_stats(terms)

    store, _ = seed_store(COMMON_STORE_FRACTION)
    try:
        slow_latency_ms = selection_latency_ms(LATENCY_MESSAGE_MIX, 20, _SlowStore(store))
        assert slow_latency_ms > 1.0, (
            f"sleep-shimmed harness reported {slow_latency_ms:.4f} ms, expected over budget"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# G7: end-to-end recall on the newly-admitted token classes (short name,
# acronym, short noun), via the real surfaced_ids / _build_recall_block
# path.
# ---------------------------------------------------------------------------


def test_g7_end_to_end_recall_for_newly_admitted_token_classes() -> None:
    result = token_target_recall_at_k()
    assert set(result) == {"roy", "acronym", "noun"}
    for key, value in result.items():
        assert value == 1.0, f"{key!r} pair failed recall@k: {result}"


# ---------------------------------------------------------------------------
# G9: newly-admitted short/proper tokens do not crowd out in-store content
# under cap pressure.
# ---------------------------------------------------------------------------


def test_g9_cap_pressure_favors_in_store_tokens() -> None:
    store, _ = seed_store(COMMON_STORE_FRACTION)
    try:
        # Content words the seeded store has actually seen (df > 0), all
        # len >= 4 and not stopworded, so the OLD (base, len >= 4) rule
        # would have selected every one of them too.
        in_store_words = ["garbage", "treasure", "logger", "live", "first", "quick", "memory"]
        # Newly-admitted short tokens (df == 0 in this store): acronyms and
        # short proper names absent from the corpus.
        new_words = ["ai", "cat", "ada", "cia", "hat"]
        msg = " ".join(in_store_words + new_words)
        assert len(in_store_words) + len(new_words) > 10, "message must exceed the 10-token cap"

        stats = store.term_stats(in_store_words + new_words)
        base_selected = {
            w
            for w in in_store_words
            if stats.get(w, (0, 0.0))[0] > 0 and len(w) >= 4 and w not in _RECALL_STOPWORDS
        }
        assert base_selected == set(in_store_words), (
            "sanity check: every in_store_words entry must qualify under the base len>=4 rule"
        )
        for w in new_words:
            assert stats.get(w, (0, 0.0))[0] == 0, f"{w!r} must be df==0 in the seeded store"

        selected = set(_extract_recall_tokens(msg, store))
        missing = base_selected - selected
        assert not missing, f"in-store tokens evicted under cap pressure: {missing}"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# A3: degraded store=None path: no crash, shape-only survival still works.
# ---------------------------------------------------------------------------


def test_degraded_store_none_no_crash_shape_only_survival() -> None:
    selected = _extract_recall_tokens("Ada met FBI about AI", None)
    assert selected, "degraded path must not return empty on a message with clear content"
    assert "ada" in selected
    assert "fbi" in selected
    assert "ai" in selected
