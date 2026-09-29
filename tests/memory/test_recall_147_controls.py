"""test_recall_147_controls.py: criterion C12 (#147 control fixtures), name-recall
fix R4 (spec §5, S39, S50, S54, S68).

#147: does removing the 10-token cap and admitting short tokens make NORMAL-word
recall worse? The cutoff is what the block renders for the turn type (S68): 8 on
keyword-only turns (`SNIPPET_COUNT`), 9 when semantic results render.

Fixtures (`tests/memory/recall_147_fixtures.py`, synthetic, Bob/Canary, seed 147):
  - control set: 24 queries, targets reached through four ordinary words, each
    query > 10 salient tokens with 3 short tokens (acronym / 2-letter / digit)
    and 9 high-frequency filler words (C12 set (i));
  - rare-token set: 24 queries, the target reached through ONE rare word amid 12
    high-frequency words (the direction #147 fears, C12 set (iii) shape);
  - semantic-present: the same queries on a turn whose fixed conclusive semantic
    result is two unrelated memories, so keyword hits fill the leftover slots
    (C12 set (iv)).

BASE values below were measured 2026-09-29 by running these exact fixtures
against a worktree of `origin/main` `ed9b83ef` (the code with the cap in place,
and a conclusive semantic result suppressing the keyword search). Sets (ii) and
(iii) on the synthetic DB copy are the local harness `c12_syn_f.py`, not CI.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.chat.prompt import (
    _RECALL_STOPWORDS,
    _extract_recall_tokens,
    _legacy_capped_tokens,
)
from brain.memory.relevance import SNIPPET_COUNT
from brain.memory.semantic_recall import MAX_STANDOUT_COUNT
from tests.memory.recall_147_fixtures import (
    SHORT,
    add_semantic_decoys,
    build_control_set,
    recall_at,
    recall_at_with_semantic,
    tool_recall_at,
)

# Provenance: `origin/main` ed9b83ef, fixtures `build_control_set(seed=147, n_queries=24, ...)`.
BASE_CONTROL_AT_8 = 0.8333333333333334
BASE_CONTROL_AT_9 = 0.875  # reported only (ADV-8): recall@9 on a keyword-only set
BASE_RARE_AT_8 = 0.9583333333333334
BASE_RARE_NOSHORT_AT_8 = 1.0  # same rare-token queries with NO short tokens: isolates the cap removal
BASE_CONTROL_SEMANTIC_AT_9 = 0.0  # the base's conclusive semantic result suppresses keyword
BASE_RARE_SEMANTIC_AT_9 = 0.0
# The search TOOL's lexical mode (S81), same fixtures, same base: (limit) -> recall.
BASE_TOOL_CONTROL = {3: 1.0, 5: 1.0, 8: 1.0}
BASE_TOOL_RARE = {3: 0.875, 5: 0.9166666666666666, 8: 1.0}
_EPS = 1e-9

_KEYWORD_ONLY_CUTOFF = SNIPPET_COUNT  # 8: no semantic result renders (S68)
_SEMANTIC_PRESENT_CUTOFF = MAX_STANDOUT_COUNT  # 9: semantic results render (S68)


@pytest.fixture(scope="module")
def control():
    store, queries = build_control_set(seed=147, n_queries=24, rare_target=False)
    return store, queries, add_semantic_decoys(store)


@pytest.fixture(scope="module")
def rare():
    store, queries = build_control_set(seed=147, n_queries=24, rare_target=True)
    return store, queries, add_semantic_decoys(store)


@pytest.fixture(scope="module")
def rare_noshort():
    store, queries = build_control_set(seed=147, n_queries=24, rare_target=True, n_short=0)
    return store, queries, add_semantic_decoys(store)


def test_cutoffs_are_the_block_counts_for_the_turn_type() -> None:
    assert _KEYWORD_ONLY_CUTOFF == 8
    assert _SEMANTIC_PRESENT_CUTOFF == 9


def test_the_fixture_has_the_shape_the_criterion_requires(control, rare) -> None:
    for store, queries, _decoys in (control, rare):
        assert len(queries) >= 20
        for q in queries:
            tokens = _extract_recall_tokens(q.text, store)
            assert len(tokens) > 10, "the old ten-token cap must bind on every query"
            assert any(t.upper() in SHORT or t.isdigit() for t in tokens), (
                "each query carries a short token the admission change newly admits"
            )
            assert len(q.targets) == 1
    # targets are reached through ordinary words: >= 3 characters, no stopword
    store, queries, _ = control
    for q in queries:
        target_content = store._conn.execute(  # noqa: SLF001
            "SELECT content FROM memories WHERE id = ?", (next(iter(q.targets)),)
        ).fetchone()[0]
        shared = set(re.findall(r"[a-z]+", q.text.lower())) & set(re.findall(r"[a-z]+", target_content.lower()))
        content_words = {w for w in shared if len(w) >= 3 and w not in _RECALL_STOPWORDS and w != "bob"}
        assert len(content_words) >= 4


def test_control_recall_at_8_is_not_below_base(control, tmp_path: Path) -> None:
    store, queries, _ = control
    got = recall_at(store, queries, cutoff=_KEYWORD_ONLY_CUTOFF, persona_dir=tmp_path)
    assert got >= BASE_CONTROL_AT_8 - _EPS, f"recall@8 {got} < base {BASE_CONTROL_AT_8}"


def test_control_recall_at_9_advisory_adv8_does_not_regress(control, tmp_path: Path) -> None:
    store, queries, _ = control
    got = recall_at(store, queries, cutoff=_SEMANTIC_PRESENT_CUTOFF, persona_dir=tmp_path)
    assert got >= BASE_CONTROL_AT_9 - _EPS, f"recall@9 {got} < base {BASE_CONTROL_AT_9} (ADV-8)"


def test_semantic_present_recall_at_9_is_not_below_base(control, rare, rare_noshort, tmp_path: Path) -> None:
    # NOTE: the base's conclusive semantic result suppresses the keyword search, so the base
    # value is 0.0 by construction and "not below base" alone is weak; the > 0.5 line below
    # (keyword hits really fill the leftover slots) carries the set.
    for (store, queries, decoys), base in (
        (control, BASE_CONTROL_SEMANTIC_AT_9),
        (rare, BASE_RARE_SEMANTIC_AT_9),
        (rare_noshort, BASE_RARE_SEMANTIC_AT_9),
    ):
        got = recall_at_with_semantic(
            store, queries, decoys, cutoff=_SEMANTIC_PRESENT_CUTOFF, persona_dir=tmp_path
        )
        assert got >= base - _EPS
    # keyword hits DO fill the slots the semantic decoys leave (base: none do)
    store, queries, decoys = control
    assert recall_at_with_semantic(
        store, queries, decoys, cutoff=_SEMANTIC_PRESENT_CUTOFF, persona_dir=tmp_path
    ) > 0.5


def test_rare_token_recall_without_short_tokens_is_not_below_base(rare_noshort, tmp_path: Path) -> None:
    """Isolation: the same rare-word-amid-12-common-words queries with NO short
    token do not regress once the cap is gone, so the failure below is the short
    tokens' competition, not the high-frequency filler."""
    store, queries, _ = rare_noshort
    got = recall_at(store, queries, cutoff=_KEYWORD_ONLY_CUTOFF, persona_dir=tmp_path)
    assert got >= BASE_RARE_NOSHORT_AT_8 - _EPS, f"recall@8 {got} < base {BASE_RARE_NOSHORT_AT_8}"


def test_rare_token_recall_at_8_is_not_below_base(rare, tmp_path: Path) -> None:
    """C12 set (iii) shape. It FAILED for the plain uncapped single query (recall@8
    0.875 vs base 0.958 here; 0.667 vs 0.833 on the synthetic DB copy) because
    mid-frequency short tokens then competed with the one rare word; S79's tiers
    (today's capped query first, the rest only in leftover slots) fix it."""
    store, queries, _ = rare
    got = recall_at(store, queries, cutoff=_KEYWORD_ONLY_CUTOFF, persona_dir=tmp_path)
    assert got >= BASE_RARE_AT_8 - _EPS, f"recall@8 {got} < base {BASE_RARE_AT_8}"


def test_the_plain_uncapped_single_query_scores_below_base_on_the_rare_set(rare, tmp_path: Path) -> None:
    """Able to fail (oracle rule): the fixture discriminates. The build this
    increment started from, ONE uncapped query over every token, scores below
    base on the rare-token set at its own cutoff; the tiered search does not."""
    store, queries, _ = rare
    with patch("brain.chat.prompt._keyword_tiers", lambda tokens, legacy: (tokens, [])):
        got = recall_at(store, queries, cutoff=_KEYWORD_ONLY_CUTOFF, persona_dir=tmp_path)
    assert got < BASE_RARE_AT_8 - _EPS


def test_a_capped_selector_with_short_token_admission_scores_below_base_on_the_control_set(
    control, tmp_path: Path
) -> None:
    """Able to fail, second variant: keeping the old cap while the store admits
    the short tokens in the one query (no tier split) scores below the base on
    the control set."""
    real = _extract_recall_tokens

    def capped(user_input: str, store=None) -> list[str]:
        return _legacy_capped_tokens(real(user_input, store))

    store, queries, _ = control
    with (
        patch("brain.chat.prompt._extract_recall_tokens", capped),
        patch("brain.chat.prompt._keyword_tiers", lambda tokens, legacy: (tokens, [])),
    ):
        got = recall_at(store, queries, cutoff=_KEYWORD_ONLY_CUTOFF, persona_dir=tmp_path)
    assert got < BASE_CONTROL_AT_8 - _EPS


@pytest.mark.parametrize("limit", [3, 5, 8])
def test_tool_lexical_recall_is_not_below_base(control, rare, limit: int, tmp_path: Path) -> None:
    """#147 on the tool (S52, S81): the tool sends every word of a long query, and
    the 1-2 character words must only add (tier 2), never out-rank the rare word."""
    for (store, queries, _), base in ((control, BASE_TOOL_CONTROL), (rare, BASE_TOOL_RARE)):
        got = tool_recall_at(store, queries, limit=limit, persona_dir=tmp_path)
        assert got >= base[limit] - _EPS, f"tool recall@{limit} {got} < base {base[limit]}"


def test_a_single_untiered_tool_query_scores_below_base(rare, tmp_path: Path) -> None:
    """Able to fail: sending every word, short ones included, in ONE query (the
    tool as first built for S81) drops recall below the base on the rare set."""
    import brain.tools.impls.search_memories as tool  # noqa: PLC0415

    store, queries, _ = rare
    with patch.object(tool, "split_by_raw_query_floor", lambda words: (list(words), [])):
        got = tool_recall_at(store, queries, limit=3, persona_dir=tmp_path)
    assert got < BASE_TOOL_RARE[3] - _EPS
