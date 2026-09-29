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

from brain.chat.prompt import _RECALL_STOPWORDS, _extract_recall_tokens
from brain.memory.relevance import SNIPPET_COUNT
from brain.memory.semantic_recall import MAX_STANDOUT_COUNT
from tests.memory.recall_147_fixtures import (
    SHORT,
    add_semantic_decoys,
    build_control_set,
    recall_at,
    recall_at_with_semantic,
)

# Provenance: `origin/main` ed9b83ef, fixtures `build_control_set(seed=147, n_queries=24, ...)`.
BASE_CONTROL_AT_8 = 0.8333333333333334
BASE_CONTROL_AT_9 = 0.875  # reported only (ADV-8): recall@9 on a keyword-only set
BASE_RARE_AT_8 = 0.9583333333333334
BASE_CONTROL_SEMANTIC_AT_9 = 0.0  # the base's conclusive semantic result suppresses keyword
BASE_RARE_SEMANTIC_AT_9 = 0.0
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


def test_control_recall_at_9_is_reported_not_gated(control, tmp_path: Path) -> None:
    store, queries, _ = control
    got = recall_at(store, queries, cutoff=_SEMANTIC_PRESENT_CUTOFF, persona_dir=tmp_path)
    assert got >= BASE_CONTROL_AT_9 - _EPS, f"recall@9 {got} < base {BASE_CONTROL_AT_9} (ADV-8)"


def test_semantic_present_recall_at_9_is_not_below_base(control, rare, tmp_path: Path) -> None:
    for (store, queries, decoys), base in ((control, BASE_CONTROL_SEMANTIC_AT_9), (rare, BASE_RARE_SEMANTIC_AT_9)):
        got = recall_at_with_semantic(
            store, queries, decoys, cutoff=_SEMANTIC_PRESENT_CUTOFF, persona_dir=tmp_path
        )
        assert got >= base - _EPS
    # keyword hits DO fill the slots the semantic decoys leave (base: none do)
    store, queries, decoys = control
    assert recall_at_with_semantic(
        store, queries, decoys, cutoff=_SEMANTIC_PRESENT_CUTOFF, persona_dir=tmp_path
    ) > 0.5


@pytest.mark.xfail(
    strict=True,
    reason=(
        "C12 set (iii) shape FAILS: with the ten-token cap removed (S9/S52) a target reached "
        "through ONE rare word amid 12 high-frequency words is outranked by memories matching "
        "several of the common words (recall@8 0.875 vs base 0.958 here; 0.667 vs 0.833 on "
        "the synthetic DB copy, c12_syn_f.py). Question to Planning/owner in R4/5-build.md; "
        "remove this marker when a ruling lands and the gate is met."
    ),
)
def test_rare_token_recall_at_8_is_not_below_base(rare, tmp_path: Path) -> None:
    store, queries, _ = rare
    got = recall_at(store, queries, cutoff=_KEYWORD_ONLY_CUTOFF, persona_dir=tmp_path)
    assert got >= BASE_RARE_AT_8 - _EPS, f"recall@8 {got} < base {BASE_RARE_AT_8}"


def test_a_variant_admitting_every_raw_token_scores_below_base_on_a_set(rare, tmp_path: Path) -> None:
    """Able to fail (oracle rule): the fixtures discriminate. A selector that
    admits EVERY raw token, stopwords included, scores below the base on the
    rare-token set at its own cutoff."""

    def every_raw_token(user_input: str, store=None) -> list[str]:
        out: list[str] = []
        for m in re.finditer(r"[A-Za-z0-9]+", user_input):
            tok = m.group().lower()
            if tok not in out:
                out.append(tok)
        return out

    store, queries, _ = rare
    with patch("brain.chat.prompt._extract_recall_tokens", every_raw_token):
        got = recall_at(store, queries, cutoff=_KEYWORD_ONLY_CUTOFF, persona_dir=tmp_path)
    assert got < BASE_RARE_AT_8 - _EPS
