"""Real-model validation for ``brain.memory.reranker`` — #231.

NETWORK-ENABLED: downloads and runs the actual ``Xenova/ms-marco-MiniLM-L-
6-v2`` ONNX cross-encoder (~80MB, via fastembed) through the SAME
production path ``search_memories``/``semantic_recall`` use
(``brain.memory.reranker.build_reranker_provider().rerank(...)``).
``tests/unit/brain/memory/test_reranker.py`` covers the module's own logic
(floor gating, auto-scaling width, latency measurement, etc.) entirely
OFFLINE via ``FakeRerankerProvider`` — this file is the one place that
proves those offline assumptions (score shape, positional alignment,
"higher = more relevant", and the fixed ``RERANK_FLOOR`` actually
separating genuine matches from decoys) hold against the REAL model. A
shape+sanity guard, not a calibration suite — keep the sample small.

Marked BOTH ``@pytest.mark.requires_network`` (opts this test OUT of
``tests/conftest.py``'s autouse fake-embedding/fake-reranker-provider
fixtures, so the REAL ``build_reranker_provider()`` wiring runs — the same
convention ``test_embeddings.py``'s ``requires_network``-marked test uses)
AND ``@pytest.mark.integration`` (the marker actually deselected by this
project's conventional local pre-check gate, ``-m "not live and not
requires_claude_cli and not integration"`` — ``requires_network`` alone is
NOT part of that ``-m`` expression, so a test needs ``integration`` too to
be excluded from it; mirrors ``tests/unit/harness/test_dropin_integration.
py``, the other network-enabled test kept out of that gate the same way).
The real GitHub Actions CI (``.github/workflows/test.yml``) runs bare
``uv run pytest -v --tb=short`` with no ``-m`` filter at all, so this test
DOES run there (GitHub-hosted runners have real network) — only the local
pre-check convention needs the extra marker to skip it.

Run by hand with network enabled, e.g.:
    uv run pytest tests/unit/brain/memory/test_reranker_real_model.py -m requires_network -v -s
"""

from __future__ import annotations

import pytest

from brain.memory.reranker import build_reranker_provider
from brain.memory.semantic_recall import RERANK_FLOOR

pytestmark = [pytest.mark.requires_network, pytest.mark.integration]

_MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# (query, document) pairs. Reuses TEXT from the checked-in #88
# acceptance-case pairs where the real model agrees with the fake-reranker
# script (tests/unit/brain/chat/test_semantic_primary_recall.py's
# test_88_paraphrase_beats_keyword_overlap_decoy and
# tests/unit/brain/tools/test_search_memories_mode.py's
# test_mode_semantic_paraphrase_beats_keyword_overlap_decoy /
# test_default_mode_is_semantic) — NOT every pair from those files
# reproduces under the real model (a couple of their scripted-score pairs
# land on the other side of RERANK_FLOOR for real), so this file only
# reuses the subset independently confirmed against the real model below,
# plus a few new pairs for a small, reliable sample.
# ---------------------------------------------------------------------------

_GENUINE_PAIRS = [
    # reused from #88 / test_search_memories_mode.py — confirmed above floor
    ("how do I calm down when everything feels like too much",
     "deep breathing helps when you are feeling anxious"),
    ("quiet evening",
     "a quiet evening with nothing much happening"),
    ("what does Bob like to drink in the morning",
     "Bob always starts his day with a strong cup of black coffee"),
]

_DECOY_PAIRS = [
    # reused from #88 — a keyword-overlap-but-unrelated decoy, confirmed below floor
    ("too much of a flood of party invitations this week",
     "how do I calm down when everything feels like too much"),
    ("what's the capital of France",
     "my cat knocked a glass off the kitchen counter this morning"),
    ("how do I calm down when everything feels like too much",
     "the stock market closed higher today on tech earnings"),
]


@pytest.fixture(scope="module")
def real_provider():
    """Build the REAL provider once for this module (ONNX session
    construction + first-use model load has real, non-trivial cost) — the
    autouse ``_reset_reranker_provider_cache`` fixture in tests/conftest.py
    clears the PROCESS-wide cache before/after every test regardless, so
    this only avoids re-requesting the provider object between the
    assertions below, not the process-cache reset itself."""
    provider = build_reranker_provider()
    assert provider.model_id() == _MODEL_ID, (
        "expected the pinned #231 reranker model id — model_tier.py's "
        "TIER_RERANKER mapping changed out from under this test"
    )
    return provider


def test_real_reranker_score_shape_and_ordering(real_provider) -> None:
    """API-shape invariants the production code depends on: `.rerank(query,
    documents)` returns a `list[float]` (plain floats, not wrapper
    objects), POSITIONALLY aligned with the input document order (not
    pre-sorted), and a higher score means more relevant."""
    query = "how do I calm down when everything feels like too much"
    # Deliberately NOT in relevance order — a low-scoring decoy, then the
    # high-scoring genuine match, then another low-scoring decoy — so a
    # provider that silently sorted its output (instead of returning
    # positionally) would be caught by the position assertions below.
    docs = [
        "the stock market closed higher today on tech earnings",  # decoy
        "deep breathing helps when you are feeling anxious",  # genuine
        "too much of a flood of party invitations this week",  # decoy
    ]

    scores = real_provider.rerank(query, docs)

    assert isinstance(scores, list), "rerank() must return a list, not a lazy iterable/generator"
    assert len(scores) == len(docs), "one score per input document"
    for score in scores:
        assert isinstance(score, float), (
            f"expected a plain float, got {type(score)!r} ({score!r}) — production code "
            "(RERANK_FLOOR comparisons, sort keys) assumes plain float scores"
        )

    decoy_a_score, genuine_score, decoy_b_score = scores
    print(f"\nreal reranker scores (mixed batch): {list(zip(docs, scores, strict=True))}")

    # NOT pre-sorted: the genuine match stays at index 1 (its input
    # position), not moved to the front despite scoring highest.
    assert genuine_score > decoy_a_score, "higher score = more relevant (genuine beats decoy)"
    assert genuine_score > decoy_b_score, "higher score = more relevant (genuine beats decoy)"


def test_real_reranker_floor_separates_genuine_from_decoy(real_provider) -> None:
    """`RERANK_FLOOR` (the FIXED, empirically-set floor `_semantic_top_k`
    and `run_semantic_recall` gate on) must actually separate clearly-
    genuine matches from clearly-unrelated decoys under the REAL model —
    the whole premise #231 replaced the old per-persona cosine
    auto-calibration with. Small sample (3 + 3): a shape+sanity guard, not
    a calibration suite."""
    genuine_scores = []
    for query, doc in _GENUINE_PAIRS:
        (score,) = real_provider.rerank(query, [doc])
        genuine_scores.append((query, doc, score))

    decoy_scores = []
    for query, doc in _DECOY_PAIRS:
        (score,) = real_provider.rerank(query, [doc])
        decoy_scores.append((query, doc, score))

    print(f"\nRERANK_FLOOR = {RERANK_FLOOR}")
    print("genuine pairs:")
    for query, doc, score in genuine_scores:
        print(f"  {score:8.3f}  q={query!r} doc={doc!r}")
    print("decoy pairs:")
    for query, doc, score in decoy_scores:
        print(f"  {score:8.3f}  q={query!r} doc={doc!r}")

    for query, doc, score in genuine_scores:
        assert score >= RERANK_FLOOR, (
            f"genuine pair scored {score} below RERANK_FLOOR={RERANK_FLOOR} — "
            f"q={query!r} doc={doc!r}"
        )
    for query, doc, score in decoy_scores:
        assert score < RERANK_FLOOR, (
            f"decoy pair scored {score} at/above RERANK_FLOOR={RERANK_FLOOR} — "
            f"q={query!r} doc={doc!r}"
        )
