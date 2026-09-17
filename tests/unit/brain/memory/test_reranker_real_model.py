"""Real-model validation for ``brain.memory.reranker`` — #231, swapped to the
multilingual jina reranker under #250 F2a inc1.

NETWORK-ENABLED: downloads and runs the actual ``jinaai/jina-reranker-v2-
base-multilingual`` ONNX cross-encoder (fp32, ~1.11GB, via fastembed)
through the SAME production path ``search_memories``/``semantic_recall``
use (``brain.memory.reranker.build_reranker_provider().rerank(...)``).
``tests/unit/brain/memory/test_reranker.py`` covers the module's own logic
(floor gating, auto-scaling width, latency measurement, etc.) entirely
OFFLINE via ``FakeRerankerProvider`` — this file is the one place that
proves the real model actually loads and scores through the production
factory (score shape, positional alignment, "higher = more relevant"), and
specifically that it needs NO materialize-files workaround (#250 F2a §1 /
acceptance criterion #1) — unlike F1's ``multilingual-e5-large`` embedder
swap, which does hit the onnxruntime external-data-path bug because it
ships sharded external weights. jina ships a single self-contained
``model.onnx`` (``additional_files: []``), so a clean load here is the
proof no such workaround is needed. A shape+sanity+load guard, not a
calibration suite — keep the sample small.

NOTE on ``RERANK_FLOOR``: this file deliberately does NOT assert genuine/
decoy pairs land on either side of ``brain.memory.semantic_recall.
RERANK_FLOOR``. That floor is still calibrated against the OLD MiniLM
score scale as of this commit (F2a inc1 swaps the model only); jina's raw
logits are a DIFFERENT SCALE (see model_tier.py's MODEL_RERANKER comment),
so a floor-separation assertion here would be testing a known, expected
scale mismatch, not a regression. The floor gets re-derived against
jina's own scale by F2a's later daily self-calibration tick (§7) — not by
this test.

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

import math

import pytest

from brain.memory.reranker import build_reranker_provider

pytestmark = [pytest.mark.requires_network, pytest.mark.integration]

_MODEL_ID = "jinaai/jina-reranker-v2-base-multilingual"

# ---------------------------------------------------------------------------
# (query, document) pairs — reused from #88 / test_search_memories_mode.py
# for the shape+ordering check below. These only assert RELATIVE ordering
# (genuine beats decoy), a model-agnostic invariant, never an absolute
# RERANK_FLOOR comparison (see module docstring on why not).
# ---------------------------------------------------------------------------

_GENUINE_PAIRS = [
    ("how do I calm down when everything feels like too much",
     "deep breathing helps when you are feeling anxious"),
    ("quiet evening",
     "a quiet evening with nothing much happening"),
    ("what does Bob like to drink in the morning",
     "Bob always starts his day with a strong cup of black coffee"),
]

_DECOY_PAIRS = [
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
    construction + first-use model load has real, non-trivial cost — the
    jina model is ~1.11GB fp32, larger than the outgoing ~80MB MiniLM) —
    the autouse ``_reset_reranker_provider_cache`` fixture in
    tests/conftest.py clears the PROCESS-wide cache before/after every test
    regardless, so this only avoids re-requesting the provider object
    between the assertions below, not the process-cache reset itself.

    Constructing this (and the first ``.rerank()`` call below, which is
    where fastembed's lazy-load actually downloads/loads the ONNX session)
    is itself the acceptance-criterion-#1 proof: if jina needed the
    external-data materialize workaround F1's e5-large embedder needs, this
    fixture would raise (an onnxruntime "external data path escapes"
    error) instead of returning a working provider."""
    provider = build_reranker_provider()
    assert provider.model_id() == _MODEL_ID, (
        "expected the #250 F2a inc1 jina reranker model id — model_tier.py's "
        "TIER_RERANKER mapping changed out from under this test"
    )
    return provider


def test_real_reranker_loads_and_scores(real_provider) -> None:
    """Acceptance criterion #1: the real jina model loads via the
    production factory (``build_reranker_provider()``) and scores real
    input with NO materialize-files workaround needed (proven implicitly —
    ``real_provider`` above would have raised on construction/first-call
    otherwise). Asserts the API-shape invariants production code depends
    on: `.rerank(query, documents)` returns a `list[float]` (plain floats,
    not wrapper objects), POSITIONALLY aligned with the input document
    order (not pre-sorted), every score finite (no NaN/Inf), and a higher
    score means more relevant."""
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
        assert math.isfinite(score), f"reranker score must be finite, got {score!r}"

    decoy_a_score, genuine_score, decoy_b_score = scores
    print(f"\njina real reranker scores (mixed batch): {list(zip(docs, scores, strict=True))}")

    # NOT pre-sorted: the genuine match stays at index 1 (its input
    # position), not moved to the front despite scoring highest.
    assert genuine_score > decoy_a_score, "higher score = more relevant (genuine beats decoy)"
    assert genuine_score > decoy_b_score, "higher score = more relevant (genuine beats decoy)"


def test_real_reranker_orders_genuine_above_decoy_across_sample(real_provider) -> None:
    """A model-agnostic sanity check (small sample, 3 + 3): the real jina
    model scores every genuine pair higher than every decoy pair's score —
    proving the swapped model still produces a query-conditioned relevance
    signal usable for reranking, WITHOUT asserting any absolute
    RERANK_FLOOR cutoff (that floor is still MiniLM-scaled as of this
    commit — see module docstring). This is NOT a calibration suite; the
    actual floor for jina's scale is F2a's later work (§7)."""
    genuine_scores = []
    for query, doc in _GENUINE_PAIRS:
        (score,) = real_provider.rerank(query, [doc])
        genuine_scores.append((query, doc, score))

    decoy_scores = []
    for query, doc in _DECOY_PAIRS:
        (score,) = real_provider.rerank(query, [doc])
        decoy_scores.append((query, doc, score))

    print("\ngenuine pairs:")
    for query, doc, score in genuine_scores:
        print(f"  {score:8.3f}  q={query!r} doc={doc!r}")
    print("decoy pairs:")
    for query, doc, score in decoy_scores:
        print(f"  {score:8.3f}  q={query!r} doc={doc!r}")

    min_genuine = min(score for _, _, score in genuine_scores)
    max_decoy = max(score for _, _, score in decoy_scores)
    assert min_genuine > max_decoy, (
        f"expected every genuine pair to outscore every decoy pair — "
        f"weakest genuine={min_genuine} vs strongest decoy={max_decoy}"
    )
