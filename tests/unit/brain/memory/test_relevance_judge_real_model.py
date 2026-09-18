"""Real-model validation for ``brain.memory.relevance_judge`` — F2a #250
inc6, the offline daily-calibration relevance judge.

NETWORK-ENABLED + HEAVY DOWNLOAD: loads the actual ``BAAI/bge-reranker-v2-m3``
cross-encoder (~2.1GB ``model.safetensors``) via ``sentence_transformers.
CrossEncoder`` (torch backend, CPU-only install — see pyproject.toml),
through the SAME production path the daily calibration tick uses
(``brain.memory.relevance_judge.build_judge_provider().score(...)``).
``tests/unit/brain/memory/test_relevance_judge.py`` covers the module's own
logic (ambiguous band, sample orchestration, fault isolation, Haiku
tie-break) entirely OFFLINE via ``FakeRelevanceJudgeProvider`` — this file
is the one place that proves the real judge actually loads and scores
through the production factory, and specifically that ``TorchCrossEncoderJudge``
returns a RAW logit (not a pre-sigmoided score — see that class's ``score()``
docstring on the ``activation_fn=None`` trap this test would catch if it
regressed).

Marked BOTH ``@pytest.mark.requires_network`` (opts this test OUT of
``tests/conftest.py``'s autouse fake-judge-provider fixture, so the REAL
``build_judge_provider()`` wiring runs) AND ``@pytest.mark.integration``
(the marker actually deselected by this project's conventional local
pre-check gate, ``-m "not live and not requires_claude_cli and not
integration"``) — mirrors ``test_reranker_real_model.py``'s identical
convention.

Run by hand with network enabled, e.g.:
    uv run pytest tests/unit/brain/memory/test_relevance_judge_real_model.py -m requires_network -v -s
"""

from __future__ import annotations

import math
import time

import pytest

from brain.memory.relevance_judge import (
    TorchCrossEncoderJudge,
    build_judge_provider,
    label_for_score,
)

pytestmark = [pytest.mark.requires_network, pytest.mark.integration]

_MODEL_ID = "BAAI/bge-reranker-v2-m3"

# (query, document) pairs — same shape/spirit as test_reranker_real_model.py's
# genuine/decoy sets, reused here for the judge's own real-model sanity check.
_GENUINE_PAIRS = [
    ("how do I calm down when everything feels like too much",
     "deep breathing helps when you are feeling anxious"),
    ("what does Bob like to drink in the morning",
     "Bob always starts his day with a strong cup of black coffee"),
]

_DECOY_PAIRS = [
    ("how do I calm down when everything feels like too much",
     "the stock market closed higher today on tech earnings"),
    ("what does Bob like to drink in the morning",
     "my cat knocked a glass off the kitchen counter this morning"),
]


@pytest.fixture(scope="module")
def real_judge():
    """Build the REAL judge provider once for this module (torch model
    construction + first-use load has real, non-trivial cost — bge-reranker-
    v2-m3's model.safetensors is ~2.1GB fp32) — the autouse
    `_reset_judge_provider_cache` fixture in tests/conftest.py clears the
    PROCESS-wide cache before/after every test regardless, so this only
    avoids re-requesting the provider object between assertions, not the
    process-cache reset itself.

    Constructing this is itself the acceptance-criterion proof: if the
    torch/sentence-transformers load path were broken, this fixture would
    raise instead of returning a working provider.
    """
    provider = build_judge_provider()
    assert provider.model_id() == _MODEL_ID
    assert isinstance(provider, TorchCrossEncoderJudge), (
        "expected the REAL torch-backed judge, not the autouse fake — "
        "requires_network should have opted this test out of that fixture"
    )
    return provider


def test_real_judge_loads_and_scores(real_judge) -> None:
    """Proof the real judge loads via the production factory
    (`build_judge_provider()`) and scores real input. Asserts the shape
    invariant production code depends on: `.score(query, document)` returns
    a plain finite float."""
    query = "how do I calm down when everything feels like too much"
    doc = "deep breathing helps when you are feeling anxious"

    start = time.monotonic()
    score = real_judge.score(query, doc)
    elapsed = time.monotonic() - start

    print(f"\nbge-reranker-v2-m3 real judge load+score time: {elapsed:.2f}s")
    print(f"bge-reranker-v2-m3 real judge score (genuine pair): {score!r}")

    assert isinstance(score, float), f"expected a plain float, got {type(score)!r} ({score!r})"
    assert math.isfinite(score), f"judge score must be finite, got {score!r}"


def test_real_judge_returns_a_raw_logit_not_a_presigmoided_score(real_judge) -> None:
    """The load-bearing regression guard for the `activation_fn=None` trap
    documented on `TorchCrossEncoderJudge.score`: bge-reranker-v2-m3 is a
    num_labels=1 cross-encoder, whose sentence-transformers default
    activation is Sigmoid — if `score()` ever regressed to
    `activation_fn=None` (or dropped the explicit identity override), every
    score would already be squashed into (0, 1) BEFORE label_for_score
    applies its own sigmoid, and this assertion would catch it: a genuine,
    clearly-relevant pair's raw logit should land measurably outside a
    bare-probability's (0, 1) range on this model (real cross-encoder
    logits routinely run well past +/-1 for confident pairs)."""
    query, doc = _GENUINE_PAIRS[0]
    score = real_judge.score(query, doc)
    print(f"\nbge-reranker-v2-m3 raw logit for a clearly-genuine pair: {score!r}")
    assert not (0.0 < score < 1.0), (
        f"score={score!r} looks pre-sigmoided (landed inside the open unit interval) — "
        "activation_fn must be forcing an identity pass-through, not the model's default Sigmoid"
    )


def test_real_judge_orders_genuine_above_decoy_across_sample(real_judge) -> None:
    """Model-agnostic sanity check: the real judge scores every genuine
    pair higher (raw logit) than every decoy pair — proving it produces a
    query-conditioned relevance signal usable for judging, independent of
    any particular ambiguous-band placement."""
    genuine_scores = [real_judge.score(q, d) for q, d in _GENUINE_PAIRS]
    decoy_scores = [real_judge.score(q, d) for q, d in _DECOY_PAIRS]

    print("\ngenuine pair scores:", genuine_scores)
    print("decoy pair scores:", decoy_scores)

    assert min(genuine_scores) > max(decoy_scores), (
        f"expected every genuine pair to outscore every decoy pair — "
        f"weakest genuine={min(genuine_scores)} vs strongest decoy={max(decoy_scores)}"
    )


def test_real_judge_labels_genuine_pair_relevant_via_label_for_score(real_judge) -> None:
    """End-to-end (real judge -> label_for_score): a strongly genuine pair's
    raw logit, run through the actual sigmoid + ambiguous-band logic, lands
    on "relevant".

    Uses `_GENUINE_PAIRS[1]` specifically (measured well positive, +3.06 in
    manual verification) rather than `_GENUINE_PAIRS[0]` — bge-reranker-v2-m3
    is a DIFFERENT model from jina (this pair set is reused from jina's own
    real-model test), and its raw-logit zero-crossing is its own learned
    decision boundary, not something this test should assume matches
    another model's judgment of the same pair (mirrors
    test_reranker_real_model.py's module docstring on why it never asserts
    an absolute RERANK_FLOOR-relative outcome for a specific pair either —
    `_GENUINE_PAIRS[0]` scored measurably NEGATIVE on this model despite
    reading as "genuine" to a human, which is exactly the kind of
    model-specific calibration fact this file's ordering-only test above
    is deliberately built to not depend on). This test only asserts the
    mechanical label -> sigmoid -> band pipeline lands where the raw score
    unambiguously implies, for a pair confirmed to be unambiguously on one
    side."""
    query, doc = _GENUINE_PAIRS[1]
    score = real_judge.score(query, doc)
    label, is_ambiguous = label_for_score(score)
    print(f"\nbge-reranker-v2-m3 raw logit for the strongly-genuine pair: {score!r}")
    assert label == "relevant"
    assert is_ambiguous is False, "a strongly positive logit should be well outside the ambiguous band"
